// Compact SM86 GQA: direct-P only; complete history tiles use unmasked scaling.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <math_constants.h>
#include <vector>
#include "ampere_mma_primitives.cuh"

namespace {
using namespace llm_gpu_kernels_sm86;
constexpr int Queries=64,Keys=32,Dim=128,Threads=128;
constexpr int QElements=Queries*Dim,KVElements=2*Keys*Dim;
template<int Stages>
constexpr int SharedBytes=(QElements+Stages*KVElements)*2;
struct Segments {
  const __nv_bfloat16* k[4];
  const __nv_bfloat16* v[4];
  int length[4];
  int64_t k_stride[4],v_stride[4];
};
__device__ __forceinline__ int wide_index(int row,int col) {
  return (row*Dim+col)^((row&7)*8);
}
__device__ __forceinline__ void async_copy(__nv_bfloat16* dst,
    const __nv_bfloat16* src,bool valid) {
  const unsigned p=address(dst);
  const int bytes=valid?16:0;
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;"
      ::"r"(p),"l"(src),"r"(bytes):"memory");
}
__device__ __forceinline__ void issue_kv(__nv_bfloat16* dst,Segments s,
    int segment,int start,int kv_head,const __nv_bfloat16* fallback) {
  for(int vector=threadIdx.x;vector<Keys*16;vector+=Threads) {
    const int row=vector/16,col=(vector%16)*8;
    const bool valid=start+row<s.length[segment];
    const auto* kp=valid?s.k[segment]+static_cast<int64_t>(start+row)*s.k_stride[segment]+kv_head*Dim+col:fallback;
    const auto* vp=valid?s.v[segment]+static_cast<int64_t>(start+row)*s.v_stride[segment]+kv_head*Dim+col:fallback;
    async_copy(dst+wide_index(row,col),kp,valid);
    async_copy(dst+Keys*Dim+wide_index(row,col),vp,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void load_wide_a(uint32_t (&r)[4],
    const __nv_bfloat16* ptr,int row,int kk,int lane) {
  const unsigned p=address(ptr+wide_index(row+lane%16,kk+(lane/16)*8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(p):"memory");
}
__device__ __forceinline__ void load_wide_b(uint32_t (&r)[2],
    const __nv_bfloat16* ptr,int row,int kk,int lane) {
  const unsigned p=address(ptr+wide_index(row+lane%8,kk+((lane/8)%2)*8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      :"=r"(r[0]),"=r"(r[1]):"r"(p):"memory");
}
__device__ __forceinline__ void load_value_b(uint32_t (&r)[2],
    const __nv_bfloat16* ptr,int kk,int col,int lane) {
  const unsigned p=address(ptr+wide_index(kk+lane%16,col));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
      :"=r"(r[0]),"=r"(r[1]):"r"(p):"memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float low,float high) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(low))) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(high)))<<16);
}
__device__ __forceinline__ float row_max(float x) {
  x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1));
  return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2));
}
__device__ __forceinline__ float row_sum(float x) {
  x+=__shfl_xor_sync(0xffffffff,x,1);
  return x+__shfl_xor_sync(0xffffffff,x,2);
}

template<int Stages>
__global__ void compact_gqa_forward(const __nv_bfloat16* q,Segments segments,
    __nv_bfloat16* out,int tokens,int64_t q_stride) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int head=blockIdx.y,kv_head=head/4,start_q=blockIdx.x*Queries;
  const int local_row=warp*16+lane/4;
  const int query[2]={start_q+local_row,start_q+local_row+8};
  extern __shared__ __align__(32) __nv_bfloat16 shared[];
  auto* sq=shared;
  auto* skv=shared+QElements;
  for(int vector=threadIdx.x;vector<Queries*16;vector+=Threads) {
    const int row=vector/16,col=(vector%16)*8;
    const bool valid=start_q+row<tokens;
    const auto* src=valid?q+static_cast<int64_t>(start_q+row)*q_stride+head*Dim+col:q;
    async_copy(sq+wide_index(row,col),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  float output[16][4]={};
  float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},normalizer[2]={0,0};
  #pragma unroll
  for(int segment=0;segment<4;++segment) {
    int visible=segments.length[segment];
    if(segment==3 && visible>start_q+Queries)visible=start_q+Queries;
    const int tiles=(visible+Keys-1)/Keys;
    if(tiles==0)continue;
    issue_kv(skv,segments,segment,0,kv_head,q);
    for(int tile=0;tile<tiles;++tile) {
      auto* current=skv+(tile%Stages)*KVElements;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if constexpr(Stages==2) {
        if(tile+1<tiles)
          issue_kv(skv+((tile+1)%Stages)*KVElements,segments,segment,(tile+1)*Keys,kv_head,q);
      }
      float score[4][4]={};
      #pragma unroll
      for(int kk=0;kk<Dim;kk+=16) {
        uint32_t ar[4];
        load_wide_a(ar,sq,warp*16,kk,lane);
        #pragma unroll
        for(int ni=0;ni<4;++ni) {
          uint32_t br[2];
          load_wide_b(br,current,ni*8,kk,lane);
          mma(score[ni],ar,br);
        }
      }
      // Uniform tile validity is derived from bounds, not a caller option.
      const bool full_history=segment<3 && start_q+Queries<=tokens &&
          tile*Keys+Keys<=segments.length[segment];
      if(full_history) {
        #pragma unroll
        for(int ni=0;ni<4;++ni) {
          #pragma unroll
          for(int e=0;e<4;++e)score[ni][e]*=0.08838834764831845f;
        }
      } else {
        #pragma unroll
        for(int ni=0;ni<4;++ni) {
          #pragma unroll
          for(int e=0;e<4;++e) {
            const int r=e/2,key=tile*Keys+ni*8+(lane%4)*2+e%2;
            const bool valid=query[r]<tokens && key<segments.length[segment] &&
                (segment!=3 || key<=query[r]);
            score[ni][e]=valid?score[ni][e]*0.08838834764831845f:-CUDART_INF_F;
          }
        }
      }
      float local_max[2]={-CUDART_INF_F,-CUDART_INF_F};
      #pragma unroll
      for(int ni=0;ni<4;++ni) {
        #pragma unroll
        for(int e=0;e<4;++e)local_max[e/2]=fmaxf(local_max[e/2],score[ni][e]);
      }
      float alpha[2];
      #pragma unroll
      for(int r=0;r<2;++r) {
        const float next=fmaxf(maximum[r],row_max(local_max[r]));
        alpha[r]=maximum[r]==-CUDART_INF_F?0.0f:__expf(maximum[r]-next);
        maximum[r]=next;
      }
      float partial_sum[2]={0,0};
      #pragma unroll
      for(int ni=0;ni<4;++ni) {
        #pragma unroll
        for(int e=0;e<4;++e) {
          const int r=e/2;
          const float probability=score[ni][e]==-CUDART_INF_F?0.0f:__expf(score[ni][e]-maximum[r]);
          score[ni][e]=probability;
          partial_sum[r]+=probability;
        }
      }
      #pragma unroll
      for(int r=0;r<2;++r)
        normalizer[r]=normalizer[r]*alpha[r]+row_sum(partial_sum[r]);
      #pragma unroll
      for(int ni=0;ni<16;++ni) {
        #pragma unroll
        for(int e=0;e<4;++e)output[ni][e]*=alpha[e/2];
      }
      #pragma unroll
      for(int kk=0;kk<Keys;kk+=16) {
        uint32_t ar[4];
        const int ni=kk/8;
        ar[0]=pack_bf16(score[ni][0],score[ni][1]);
        ar[1]=pack_bf16(score[ni][2],score[ni][3]);
        ar[2]=pack_bf16(score[ni+1][0],score[ni+1][1]);
        ar[3]=pack_bf16(score[ni+1][2],score[ni+1][3]);
        #pragma unroll
        for(int ni=0;ni<16;++ni) {
          uint32_t br[2];
          load_value_b(br,current+Keys*Dim,kk,ni*8,lane);
          mma(output[ni],ar,br);
        }
      }
      __syncthreads(); // All current K/V and P readers finish before slot reuse.
      if constexpr(Stages==1)
        if(tile+1<tiles)issue_kv(skv,segments,segment,(tile+1)*Keys,kv_head,q);
    }
  }
  #pragma unroll
  for(int ni=0;ni<16;++ni) {
    #pragma unroll
    for(int r=0;r<2;++r) {
      if(query[r]<tokens) {
        const int64_t offset=(static_cast<int64_t>(query[r])*32+head)*Dim+ni*8+(lane%4)*2;
        const float inv=1.0f/normalizer[r];
        *reinterpret_cast<__nv_bfloat162*>(out+offset)=__floats2bfloat162_rn(output[ni][2*r]*inv,output[ni][2*r+1]*inv);
      }
    }
  }
}

class CompactCudaGqaPlan {
  torch::Tensor q_,out_;
  std::vector<torch::Tensor> ks_,vs_;
  Segments segments_{};
  int device_,tokens_,stages_;
  template<int Stages>
  void initialize(const cudaDeviceProp& props) {
    constexpr int Bytes=SharedBytes<Stages>;
    TORCH_CHECK(Bytes<=props.sharedMemPerBlockOptin,"shared memory opt-in limit exceeded");
    if(Bytes>props.sharedMemPerBlock)
      C10_CUDA_CHECK(cudaFuncSetAttribute(compact_gqa_forward<Stages>,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
  }
  template<int Stages>
  void launch() {
    const dim3 grid((tokens_+Queries-1)/Queries,32);
    const auto stream=c10::cuda::getCurrentCUDAStream(device_);
    compact_gqa_forward<Stages><<<grid,Threads,SharedBytes<Stages>,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_.data_ptr()),segments_,
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),tokens_,q_.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  void validate_input(const torch::Tensor& t,int heads) {
    TORCH_CHECK(t.is_cuda() && t.get_device()==device_ && t.scalar_type()==at::kBFloat16,
                "same-device CUDA BF16 required");
    TORCH_CHECK(t.dim()==3 && t.size(1)==heads && t.size(2)==Dim,"invalid input shape");
    TORCH_CHECK(t.size(0)<=INT_MAX-Queries,"token length exceeds int32 bound");
    TORCH_CHECK(t.stride(1)==Dim && t.stride(2)==1 && t.stride(0)>=heads*Dim && t.stride(0)%8==0,
                "inner contiguous, aligned token padding required");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"16-byte base alignment required");
    TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),"output aliases input");
  }
 public:
  CompactCudaGqaPlan(torch::Tensor q,std::vector<torch::Tensor> ks,
      std::vector<torch::Tensor> vs,torch::Tensor out,int stages)
      :q_(q),out_(out),ks_(std::move(ks)),vs_(std::move(vs)),stages_(stages) {
    TORCH_CHECK(q_.is_cuda() && out_.is_cuda(),"CUDA required");
    device_=q_.get_device();
    TORCH_CHECK(out_.get_device()==device_ && out_.scalar_type()==at::kBFloat16 &&
                out_.is_contiguous() && out_.sizes()==q_.sizes(),"invalid output");
    TORCH_CHECK(ks_.size()==4 && vs_.size()==4,"four KV pairs required");
    TORCH_CHECK(stages_==1 || stages_==2,"invalid configuration");
    validate_input(q_,32);tokens_=q_.size(0);
    for(int segment=0;segment<4;++segment) {
      validate_input(ks_[segment],8);validate_input(vs_[segment],8);
      TORCH_CHECK(ks_[segment].sizes()==vs_[segment].sizes(),"K/V shapes differ");
      segments_.k[segment]=reinterpret_cast<const __nv_bfloat16*>(ks_[segment].data_ptr());
      segments_.v[segment]=reinterpret_cast<const __nv_bfloat16*>(vs_[segment].data_ptr());
      segments_.length[segment]=ks_[segment].size(0);
      segments_.k_stride[segment]=ks_[segment].stride(0);segments_.v_stride[segment]=vs_[segment].stride(0);
    }
    TORCH_CHECK(segments_.length[3]==tokens_,"current KV length must equal query length");
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6,"M36 targets SM86");
    if(stages_==1) {
      initialize<1>(props);
    } else {
      initialize<2>(props);
    }
  }
  void run() {
    if(tokens_==0)return;
    c10::cuda::CUDAGuard guard(device_);
    if(stages_==1) {
      launch<1>();
    } else {
      launch<2>();
    }
  }
};
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<CompactCudaGqaPlan>(m,"CompactCudaGqaPlan")
      .def(pybind11::init<torch::Tensor,std::vector<torch::Tensor>,std::vector<torch::Tensor>,torch::Tensor,int>())
      .def("run",&CompactCudaGqaPlan::run);
}
