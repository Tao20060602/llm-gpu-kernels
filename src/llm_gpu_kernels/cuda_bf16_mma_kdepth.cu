// M30: vary K depth only; CTA64x128, four warps, G2 B-register prefetch.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "ampere_mma_primitives.cuh"

namespace {
using namespace llm_gpu_kernels_sm86;
constexpr int BM=64, BN=128, MF=2, NF=8, Group=2, Chunks=NF/Group;

template<int Depth>
__device__ __forceinline__ int depth_index(int row,int col) {
  // Preserve each aligned 16-byte vector. XOR is a bijection for both depths.
  // BK32 keeps the M29 mapping; BK64 mixes only within its physical row.
  return (row*Depth+col)^((row&7)*8);
}

template<int Depth>
__device__ __forceinline__ void stage_depth(__nv_bfloat16* smem,
    const __nv_bfloat16* a,const __nv_bfloat16* w,int m,int n,int k,
    int row_start,int col_start,int k_start) {
  constexpr int Vectors=Depth/8;
  for(int v=threadIdx.x;v<BM*Vectors;v+=128) {
    const int row=v/Vectors,kk=(v%Vectors)*8;
    const bool valid=row_start+row<m && k_start+kk<k;
    const auto* src=valid?a+static_cast<int64_t>(row_start+row)*k+k_start+kk:a;
    copy16<true>(smem+depth_index<Depth>(row,kk),src,valid);
  }
  auto* weight=smem+BM*Depth;
  for(int v=threadIdx.x;v<BN*Vectors;v+=128) {
    const int row=v/Vectors,kk=(v%Vectors)*8;
    const bool valid=col_start+row<n && k_start+kk<k;
    const auto* src=valid?w+static_cast<int64_t>(col_start+row)*k+k_start+kk:w;
    copy16<true>(weight+depth_index<Depth>(row,kk),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Depth>
__device__ __forceinline__ void load_depth_a(uint32_t (&r)[4],
    const __nv_bfloat16* a,int lane,int row,int kk) {
  const unsigned p=address(a+depth_index<Depth>(row+lane%16,kk+(lane/16)*8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(p):"memory");
}
template<int Depth>
__device__ __forceinline__ void load_depth_b(uint32_t (&r)[2],
    const __nv_bfloat16* w,int lane,int row,int kk) {
  const unsigned p=address(w+depth_index<Depth>(row+lane%8,kk+((lane/8)%2)*8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      : "=r"(r[0]),"=r"(r[1]):"r"(p):"memory");
}

template<int Depth,int Stages>
__global__ void kdepth_mma(const __nv_bfloat16* a,const __nv_bfloat16* w,
                          __nv_bfloat16* out,int m,int n,int k) {
  constexpr int Elements=(BM+BN)*Depth;
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int warp_row=(warp/2)*32,warp_col=(warp%2)*64;
  const int row_start=blockIdx.x*BM,col_start=blockIdx.y*BN;
  const int iterations=(k+Depth-1)/Depth;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[MF][NF][4]={};
  #pragma unroll
  for(int tile=0;tile<Stages-1;++tile)
    if(tile<iterations)
      stage_depth<Depth>(smem+tile*Elements,a,w,m,n,k,row_start,col_start,tile*Depth);
  for(int tile=0;tile<iterations;++tile) {
    auto* current=smem+(tile%Stages)*Elements;
    const int future=tile+Stages-1;
    if(future<iterations) {
      stage_depth<Depth>(smem+(future%Stages)*Elements,
                        a,w,m,n,k,row_start,col_start,future*Depth);
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1):"memory");
    } else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const auto* weight=current+BM*Depth;
    #pragma unroll
    for(int kk=0;kk<Depth;kk+=16) {
      uint32_t ar[MF][4],br[2][Group][2];
      #pragma unroll
      for(int mi=0;mi<MF;++mi)
        load_depth_a<Depth>(ar[mi],current,lane,warp_row+mi*16,kk);
      #pragma unroll
      for(int ni=0;ni<Group;++ni)
        load_depth_b<Depth>(br[0][ni],weight,lane,warp_col+ni*8,kk);
      #pragma unroll
      for(int chunk=0;chunk<Chunks;++chunk) {
        const int slot=chunk&1,next=slot^1;
        if(chunk+1<Chunks) {
          #pragma unroll
          for(int ni=0;ni<Group;++ni)
            load_depth_b<Depth>(br[next][ni],weight,lane,
                               warp_col+((chunk+1)*Group+ni)*8,kk);
        }
        #pragma unroll
        for(int mi=0;mi<MF;++mi) {
          #pragma unroll
          for(int ni=0;ni<Group;++ni)
            mma(acc[mi][chunk*Group+ni],ar[mi],br[slot][ni]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int mi=0;mi<MF;++mi) {
    #pragma unroll
    for(int ni=0;ni<NF;++ni) {
      #pragma unroll
      for(int e=0;e<4;++e) {
        const int row=row_start+warp_row+mi*16+lane/4+(e/2)*8;
        const int col=col_start+warp_col+ni*8+(lane%4)*2+e%2;
        if(row<m && col<n)
          out[static_cast<int64_t>(row)*n+col]=__float2bfloat16_rn(acc[mi][ni][e]);
      }
    }
  }
}

class KDepthBf16MmaPlan {
  torch::Tensor a_,w_,out_;
  int device_,m_,n_,k_,bk_,stages_;
  template<int Depth,int Stages>
  void initialize(const cudaDeviceProp& props) {
    constexpr int Bytes=(BM+BN)*Depth*2*Stages;
    TORCH_CHECK(Bytes<=props.sharedMemPerBlockOptin,"shared memory opt-in limit exceeded");
    if(Bytes>props.sharedMemPerBlock)
      C10_CUDA_CHECK(cudaFuncSetAttribute(kdepth_mma<Depth,Stages>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes));
  }
  template<int Depth,int Stages>
  void launch() {
    const dim3 grid((m_+BM-1)/BM,(n_+BN-1)/BN);
    const auto stream=c10::cuda::getCurrentCUDAStream(device_);
    kdepth_mma<Depth,Stages><<<grid,128,(BM+BN)*Depth*2*Stages,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
 public:
  KDepthBf16MmaPlan(torch::Tensor a,torch::Tensor w,torch::Tensor out,int bk,int stages)
      :a_(a),w_(w),out_(out),bk_(bk),stages_(stages) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA tensors required");
    device_=a.get_device();
    for(const auto& t:{a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 required");
    TORCH_CHECK(a.size(0)<=INT_MAX-BM && a.size(1)<=INT_MAX-64 && w.size(0)<=INT_MAX-BN,
                "dimensions exceed int32 bound");
    m_=a.size(0);n_=w.size(0);k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_,"invalid shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}),"output shape mismatch");
    TORCH_CHECK((bk_==32 || bk_==64) && (stages_==2 || stages_==3),"invalid bk/stages");
    for(const auto& t:{a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"aligned input bases required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output must not alias input");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6,"M30 targets SM86");
    TORCH_CHECK((n_+BN-1)/BN<=props.maxGridSize[1],"N grid exceeds device limit");
    if(bk_==32) {
      if(stages_==2) initialize<32,2>(props);else initialize<32,3>(props);
    } else {
      if(stages_==2) initialize<64,2>(props);else initialize<64,3>(props);
    }
  }
  void run() {
    if(m_==0)return;
    c10::cuda::CUDAGuard guard(device_);
    if(bk_==32) {
      if(stages_==2) launch<32,2>();else launch<32,3>();
    } else {
      if(stages_==2) launch<64,2>();else launch<64,3>();
    }
  }
};
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<KDepthBf16MmaPlan>(m,"KDepthBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int,int>())
      .def("run",&KDepthBf16MmaPlan::run);
}
