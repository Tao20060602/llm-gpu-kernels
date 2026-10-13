// M33: factorial cp.async cache-policy experiment, fixed M32 GroupM16.
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
  // Retain the BK32 mapping and aligned 16-byte vectors.
  return (row*Depth+col)^((row&7)*8);
}

template<int CacheMode>
__device__ __forceinline__ void cache_copy16(__nv_bfloat16* dst,
                                            const __nv_bfloat16* src,bool valid) {
  const unsigned p=address(dst);
  const int bytes=valid?16:0;
  if constexpr(CacheMode==0)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 ::"r"(p),"l"(src),"r"(bytes):"memory");
  else if constexpr(CacheMode==1)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 ::"r"(p),"l"(src),"r"(bytes):"memory");
  else if constexpr(CacheMode==2)
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;"
                 ::"r"(p),"l"(src),"r"(bytes):"memory");
  else
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;"
                 ::"r"(p),"l"(src),"r"(bytes):"memory");
}

template<int Depth,int CacheMode>
__device__ __forceinline__ void stage_depth(__nv_bfloat16* smem,
    const __nv_bfloat16* a,const __nv_bfloat16* w,int m,int n,int k,
    int row_start,int col_start,int k_start) {
  constexpr int Vectors=Depth/8;
  for(int v=threadIdx.x;v<BM*Vectors;v+=128) {
    const int row=v/Vectors,kk=(v%Vectors)*8;
    const bool valid=row_start+row<m && k_start+kk<k;
    const auto* src=valid?a+static_cast<int64_t>(row_start+row)*k+k_start+kk:a;
    cache_copy16<CacheMode>(smem+depth_index<Depth>(row,kk),src,valid);
  }
  auto* weight=smem+BM*Depth;
  for(int v=threadIdx.x;v<BN*Vectors;v+=128) {
    const int row=v/Vectors,kk=(v%Vectors)*8;
    const bool valid=col_start+row<n && k_start+kk<k;
    const auto* src=valid?w+static_cast<int64_t>(col_start+row)*k+k_start+kk:w;
    cache_copy16<CacheMode>(weight+depth_index<Depth>(row,kk),src,valid);
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

template<int CacheMode>
__global__ void cache_policy_mma(const __nv_bfloat16* a,const __nv_bfloat16* w,
                          __nv_bfloat16* out,int m,int n,int k) {
  constexpr int GroupM=16,Depth=32,Stages=2,Elements=(BM+BN)*Depth;
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int warp_row=(warp/2)*32,warp_col=(warp%2)*64;
  const int m_tiles=(m+BM-1)/BM,n_tiles=(n+BN-1)/BN;
  const int linear=static_cast<int>(blockIdx.x);
  int tile_m,tile_n;
  if constexpr(GroupM==0) {
    tile_m=linear%m_tiles;
    tile_n=linear/m_tiles;
  } else {
    const int group_size=GroupM*n_tiles;
    const int first_m=(linear/group_size)*GroupM;
    const int remaining=m_tiles-first_m;
    const int active_m=remaining<GroupM?remaining:GroupM;
    const int local=linear%group_size;
    tile_m=first_m+local%active_m;
    tile_n=local/active_m;
  }
  const int row_start=tile_m*BM,col_start=tile_n*BN;
  const int iterations=(k+Depth-1)/Depth;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[MF][NF][4]={};
  #pragma unroll
  for(int tile=0;tile<Stages-1;++tile)
    if(tile<iterations)
      stage_depth<Depth,CacheMode>(smem+tile*Elements,a,w,m,n,k,row_start,col_start,tile*Depth);
  for(int tile=0;tile<iterations;++tile) {
    auto* current=smem+(tile%Stages)*Elements;
    const int future=tile+Stages-1;
    if(future<iterations) {
      stage_depth<Depth,CacheMode>(smem+(future%Stages)*Elements,
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

class CachePolicyBf16MmaPlan {
  torch::Tensor a_,w_,out_;
  int device_,m_,n_,k_,cache_mode_,tiles_;
  template<int CacheMode>
  void launch() {
    const auto stream=c10::cuda::getCurrentCUDAStream(device_);
    cache_policy_mma<CacheMode><<<tiles_,128,(BM+BN)*32*2*2,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
 public:
  CachePolicyBf16MmaPlan(torch::Tensor a,torch::Tensor w,torch::Tensor out,int cache_mode)
      :a_(a),w_(w),out_(out),cache_mode_(cache_mode) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA tensors required");
    device_=a.get_device();
    for(const auto& t:{a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 required");
    TORCH_CHECK(a.size(0)<=INT_MAX-BM && a.size(1)<=INT_MAX-32 && w.size(0)<=INT_MAX-BN,
                "dimensions exceed int32 bound");
    m_=a.size(0);n_=w.size(0);k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_,"invalid shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}),"output shape mismatch");
    TORCH_CHECK(cache_mode_>=0 && cache_mode_<=3,
                "invalid cache_mode");
    for(const auto& t:{a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"aligned input bases required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output must not alias input");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6,"M33 targets SM86");
    TORCH_CHECK((BM+BN)*32*2*2<=props.sharedMemPerBlock,"shared memory limit exceeded");
    const int64_t tiles=static_cast<int64_t>((m_+BM-1)/BM)*((n_+BN-1)/BN);
    TORCH_CHECK(tiles<=props.maxGridSize[0] && tiles<=INT_MAX,"linear grid exceeds device limit");
    tiles_=static_cast<int>(tiles);
  }
  void run() {
    if(m_==0)return;
    c10::cuda::CUDAGuard guard(device_);
    switch(cache_mode_) {
      case 0:launch<0>();break;
      case 1:launch<1>();break;
      case 2:launch<2>();break;
      case 3:launch<3>();break;
    }
  }
};
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<CachePolicyBf16MmaPlan>(m,"CachePolicyBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int>())
      .def("run",&CachePolicyBf16MmaPlan::run);
}
