// M28: Ampere software producer/consumer warps, named-barrier ring buffers.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "ampere_mma_primitives.cuh"

namespace {
using namespace llm_gpu_kernels_sm86;
constexpr int BM=64, BN=128, MF=2, NF=8, Group=2;
constexpr int Elements=(BM+BN)*BK;

template<int Participants,int Base,int Stages>
__device__ __forceinline__ void arrive(int slot) {
  // Immediate IDs avoid reserving all 16 hardware barriers for a dynamic ID.
  if (slot==0) asm volatile("bar.arrive %0, %1;" :: "n"(Base),"n"(Participants):"memory");
  else if (slot==1) asm volatile("bar.arrive %0, %1;" :: "n"(Base+1),"n"(Participants):"memory");
  else if constexpr (Stages==3)
    asm volatile("bar.arrive %0, %1;" :: "n"(Base+2),"n"(Participants):"memory");
}
template<int Participants,int Base,int Stages>
__device__ __forceinline__ void wait(int slot) {
  if (slot==0) asm volatile("bar.sync %0, %1;" :: "n"(Base),"n"(Participants):"memory");
  else if (slot==1) asm volatile("bar.sync %0, %1;" :: "n"(Base+1),"n"(Participants):"memory");
  else if constexpr (Stages==3)
    asm volatile("bar.sync %0, %1;" :: "n"(Base+2),"n"(Participants):"memory");
}
template<int ProducerThreads>
__device__ __forceinline__ void produce(__nv_bfloat16* smem,
    const __nv_bfloat16* a, const __nv_bfloat16* w,
    int m,int n,int k,int row_start,int col_start,int k_start) {
  for (int v=threadIdx.x;v<BM*4;v+=ProducerThreads) {
    const int row=v/4, kk=(v%4)*8;
    const bool valid=row_start+row<m && k_start+kk<k;
    const auto* src=valid ? a+static_cast<int64_t>(row_start+row)*k+k_start+kk : a;
    copy16<true>(smem+index<true>(row,kk),src,valid);
  }
  auto* weight=smem+BM*BK;
  for (int v=threadIdx.x;v<BN*4;v+=ProducerThreads) {
    const int row=v/4, kk=(v%4)*8;
    const bool valid=col_start+row<n && k_start+kk<k;
    const auto* src=valid ? w+static_cast<int64_t>(col_start+row)*k+k_start+kk : w;
    copy16<true>(weight+index<true>(row,kk),src,valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  // Producers finish their copies before contributing to the ready barrier.
  // Consumer warps may still compute an earlier slot during this wait.
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template<int ProducerWarps,int Stages>
__global__ void warp_specialized_mma(const __nv_bfloat16* a,const __nv_bfloat16* w,
                                    __nv_bfloat16* out,int m,int n,int k) {
  constexpr int ProducerThreads=ProducerWarps*32;
  constexpr int Participants=ProducerThreads+128;
  const int row_start=blockIdx.x*BM,col_start=blockIdx.y*BN;
  const int iterations=(k+BK-1)/BK;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  if (threadIdx.x<ProducerThreads) {
    for (int tile=0;tile<iterations;++tile) {
      const int slot=tile%Stages;
      // free barrier: old consumers arrive; new producers wait for reuse.
      if (tile>=Stages) wait<Participants,Stages,Stages>(slot);
      produce<ProducerThreads>(smem+slot*Elements,a,w,m,n,k,
                                row_start,col_start,tile*BK);
      // ready barrier: producers arrive without waiting, consumers wait.
      arrive<Participants,0,Stages>(slot);
    }
  } else {
    const int local=threadIdx.x-ProducerThreads,lane=local%32,warp=local/32;
    const int warp_row=(warp/2)*32,warp_col=(warp%2)*64;
    float acc[MF][NF][4]={};
    for (int tile=0;tile<iterations;++tile) {
      const int slot=tile%Stages;
      wait<Participants,0,Stages>(slot);
      const auto* current=smem+slot*Elements;
      const auto* weight=current+BM*BK;
      #pragma unroll
      for (int kk=0;kk<BK;kk+=16) {
        uint32_t ar[MF][4];
        #pragma unroll
        for (int mi=0;mi<MF;++mi)
          load_a<true>(ar[mi],current,lane,warp_row+mi*16,kk);
        #pragma unroll
        for (int start=0;start<NF;start+=Group) {
          uint32_t br[Group][2];
          #pragma unroll
          for (int ni=0;ni<Group;++ni)
            load_b<true>(br[ni],weight,lane,warp_col+(start+ni)*8,kk);
          #pragma unroll
          for (int mi=0;mi<MF;++mi) {
            #pragma unroll
            for (int ni=0;ni<Group;++ni) mma(acc[mi][start+ni],ar[mi],br[ni]);
          }
        }
      }
      // Only signal a free barrier if this slot will actually be reused.
      // This balances every arrive with a producer sync, including the tail.
      if (tile+Stages<iterations) arrive<Participants,Stages,Stages>(slot);
    }
    #pragma unroll
    for (int mi=0;mi<MF;++mi) {
      #pragma unroll
      for (int ni=0;ni<NF;++ni) {
        #pragma unroll
        for (int e=0;e<4;++e) {
          const int row=row_start+warp_row+mi*16+lane/4+(e/2)*8;
          const int col=col_start+warp_col+ni*8+(lane%4)*2+e%2;
          if (row<m && col<n)
            out[static_cast<int64_t>(row)*n+col]=__float2bfloat16_rn(acc[mi][ni][e]);
        }
      }
    }
  }
}

class WarpSpecializedBf16MmaPlan {
  torch::Tensor a_,w_,out_;
  int device_,m_,n_,k_,producers_,stages_;
  template<int ProducerWarps,int Stages>
  void launch() {
    const dim3 grid((m_+BM-1)/BM,(n_+BN-1)/BN);
    const auto stream=c10::cuda::getCurrentCUDAStream(device_);
    warp_specialized_mma<ProducerWarps,Stages><<<grid,(ProducerWarps+4)*32,Elements*2*Stages,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
 public:
  WarpSpecializedBf16MmaPlan(torch::Tensor a,torch::Tensor w,torch::Tensor out,
                            int producers,int stages)
      :a_(a),w_(w),out_(out),producers_(producers),stages_(stages) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA tensors required");
    device_=a.get_device();
    for (const auto& t:{a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 required");
    TORCH_CHECK(a.size(0)<=INT_MAX-BM && a.size(1)<=INT_MAX-BK && w.size(0)<=INT_MAX-BN,
                "dimensions exceed int32 bound");
    m_=a.size(0);n_=w.size(0);k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_,"invalid shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}),"output shape mismatch");
    TORCH_CHECK((producers_==1 || producers_==2) && (stages_==2 || stages_==3),
                "producer warps must be 1/2 and stages 2/3");
    for (const auto& t:{a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"aligned inputs required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output must not alias inputs");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6,"M28 targets SM86");
    TORCH_CHECK(Elements*2*stages_<=props.sharedMemPerBlock,"shared memory limit exceeded");
    TORCH_CHECK((n_+BN-1)/BN<=props.maxGridSize[1],"N grid exceeds device limit");
  }
  void run() {
    if (m_==0) return;
    c10::cuda::CUDAGuard guard(device_);
    if (producers_==1) {
      if (stages_==2) launch<1,2>(); else launch<1,3>();
    } else {
      if (stages_==2) launch<2,2>(); else launch<2,3>();
    }
  }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<WarpSpecializedBf16MmaPlan>(m,"WarpSpecializedBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int,int>())
      .def("run",&WarpSpecializedBf16MmaPlan::run);
}
