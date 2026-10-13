// M29: operand register double buffering, fixed CTA64x128/BK32/swizzle.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "ampere_mma_primitives.cuh"

namespace {
using namespace llm_gpu_kernels_sm86;
constexpr int BM=64,BN=128,MF=2,NF=8,Elements=(BM+BN)*BK;

template<int Group,int Mode,int Stages>
__global__ void register_pipeline_mma(const __nv_bfloat16* a,const __nv_bfloat16* w,
                                     __nv_bfloat16* out,int m,int n,int k) {
  constexpr int Chunks=NF/Group;
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int warp_row=(warp/2)*32,warp_col=(warp%2)*64;
  const int row_start=blockIdx.x*BM,col_start=blockIdx.y*BN;
  const int iterations=(k+BK-1)/BK;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[MF][NF][4]={};
  #pragma unroll
  for (int tile=0;tile<Stages-1;++tile)
    if (tile<iterations)
      stage<BM,BN,true,true>(smem+tile*Elements,a,w,m,n,k,row_start,col_start,tile*BK);
  for (int tile=0;tile<iterations;++tile) {
    auto* current=smem+(tile%Stages)*Elements;
    const int future=tile+Stages-1;
    if (future<iterations) {
      stage<BM,BN,true,true>(smem+(future%Stages)*Elements,
                            a,w,m,n,k,row_start,col_start,future*BK);
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1):"memory");
    } else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const auto* weight=current+BM*BK;

    if constexpr (Mode==2) {
      // A/B banks span all N chunks and both K16 halves of the current K32.
      // Shared-slot transitions remain at the existing CTA barriers.
      uint32_t ar[2][MF][4],br[2][Group][2];
      #pragma unroll
      for (int mi=0;mi<MF;++mi) load_a<true>(ar[0][mi],current,lane,warp_row+mi*16,0);
      #pragma unroll
      for (int ni=0;ni<Group;++ni) load_b<true>(br[0][ni],weight,lane,warp_col+ni*8,0);
      #pragma unroll
      for (int work=0;work<2*Chunks;++work) {
        const int slot=work&1,next=slot^1;
        const int kk=(work/Chunks)*16,start=(work%Chunks)*Group;
        if (work+1<2*Chunks) {
          const int next_k=((work+1)/Chunks)*16,next_n=((work+1)%Chunks)*Group;
          #pragma unroll
          for (int mi=0;mi<MF;++mi) {
            if (next_k!=kk) load_a<true>(ar[next][mi],current,lane,warp_row+mi*16,next_k);
            else {
              // Reuse current A; compiler can coalesce these register copies.
              #pragma unroll
              for (int j=0;j<4;++j) ar[next][mi][j]=ar[slot][mi][j];
            }
          }
          #pragma unroll
          for (int ni=0;ni<Group;++ni)
            load_b<true>(br[next][ni],weight,lane,warp_col+(next_n+ni)*8,next_k);
        }
        #pragma unroll
        for (int mi=0;mi<MF;++mi) {
          #pragma unroll
          for (int ni=0;ni<Group;++ni) mma(acc[mi][start+ni],ar[slot][mi],br[slot][ni]);
        }
      }
    } else {
      #pragma unroll
      for (int kk=0;kk<BK;kk+=16) {
        uint32_t ar[MF][4];
        #pragma unroll
        for (int mi=0;mi<MF;++mi) load_a<true>(ar[mi],current,lane,warp_row+mi*16,kk);
        if constexpr (Mode==1) {
          uint32_t br[2][Group][2];
          #pragma unroll
          for (int ni=0;ni<Group;++ni) load_b<true>(br[0][ni],weight,lane,warp_col+ni*8,kk);
          #pragma unroll
          for (int chunk=0;chunk<Chunks;++chunk) {
            const int slot=chunk&1,next=slot^1;
            if (chunk+1<Chunks) {
              #pragma unroll
              for (int ni=0;ni<Group;++ni)
                load_b<true>(br[next][ni],weight,lane,warp_col+((chunk+1)*Group+ni)*8,kk);
            }
            #pragma unroll
            for (int mi=0;mi<MF;++mi) {
              #pragma unroll
              for (int ni=0;ni<Group;++ni)
                mma(acc[mi][chunk*Group+ni],ar[mi],br[slot][ni]);
            }
          }
        } else {
          // Matched serial control from M27: no speculative register loads.
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
      }
    }
    __syncthreads();
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

class RegisterPipelineBf16MmaPlan {
  torch::Tensor a_,w_,out_;
  int device_,m_,n_,k_,group_,mode_,stages_;
  template<int Group,int Mode,int Stages>
  void launch() {
    const dim3 grid((m_+BM-1)/BM,(n_+BN-1)/BN);
    const auto stream=c10::cuda::getCurrentCUDAStream(device_);
    register_pipeline_mma<Group,Mode,Stages><<<grid,128,Elements*2*Stages,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  template<int Group,int Mode>
  void select_stage() { if(stages_==2) launch<Group,Mode,2>(); else launch<Group,Mode,3>(); }
  template<int Group>
  void select_mode() {
    switch(mode_) {
      case 0: select_stage<Group,0>();break;
      case 1: select_stage<Group,1>();break;
      case 2: select_stage<Group,2>();break;
    }
  }
 public:
  RegisterPipelineBf16MmaPlan(torch::Tensor a,torch::Tensor w,torch::Tensor out,
                             int group,int mode,int stages)
      :a_(a),w_(w),out_(out),group_(group),mode_(mode),stages_(stages) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA tensors required");
    device_=a.get_device();
    for(const auto& t:{a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16,"same-device contiguous BF16 required");
    TORCH_CHECK(a.size(0)<=INT_MAX-BM && a.size(1)<=INT_MAX-BK && w.size(0)<=INT_MAX-BN,
                "dimensions exceed int32 bound");
    m_=a.size(0);n_=w.size(0);k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_,"invalid shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}),"output shape mismatch");
    TORCH_CHECK((group_==1 || group_==2) && mode_>=0 && mode_<=2 &&
                (stages_==2 || stages_==3),"invalid group/mode/stages");
    for(const auto& t:{a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"aligned input bases required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output must not alias input");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6,"M29 targets SM86");
    TORCH_CHECK(Elements*2*stages_<=props.sharedMemPerBlock,"shared memory limit exceeded");
    TORCH_CHECK((n_+BN-1)/BN<=props.maxGridSize[1],"N grid exceeds device limit");
  }
  void run() {
    if(m_==0)return;
    c10::cuda::CUDAGuard guard(device_);
    if(group_==1) select_mode<1>(); else select_mode<2>();
  }
};
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<RegisterPipelineBf16MmaPlan>(m,"RegisterPipelineBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int,int,int>())
      .def("run",&RegisterPipelineBf16MmaPlan::run);
}
