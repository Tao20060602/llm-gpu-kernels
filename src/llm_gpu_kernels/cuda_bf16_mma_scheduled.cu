// M27: interleave short B-fragment batches with MMA; preserve each C's K order.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "ampere_mma_primitives.cuh"

namespace {
using namespace llm_gpu_kernels_sm86;
constexpr int BM = 64, BN = 128, MF = 2, NF = 8;

template<int Group, int Stages, bool Swizzle>
__global__ void scheduled_mma(const __nv_bfloat16* a, const __nv_bfloat16* w,
                             __nv_bfloat16* out, int m, int n, int k) {
  constexpr int LD = Swizzle ? BK : 40, Elements = (BM + BN) * LD;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int warp_row = (warp / 2) * 32, warp_col = (warp % 2) * 64;
  const int row_start = blockIdx.x * BM, col_start = blockIdx.y * BN;
  const int iterations = (k + BK - 1) / BK;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[MF][NF][4] = {};
  #pragma unroll
  for (int tile = 0; tile < Stages - 1; ++tile)
    if (tile < iterations)
      stage<BM,BN,Swizzle,true>(smem + tile * Elements,a,w,m,n,k,row_start,col_start,tile*BK);

  for (int tile = 0; tile < iterations; ++tile) {
    auto* current = smem + (tile % Stages) * Elements;
    const int future = tile + Stages - 1;
    if (future < iterations) {
      stage<BM,BN,Swizzle,true>(smem + (future % Stages) * Elements,
                               a,w,m,n,k,row_start,col_start,future*BK);
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
    } else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const auto* weight = current + BM * LD;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ar[MF][4];
      #pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        load_a<Swizzle>(ar[mi],current,lane,warp_row + mi*16,kk);
      // Group=8 is the original all-B-then-MMA schedule. Smaller groups
      // shorten B register live ranges and mix LDSM with HMMA bursts.
      #pragma unroll
      for (int start = 0; start < NF; start += Group) {
        uint32_t br[Group][2];
        #pragma unroll
        for (int ni = 0; ni < Group; ++ni)
          load_b<Swizzle>(br[ni],weight,lane,warp_col + (start+ni)*8,kk);
        #pragma unroll
        for (int mi = 0; mi < MF; ++mi) {
          #pragma unroll
          for (int ni = 0; ni < Group; ++ni)
            mma(acc[mi][start+ni],ar[mi],br[ni]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int mi = 0; mi < MF; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < NF; ++ni) {
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = row_start + warp_row + mi*16 + lane/4 + (e/2)*8;
        const int col = col_start + warp_col + ni*8 + (lane%4)*2 + e%2;
        if (row < m && col < n)
          out[static_cast<int64_t>(row)*n + col] = __float2bfloat16_rn(acc[mi][ni][e]);
      }
    }
  }
}

class ScheduledBf16MmaPlan {
  torch::Tensor a_,w_,out_;
  int device_,m_,n_,k_,group_,stages_;
  bool swizzle_;
  template<int Group, int Stages, bool Swizzle>
  void launch() {
    constexpr int Bytes = (BM+BN)*(Swizzle ? BK : 40)*2*Stages;
    const dim3 grid((m_+BM-1)/BM,(n_+BN-1)/BN);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    scheduled_mma<Group,Stages,Swizzle><<<grid,128,Bytes,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  template<int Group, bool Swizzle>
  void select_stage() {
    if (stages_==2) launch<Group,2,Swizzle>();
    else launch<Group,3,Swizzle>();
  }
  template<bool Swizzle>
  void select_group() {
    switch (group_) {
      case 1: select_stage<1,Swizzle>(); break;
      case 2: select_stage<2,Swizzle>(); break;
      case 4: select_stage<4,Swizzle>(); break;
      case 8: select_stage<8,Swizzle>(); break;
    }
  }
 public:
  ScheduledBf16MmaPlan(torch::Tensor a, torch::Tensor w, torch::Tensor out,
                       int group, int stages, bool swizzle)
      : a_(a),w_(w),out_(out),group_(group),stages_(stages),swizzle_(swizzle) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(), "CUDA tensors required");
    device_=a.get_device();
    for (const auto& t : {a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16, "same-device contiguous BF16 matrices required");
    TORCH_CHECK(a.size(0)<=INT_MAX-BM && a.size(1)<=INT_MAX-BK && w.size(0)<=INT_MAX-BN,
                "dimensions exceed int32 bound");
    m_=a.size(0); n_=w.size(0); k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_, "invalid GEMM shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}), "output shape mismatch");
    TORCH_CHECK((group_==1 || group_==2 || group_==4 || group_==8) &&
                (stages_==2 || stages_==3), "invalid group/stages");
    for (const auto& t : {a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0, "aligned inputs required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output must not alias inputs");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6, "M27 targets SM86");
    const int bytes=(BM+BN)*(swizzle_ ? BK : 40)*2*stages_;
    TORCH_CHECK(bytes<=props.sharedMemPerBlock, "shared memory limit exceeded");
    TORCH_CHECK((n_+BN-1)/BN<=props.maxGridSize[1], "N grid exceeds device limit");
  }
  void run() {
    if (m_==0) return;
    c10::cuda::CUDAGuard guard(device_);
    if (swizzle_) select_group<true>();
    else select_group<false>();
  }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<ScheduledBf16MmaPlan>(m,"ScheduledBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int,int,bool>())
      .def("run",&ScheduledBf16MmaPlan::run);
}
