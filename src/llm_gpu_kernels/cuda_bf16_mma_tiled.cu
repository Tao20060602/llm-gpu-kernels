// M26: larger CTA tiles and a bit-preserving shared-memory swizzle on SM86.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>

namespace {
constexpr int BK = 32;
template<bool Swizzle>
__device__ __forceinline__ int index(int row, int col) {
  // Swizzle leaves the low three BF16 address bits intact (16-byte vectors).
  // Mixing bits 3..5 is a bijection inside each group of eight K32 rows:
  // physical bit5 = row bit0 XOR row bit2; row bit2 remains recoverable.
  if constexpr (Swizzle) return (row * BK + col) ^ ((row & 7) * 8);
  else return row * 40 + col;
}
__device__ __forceinline__ unsigned address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
template<bool Async>
__device__ __forceinline__ void copy16(__nv_bfloat16* dst,
                                     const __nv_bfloat16* src, bool valid) {
  if constexpr (Async) {
    const unsigned p = address(dst);
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(p), "l"(src), "r"(bytes) : "memory");
  } else {
    *reinterpret_cast<uint4*>(dst) = valid
        ? *reinterpret_cast<const uint4*>(src) : make_uint4(0,0,0,0);
  }
}
template<int BM, int BN, bool Swizzle, bool Async>
__device__ __forceinline__ void stage(__nv_bfloat16* smem,
    const __nv_bfloat16* a, const __nv_bfloat16* w, int m, int n, int k,
    int row_start, int col_start, int k_start) {
  constexpr int Threads = (BM == 128 ? 8 : 4) * 32;
  constexpr int LD = Swizzle ? BK : 40;
  for (int v = threadIdx.x; v < BM * 4; v += Threads) {
    const int row = v / 4, kk = (v % 4) * 8;
    const bool valid = row_start + row < m && k_start + kk < k;
    const auto* src = valid
        ? a + static_cast<int64_t>(row_start + row) * k + k_start + kk : a;
    copy16<Async>(smem + index<Swizzle>(row, kk), src, valid);
  }
  auto* weight = smem + BM * LD;
  for (int v = threadIdx.x; v < BN * 4; v += Threads) {
    const int row = v / 4, kk = (v % 4) * 8;
    const bool valid = col_start + row < n && k_start + kk < k;
    const auto* src = valid
        ? w + static_cast<int64_t>(col_start + row) * k + k_start + kk : w;
    copy16<Async>(weight + index<Swizzle>(row, kk), src, valid);
  }
  if constexpr (Async) asm volatile("cp.async.commit_group;" ::: "memory");
}
template<bool Swizzle>
__device__ __forceinline__ void load_a(uint32_t (&r)[4],
    const __nv_bfloat16* a, int lane, int row, int kk) {
  const unsigned p = address(a + index<Swizzle>(row + lane % 16, kk + (lane / 16) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(p) : "memory");
}
template<bool Swizzle>
__device__ __forceinline__ void load_b(uint32_t (&r)[2],
    const __nv_bfloat16* w, int lane, int row, int kk) {
  const unsigned p = address(w + index<Swizzle>(row + lane % 8, kk + ((lane / 8) % 2) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      : "=r"(r[0]), "=r"(r[1]) : "r"(p) : "memory");
}
__device__ __forceinline__ void mma(float (&c)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template<int BM, int BN, int Stages, bool Swizzle>
__global__ void tiled_mma(const __nv_bfloat16* a, const __nv_bfloat16* w,
                         __nv_bfloat16* out, int m, int n, int k) {
  constexpr int WarpsM = BM == 128 ? 4 : 2, WarpsN = 2;
  constexpr int WarpM = BM / WarpsM, WarpN = BN / WarpsN;
  constexpr int MF = WarpM / 16, NF = WarpN / 8;
  constexpr int LD = Swizzle ? BK : 40, Elements = (BM + BN) * LD;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int warp_row = (warp / WarpsN) * WarpM, warp_col = (warp % WarpsN) * WarpN;
  const int row_start = blockIdx.x * BM, col_start = blockIdx.y * BN;
  const int iterations = (k + BK - 1) / BK;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[MF][NF][4] = {};
  if constexpr (Stages > 1) {
    #pragma unroll
    for (int tile = 0; tile < Stages - 1; ++tile)
      if (tile < iterations)
        stage<BM,BN,Swizzle,true>(smem + tile * Elements, a,w,m,n,k,row_start,col_start,tile*BK);
  }
  for (int tile = 0; tile < iterations; ++tile) {
    auto* current = smem + (tile % Stages) * Elements;
    if constexpr (Stages == 1) {
      stage<BM,BN,Swizzle,false>(current,a,w,m,n,k,row_start,col_start,tile*BK);
    } else {
      const int future = tile + Stages - 1;
      if (future < iterations) {
        stage<BM,BN,Swizzle,true>(smem + (future % Stages) * Elements,
                                 a,w,m,n,k,row_start,col_start,future*BK);
        asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
      } else asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    const auto* weight = current + BM * LD;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ar[MF][4], br[NF][2];
      #pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        load_a<Swizzle>(ar[mi],current,lane,warp_row + mi*16,kk);
      #pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        load_b<Swizzle>(br[ni],weight,lane,warp_col + ni*8,kk);
      #pragma unroll
      for (int mi = 0; mi < MF; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < NF; ++ni) mma(acc[mi][ni],ar[mi],br[ni]);
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

class TiledBf16MmaPlan {
  torch::Tensor a_, w_, out_;
  int device_, m_, n_, k_, tile_, stages_;
  bool swizzle_;
  template<int BM, int BN, int Stages, bool Swizzle>
  void configure(bool launch) {
    constexpr int Bytes = (BM + BN) * (Swizzle ? BK : 40) * 2 * Stages;
    constexpr int Threads = (BM == 128 ? 8 : 4) * 32;
    if (!launch) {
      cudaDeviceProp props;
      C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device_));
      TORCH_CHECK(Bytes <= props.sharedMemPerBlockOptin, "tile/stages exceed shared memory limit");
      TORCH_CHECK((n_+BN-1)/BN <= props.maxGridSize[1], "N grid exceeds device limit");
      if constexpr (Bytes > 48*1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(tiled_mma<BM,BN,Stages,Swizzle>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
      return;
    }
    if (m_ == 0) return;
    const dim3 grid((m_+BM-1)/BM,(n_+BN-1)/BN);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    tiled_mma<BM,BN,Stages,Swizzle><<<grid,Threads,Bytes,stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()),m_,n_,k_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  template<int BM, int BN, bool Swizzle>
  void select_stages(bool launch) {
    switch (stages_) {
      case 1: configure<BM,BN,1,Swizzle>(launch); break;
      case 2: configure<BM,BN,2,Swizzle>(launch); break;
      case 3: configure<BM,BN,3,Swizzle>(launch); break;
      case 4: configure<BM,BN,4,Swizzle>(launch); break;
    }
  }
  template<bool Swizzle>
  void select_tile(bool launch) {
    switch (tile_) {
      case 0: select_stages<64,64,Swizzle>(launch); break;
      case 1: select_stages<64,128,Swizzle>(launch); break;
      case 2: select_stages<128,128,Swizzle>(launch); break;
    }
  }
  void dispatch(bool launch) {
    if (swizzle_) select_tile<true>(launch);
    else select_tile<false>(launch);
  }
 public:
  TiledBf16MmaPlan(torch::Tensor a, torch::Tensor w, torch::Tensor out,
                  int tile, int stages, bool swizzle)
      : a_(a),w_(w),out_(out),tile_(tile),stages_(stages),swizzle_(swizzle) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(), "CUDA tensors required");
    device_ = a.get_device();
    for (const auto& t : {a_,w_,out_})
      TORCH_CHECK(t.dim()==2 && t.is_contiguous() && t.get_device()==device_ &&
                  t.scalar_type()==at::kBFloat16, "same-device contiguous BF16 matrices required");
    TORCH_CHECK(a.size(0)<=INT_MAX-128 && a.size(1)<=INT_MAX-BK && w.size(0)<=INT_MAX-128,
                "dimensions exceed int32 arithmetic bound");
    m_=a.size(0); n_=w.size(0); k_=a.size(1);
    TORCH_CHECK(k_>=16 && k_%16==0 && n_>0 && w.size(1)==k_, "invalid GEMM shape");
    TORCH_CHECK(out.sizes()==torch::IntArrayRef({m_,n_}), "invalid output shape");
    TORCH_CHECK(tile_>=0 && tile_<=2 && stages_>=1 && stages_<=4, "invalid tile/stages");
    for (const auto& t : {a_,w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0, "aligned input bases required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=out_.storage().unsafeGetStorageImpl(),
                  "output aliases input");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device_));
    TORCH_CHECK(props.major==8 && props.minor==6, "M26 targets SM86");
    dispatch(false);  // opt-in attributes are preparation, outside Graph/run
  }
  void run() { c10::cuda::CUDAGuard guard(device_); dispatch(true); }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<TiledBf16MmaPlan>(m,"TiledBf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int,int,bool>())
      .def("run",&TiledBf16MmaPlan::run);
}
