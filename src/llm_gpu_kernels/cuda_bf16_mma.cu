// M25: explicit Ampere lane mapping, C = A @ W.T. Experimental, not tuned.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>

namespace {
constexpr int BM = 32, BN = 64, BK = 32, LD = 40;
constexpr int STAGE_ELEMENTS = (BM + BN) * LD;

__device__ __forceinline__ unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

template<bool Async>
__device__ __forceinline__ void copy16(__nv_bfloat16* dst,
                                     const __nv_bfloat16* src, bool valid) {
  if constexpr (Async) {
    const unsigned address = shared_address(dst);
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes) : "memory");
  } else {
    *reinterpret_cast<uint4*>(dst) = valid
        ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
  }
}

template<bool Async>
__device__ __forceinline__ void stage(__nv_bfloat16* smem,
    const __nv_bfloat16* a, const __nv_bfloat16* w,
    int m, int n, int k, int row_start, int col_start, int k_start) {
  // Four threads per row; each owns one aligned 8-BF16 vector.
  const int t = threadIdx.x, row = t / 4, kk = (t % 4) * 8;
  const bool a_valid = row_start + row < m && k_start + kk < k;
  const auto* a_source = a_valid
      ? a + static_cast<int64_t>(row_start + row) * k + k_start + kk : a;
  copy16<Async>(smem + row * LD + kk, a_source, a_valid);
  auto* weight = smem + BM * LD;
  for (int vector = t; vector < BN * 4; vector += 128) {
    const int col = vector / 4, wk = (vector % 4) * 8;
    const bool valid = col_start + col < n && k_start + wk < k;
    const auto* source = valid
        ? w + static_cast<int64_t>(col_start + col) * k + k_start + wk : w;
    copy16<Async>(weight + col * LD + wk, source, valid);
  }
  if constexpr (Async) asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void load_a(uint32_t (&r)[4],
    const __nv_bfloat16* a, int lane, int m_start, int k_start) {
  // x4 matrix order: top-left, bottom-left, top-right, bottom-right.
  const unsigned p = shared_address(a + (m_start + lane % 16) * LD
                                      + k_start + (lane / 16) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(p) : "memory");
}

__device__ __forceinline__ void load_b(uint32_t (&r)[2],
    const __nv_bfloat16* w, int lane, int n_start, int k_start) {
  // W[n,k] is column-major B[k,n]. No .trans: each register holds a K pair.
  // First 16 lanes supply the two matrices; upper lanes repeat valid addresses.
  const unsigned p = shared_address(w + (n_start + lane % 8) * LD
                                      + k_start + ((lane / 8) % 2) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(p) : "memory");
}

__device__ __forceinline__ void mma(float (&c)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

template<int Stages>
__global__ void bf16_mma(const __nv_bfloat16* a, const __nv_bfloat16* w,
                        __nv_bfloat16* out, int m, int n, int k) {
  const int row_start = blockIdx.x * BM, col_start = blockIdx.y * BN;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int iterations = (k + BK - 1) / BK;
  extern __shared__ __align__(32) __nv_bfloat16 smem[];
  float acc[2][2][4] = {};  // two M16 and two N8 tiles per warp

  if constexpr (Stages > 1) {
    // Prime S-1 groups. Main loop issues one more before consuming oldest.
    #pragma unroll
    for (int i = 0; i < Stages - 1; ++i)
      if (i < iterations)
        stage<true>(smem + i * STAGE_ELEMENTS, a, w, m, n, k,
                    row_start, col_start, i * BK);
  }
  for (int tile = 0; tile < iterations; ++tile) {
    auto* current = smem + (tile % Stages) * STAGE_ELEMENTS;
    if constexpr (Stages == 1) {
      stage<false>(current, a, w, m, n, k, row_start, col_start, tile * BK);
    } else {
      const int future = tile + Stages - 1;
      if (future < iterations) {
        stage<true>(smem + (future % Stages) * STAGE_ELEMENTS,
                    a, w, m, n, k, row_start, col_start, future * BK);
        // Each thread waits for its oldest group; CTA barrier covers peers.
        asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 1) : "memory");
      } else {
        // Tail: fewer than S groups exist, so drain rather than over-wait.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      }
    }
    __syncthreads();
    const auto* weight = current + BM * LD;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ar[2][4], br[2][2];
      load_a(ar[0], current, lane, 0, kk);
      load_a(ar[1], current, lane, 16, kk);
      load_b(br[0], weight, lane, warp * 16, kk);
      load_b(br[1], weight, lane, warp * 16 + 8, kk);
      #pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) mma(acc[mi][ni], ar[mi], br[ni]);
      }
    }
    // Consumers finished; the next iteration may overwrite this slot.
    __syncthreads();
  }
  // PTX m16n8 accumulator: lane group selects row, lane-in-group a col pair.
  #pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      #pragma unroll
      for (int element = 0; element < 4; ++element) {
        const int row = row_start + mi * 16 + lane / 4 + (element / 2) * 8;
        const int col = col_start + warp * 16 + ni * 8 + (lane % 4) * 2 + element % 2;
        if (row < m && col < n)
          out[static_cast<int64_t>(row) * n + col] = __float2bfloat16_rn(acc[mi][ni][element]);
      }
    }
  }
}

class Bf16MmaPlan {
  torch::Tensor a_, w_, out_;
  int device_, m_, n_, k_, stages_;
 public:
  Bf16MmaPlan(torch::Tensor a, torch::Tensor w, torch::Tensor out, int stages)
      : a_(a), w_(w), out_(out), stages_(stages) {
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(), "CUDA tensors required");
    device_ = a.get_device();
    for (const auto& t : {a_, w_, out_}) {
      TORCH_CHECK(t.get_device() == device_ && t.is_contiguous() &&
                  t.scalar_type() == at::kBFloat16, "same-device contiguous BF16 required");
      TORCH_CHECK(t.dim() == 2, "rank-two tensors required");
    }
    TORCH_CHECK(a.size(0) <= INT_MAX - BM && a.size(1) <= INT_MAX - BK &&
                w.size(0) <= INT_MAX - BN, "dimensions exceed int32 arithmetic bound");
    m_ = a.size(0); k_ = a.size(1); n_ = w.size(0);
    TORCH_CHECK(k_ >= 16 && k_ % 16 == 0 && n_ > 0 && w.size(1) == k_,
                "C=A@W.T requires positive N and K divisible by 16");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({m_, n_}), "output shape mismatch");
    TORCH_CHECK(stages_ >= 1 && stages_ <= 4, "stages must be 1..4");
    for (const auto& t : {a_, w_}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                  "16-byte aligned input bases required");
      TORCH_CHECK(t.storage().unsafeGetStorageImpl() != out_.storage().unsafeGetStorageImpl(),
                  "output must not alias inputs");
    }
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device_));
    TORCH_CHECK(props.major == 8 && props.minor == 6, "M25 targets SM86");
    TORCH_CHECK((n_ + BN - 1) / BN <= props.maxGridSize[1], "N grid exceeds device limit");
  }
  void run() {
    if (m_ == 0) return;
    c10::cuda::CUDAGuard guard(device_);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    const dim3 grid((m_ + BM - 1) / BM, (n_ + BN - 1) / BN);
    const auto* a = reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr());
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr());
    auto* out = reinterpret_cast<__nv_bfloat16*>(out_.data_ptr());
    const size_t bytes = stages_ * STAGE_ELEMENTS * sizeof(__nv_bfloat16);
    switch (stages_) {
      case 1: bf16_mma<1><<<grid,128,bytes,stream>>>(a,w,out,m_,n_,k_); break;
      case 2: bf16_mma<2><<<grid,128,bytes,stream>>>(a,w,out,m_,n_,k_); break;
      case 3: bf16_mma<3><<<grid,128,bytes,stream>>>(a,w,out,m_,n_,k_); break;
      case 4: bf16_mma<4><<<grid,128,bytes,stream>>>(a,w,out,m_,n_,k_); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Bf16MmaPlan>(m, "Bf16MmaPlan")
      .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,int>())
      .def("run", &Bf16MmaPlan::run);
}
