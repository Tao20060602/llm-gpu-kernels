#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace {
namespace wm = nvcuda::wmma;
constexpr int E = 128, H = 2048, I = 768, BM = 32, BN = 64, BK = 32, LD = 40;
constexpr int STAGE_ELEMENTS = (BM + 2 * BN) * LD;
constexpr int EPILOGUE_BYTES = 2 * BM * BN * sizeof(float);

template<bool Async>
__device__ __forceinline__ void copy16(__nv_bfloat16* dst,
                                      const __nv_bfloat16* src, bool valid) {
  if constexpr (Async) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int source_bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(source_bytes) : "memory");
  } else {
    const uint4 value = valid ? *reinterpret_cast<const uint4*>(src)
                             : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(dst) = value;
  }
}

template<bool Async>
__device__ void stage(__nv_bfloat16* shared, const __nv_bfloat16* a,
                      const __nv_bfloat16* w, int64_t expert, int row_base,
                      int local_start, int row_count, int n_start, int k_start) {
  const int thread = threadIdx.x;
  const int a_row = thread / (BK / 8), a_k = (thread % (BK / 8)) * 8;
  const bool valid = local_start + a_row < row_count;
  const __nv_bfloat16* source = valid
      ? a + static_cast<int64_t>(row_base + local_start + a_row) * H + k_start + a_k
      : a;  // zero-fill never receives an out-of-range source pointer
  copy16<Async>(shared + a_row * LD + a_k, source, valid);
  auto* gate = shared + BM * LD;
  auto* up = gate + BN * LD;
  for (int vector = thread; vector < BN * (BK / 8); vector += 128) {
    const int n = vector / (BK / 8), k = (vector % (BK / 8)) * 8;
    const int64_t weight_index = expert * (2LL * I * H) + (n_start + n) * H + k_start + k;
    copy16<Async>(gate + n * LD + k, w + weight_index, true);
    copy16<Async>(up + n * LD + k, w + weight_index + static_cast<int64_t>(I) * H, true);
  }
  if constexpr (Async) asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool Async, bool SkipUpper>
__global__ void cuda_gate_up_bf16(const __nv_bfloat16* a, const __nv_bfloat16* w,
                                 __nv_bfloat16* out, const int* offsets,
                                 const int* counts, const int* tasks) {
  const int task = blockIdx.x;
  if (task >= tasks[E]) return;  // uniform CTA branch before barriers
  int low = 0, high = E;
  while (low < high) {
    const int middle = (low + high) / 2;
    if (task >= tasks[middle + 1]) low = middle + 1;
    else high = middle;
  }
  const int expert = low;
  const int local_start = (task - tasks[expert]) * BM;
  const int row_base = offsets[expert], row_count = counts[expert];
  const int n_start = blockIdx.y * BN, warp = threadIdx.x / 32;
  const bool upper = !SkipUpper || local_start + 16 < row_count;
  extern __shared__ __align__(32) unsigned char memory[];
  auto* shared = reinterpret_cast<__nv_bfloat16*>(memory);
  using A = wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major>;
  using B = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major>;
  using C = wm::fragment<wm::accumulator, 16, 16, 16, float>;
  C gate0, up0, gate1, up1;
  wm::fill_fragment(gate0, 0.0f); wm::fill_fragment(up0, 0.0f);
  wm::fill_fragment(gate1, 0.0f); wm::fill_fragment(up1, 0.0f);
  if constexpr (Async) {
    stage<true>(shared, a, w, expert, row_base, local_start, row_count, n_start, 0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }
  for (int k_start = 0; k_start < H; k_start += BK) {
    const int slot = Async ? ((k_start / BK) & 1) : 0;
    auto* current = shared + slot * STAGE_ELEMENTS;
    if constexpr (Async) {
      if (k_start + BK < H)
        stage<true>(shared + (slot ^ 1) * STAGE_ELEMENTS, a, w, expert,
                    row_base, local_start, row_count, n_start, k_start + BK);
    } else {
      stage<false>(current, a, w, expert, row_base, local_start, row_count, n_start, k_start);
      __syncthreads();
    }
    const auto* gate = current + BM * LD + warp * 16 * LD;
    const auto* up = gate + BN * LD;
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      A a0, a1; B g, u;
      wm::load_matrix_sync(a0, current + k, LD);
      wm::load_matrix_sync(g, gate + k, LD);
      wm::load_matrix_sync(u, up + k, LD);
      wm::mma_sync(gate0, a0, g, gate0); wm::mma_sync(up0, a0, u, up0);
      if (upper) {
        wm::load_matrix_sync(a1, current + 16 * LD + k, LD);
        wm::mma_sync(gate1, a1, g, gate1); wm::mma_sync(up1, a1, u, up1);
      }
    }
    if constexpr (Async) {
      if (k_start + BK < H) asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();  // all consumers done; next slot visible to the whole CTA
  }
  auto* gate_result = reinterpret_cast<float*>(memory);
  auto* up_result = gate_result + BM * BN;
  wm::store_matrix_sync(gate_result + warp * 16, gate0, BN, wm::mem_row_major);
  wm::store_matrix_sync(up_result + warp * 16, up0, BN, wm::mem_row_major);
  if (upper) {
    wm::store_matrix_sync(gate_result + 16 * BN + warp * 16, gate1, BN, wm::mem_row_major);
    wm::store_matrix_sync(up_result + 16 * BN + warp * 16, up1, BN, wm::mem_row_major);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < BM * BN; index += 128) {
    const int row = index / BN, col = index % BN;
    if (local_start + row < row_count) {
      const float g = __bfloat162float(__float2bfloat16_rn(gate_result[index]));
      const float u = __bfloat162float(__float2bfloat16_rn(up_result[index]));
      out[static_cast<int64_t>(row_base + local_start + row) * I + n_start + col] =
          __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
    }
  }
}

class GateUpPlan {
  torch::Tensor a_, w_, out_, offsets_, counts_, tasks_;
  int device_, bound_, variant_;
 public:
  GateUpPlan(torch::Tensor a, torch::Tensor w, torch::Tensor out,
             torch::Tensor offsets, torch::Tensor counts, torch::Tensor tasks, int variant)
      : a_(a), w_(w), out_(out), offsets_(offsets), counts_(counts), tasks_(tasks),
        device_(a.get_device()), variant_(variant) {
    TORCH_CHECK(a.is_cuda() && variant >= 0 && variant <= 2, "CUDA tensors / variant 0..2 required");
    for (const auto& t : {a_, w_, out_, offsets_, counts_, tasks_}) {
      TORCH_CHECK(t.is_cuda() && t.get_device() == device_ && t.is_contiguous(), "same-device contiguous tensors required");
    }
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                out.scalar_type() == at::kBFloat16, "BF16 operands required");
    TORCH_CHECK(a.dim() == 2 && a.size(1) == H && w.sizes() == torch::IntArrayRef({E, 2 * I, H}) &&
                out.sizes() == torch::IntArrayRef({a.size(0), I}), "only E128/H2048/I768 supported");
    TORCH_CHECK(a.size(0) <= INT32_MAX - E * BM, "route rows exceed int32 task bound");
    for (const auto& t : {offsets_, counts_, tasks_})
      TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 1, "int32 vector metadata required");
    TORCH_CHECK(offsets.numel() == E && counts.numel() == E && tasks.numel() == E + 1,
                "metadata shape mismatch; caller must use BM32 prefix");
    for (const auto& t : {a_, w_, offsets_, counts_, tasks_})
      TORCH_CHECK(t.storage().unsafeGetStorageImpl() != out_.storage().unsafeGetStorageImpl(), "out must not alias inputs");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "16-byte aligned inputs required");
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device_));
    TORCH_CHECK(props.major == 8 && props.minor == 6, "requires SM86");
    bound_ = (a.size(0) + BM - 1) / BM + E - 1;
  }
  void run() {
    if (a_.size(0) == 0) return;
    c10::cuda::CUDAGuard guard(device_);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    const auto* a = reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr());
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(w_.data_ptr());
    auto* out = reinterpret_cast<__nv_bfloat16*>(out_.data_ptr());
    const dim3 grid(bound_, I / BN);
    const auto args_offsets = offsets_.data_ptr<int>();
    const auto args_counts = counts_.data_ptr<int>();
    const auto args_tasks = tasks_.data_ptr<int>();
    if (variant_ == 0)
      cuda_gate_up_bf16<false, false><<<grid, 128, EPILOGUE_BYTES, stream>>>(a, w, out, args_offsets, args_counts, args_tasks);
    else if (variant_ == 1)
      cuda_gate_up_bf16<true, false><<<grid, 128, 2 * STAGE_ELEMENTS * 2, stream>>>(a, w, out, args_offsets, args_counts, args_tasks);
    else
      cuda_gate_up_bf16<true, true><<<grid, 128, 2 * STAGE_ELEMENTS * 2, stream>>>(a, w, out, args_offsets, args_counts, args_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<GateUpPlan>(m, "GateUpPlan")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, int>())
      .def("run", &GateUpPlan::run);
}
