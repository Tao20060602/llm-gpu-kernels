// M24 W4A16 layout candidate: synchronous scalar-word reads from N-contiguous
// packed storage. Arithmetic, shared staging, WMMA order and epilogue match M22.
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
constexpr int SHARED_BYTES = 2 * BM * BN * sizeof(float);

__device__ void decode_stage(__nv_bfloat16* shared, const __nv_bfloat16* a,
                             const uint32_t* words, const __nv_bfloat16* scales,
                             int expert, int row_base, int local_start,
                             int row_count, int n_start, int k_start) {
  const int thread = threadIdx.x;
  const int row = thread / 4, k = (thread % 4) * 8;
  uint4 a_vector = make_uint4(0, 0, 0, 0);
  if (local_start + row < row_count)
    a_vector = *reinterpret_cast<const uint4*>(a + static_cast<int64_t>(row_base + local_start + row) * H + k_start + k);
  *reinterpret_cast<uint4*>(shared + row * LD + k) = a_vector;

  // Threads remain projection=t/64 and channel=t%64. The final physical
  // axis is channel, so each of the four scalar word planes is coalesced
  // across channels; do not vector-load the four separated word planes.
  const int projection = thread / BN, col = thread % BN;
  const int n_tile = n_start / BN, k_tile = k_start / BK;
  const int64_t word_base =
      (((((static_cast<int64_t>(expert) * 2 + projection) * (I / BN) + n_tile)
          * (H / BK) + k_tile) * 4) * BN) + col;
  const uint32_t vector_words[4] = {
      words[word_base], words[word_base + BN],
      words[word_base + 2 * BN], words[word_base + 3 * BN]};
  const int64_t scale_index =
      ((((static_cast<int64_t>(expert) * 2 + projection) * (I / BN) + n_tile)
          * (H / 128) + k_start / 128) * BN) + col;
  const float scale = __bfloat162float(scales[scale_index]);
  auto* destination = shared + BM * LD + projection * BN * LD + col * LD;
  #pragma unroll
  for (int word = 0; word < 4; ++word) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int q0 = static_cast<int>((vector_words[word] >> (8 * pair)) & 15) - 8;
      const int q1 = static_cast<int>((vector_words[word] >> (8 * pair + 4)) & 15) - 8;
      reinterpret_cast<__nv_bfloat162*>(destination)[word * 4 + pair] =
          __floats2bfloat162_rn(static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
    }
  }
}

__global__ void cuda_gate_up_w4_layout_sync(const __nv_bfloat16* a, const uint32_t* words,
                                           const __nv_bfloat16* scales, __nv_bfloat16* out,
                                           const int* offsets, const int* counts, const int* tasks) {
  const int task = blockIdx.x;
  if (task >= tasks[E]) return;
  int low = 0, high = E;
  while (low < high) {
    const int middle = (low + high) / 2;
    if (task >= tasks[middle + 1]) low = middle + 1;
    else high = middle;
  }
  const int expert = low, local_start = (task - tasks[expert]) * BM;
  const int row_base = offsets[expert], row_count = counts[expert];
  const int n_start = blockIdx.y * BN, warp = threadIdx.x / 32;
  extern __shared__ __align__(32) unsigned char memory[];
  auto* shared = reinterpret_cast<__nv_bfloat16*>(memory);
  using A = wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major>;
  using B = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major>;
  using C = wm::fragment<wm::accumulator, 16, 16, 16, float>;
  C gate0, gate1, up0, up1;
  wm::fill_fragment(gate0, 0.0f); wm::fill_fragment(gate1, 0.0f);
  wm::fill_fragment(up0, 0.0f); wm::fill_fragment(up1, 0.0f);
  for (int start = 0; start < H; start += BK) {
    decode_stage(shared, a, words, scales, expert, row_base, local_start, row_count, n_start, start);
    __syncthreads();
    const auto* gate = shared + BM * LD + warp * 16 * LD;
    const auto* up = gate + BN * LD;
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      A a0, a1; B g, u;
      wm::load_matrix_sync(a0, shared + k, LD);
      wm::load_matrix_sync(a1, shared + 16 * LD + k, LD);
      wm::load_matrix_sync(g, gate + k, LD); wm::load_matrix_sync(u, up + k, LD);
      wm::mma_sync(gate0, a0, g, gate0); wm::mma_sync(up0, a0, u, up0);
      wm::mma_sync(gate1, a1, g, gate1); wm::mma_sync(up1, a1, u, up1);
    }
    __syncthreads();
  }
  auto* gate_result = reinterpret_cast<float*>(memory);
  auto* up_result = gate_result + BM * BN;
  wm::store_matrix_sync(gate_result + warp * 16, gate0, BN, wm::mem_row_major);
  wm::store_matrix_sync(gate_result + 16 * BN + warp * 16, gate1, BN, wm::mem_row_major);
  wm::store_matrix_sync(up_result + warp * 16, up0, BN, wm::mem_row_major);
  wm::store_matrix_sync(up_result + 16 * BN + warp * 16, up1, BN, wm::mem_row_major);
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

class W4LayoutGateUpPlan {
  torch::Tensor a_, words_, scales_, out_, offsets_, counts_, tasks_;
  int device_, bound_;
 public:
  W4LayoutGateUpPlan(torch::Tensor a, torch::Tensor words, torch::Tensor scales, torch::Tensor out,
                     torch::Tensor offsets, torch::Tensor counts, torch::Tensor tasks)
      : a_(a), words_(words), scales_(scales), out_(out), offsets_(offsets), counts_(counts), tasks_(tasks),
        device_(a.get_device()) {
    TORCH_CHECK(a.is_cuda(), "CUDA input required");
    for (const auto& t : {a_, words_, scales_, out_, offsets_, counts_, tasks_})
      TORCH_CHECK(t.is_cuda() && t.get_device() == device_ && t.is_contiguous(), "same-device contiguous buffers required");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
                scales.scalar_type() == at::kBFloat16 && words.scalar_type() == at::kInt,
                "BF16 activation/scales and int32 words required");
    TORCH_CHECK(a.dim() == 2 && a.size(1) == H && out.sizes() == torch::IntArrayRef({a.size(0), I}),
                "unsupported activation shape");
    TORCH_CHECK(words.sizes() == torch::IntArrayRef({E, 2, I / BN, H / BK, 4, BN}) &&
                scales.sizes() == torch::IntArrayRef({E, 2, I / BN, H / 128, BN}),
                "unsupported M24 W4G128 physical layout");
    for (const auto& t : {offsets_, counts_, tasks_})
      TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 1, "int32 vector metadata required");
    TORCH_CHECK(offsets.numel() == E && counts.numel() == E && tasks.numel() == E + 1,
                "BM32 metadata required");
    TORCH_CHECK(a.size(0) <= INT32_MAX - E * BM, "int32 route bound exceeded");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(words.data_ptr()) % 16 == 0,
                "16-byte aligned inputs required");
    for (const auto& t : {a_, words_, scales_, offsets_, counts_, tasks_})
      TORCH_CHECK(t.storage().unsafeGetStorageImpl() != out.storage().unsafeGetStorageImpl(),
                  "output must not alias inputs");
    c10::cuda::CUDAGuard guard(device_);
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_));
    TORCH_CHECK(properties.major == 8 && properties.minor == 6, "requires SM86");
    bound_ = (a.size(0) + BM - 1) / BM + E - 1;
  }
  void run() {
    if (a_.size(0) == 0) return;
    c10::cuda::CUDAGuard guard(device_);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    cuda_gate_up_w4_layout_sync<<<dim3(bound_, I / BN), 128, SHARED_BYTES, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_.data_ptr()),
        reinterpret_cast<const uint32_t*>(words_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales_.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_.data_ptr()), offsets_.data_ptr<int>(),
        counts_.data_ptr<int>(), tasks_.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<W4LayoutGateUpPlan>(m, "W4LayoutGateUpPlan")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor>())
      .def("run", &W4LayoutGateUpPlan::run);
}
