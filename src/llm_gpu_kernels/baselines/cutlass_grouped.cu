#include <torch/extension.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>
#include <cutlass/version.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <pybind11/stl.h>

#if CUTLASS_MAJOR != 3 || CUTLASS_MINOR != 8 || CUTLASS_PATCH != 0
#error "This baseline requires CUTLASS 3.8.0 headers."
#endif

namespace py = pybind11;

namespace {

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAccumulator, ElementAccumulator>;
using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

using GroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    Element,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    GroupScheduleMode::kDeviceOnly>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GroupedKernel>;
using Problem = cutlass::gemm::GemmCoord;

static_assert(sizeof(Element) == sizeof(at::BFloat16), "BF16 storage size mismatch");
static_assert(sizeof(Problem) == 3 * sizeof(int32_t), "Unexpected GemmCoord representation");
static_assert(sizeof(void*) == sizeof(int64_t), "This wrapper expects 64-bit device pointers");

struct MetadataOffsets {
  size_t problems = 0;
  size_t ptr_a = 0;
  size_t ptr_b = 0;
  size_t ptr_c = 0;
  size_t ptr_d = 0;
  size_t lda = 0;
  size_t ldb = 0;
  size_t ldc = 0;
  size_t ldd = 0;
  size_t total_bytes = 0;
};

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t reserve_bytes(size_t& cursor, size_t count, size_t element_size, size_t alignment) {
  cursor = align_up(cursor, alignment);
  const size_t offset = cursor;
  TORCH_CHECK(count == 0 || element_size <= std::numeric_limits<size_t>::max() / count,
              "grouped metadata allocation size overflow");
  const size_t bytes = count * element_size;
  TORCH_CHECK(cursor <= std::numeric_limits<size_t>::max() - bytes,
              "grouped metadata allocation size overflow");
  cursor += bytes;
  return offset;
}

void check_cuda_status(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

class PreparedGroupedGemm {
 public:
  PreparedGroupedGemm(at::Tensor a,
                      at::Tensor weights,
                      at::Tensor out,
                      std::vector<int64_t> counts)
      : a_(std::move(a)),
        weights_(std::move(weights)),
        out_(std::move(out)),
        counts_(std::move(counts)) {
    validate_inputs();
    c10::cuda::CUDAGuard device_guard(device_index_);

    const int64_t n = weights_.size(1);
    const int64_t k = weights_.size(2);
    int64_t row_offset = 0;
    active_experts_.reserve(counts_.size());
    host_problems_.reserve(counts_.size());

    std::vector<Element*> host_ptr_a;
    std::vector<Element*> host_ptr_b;
    std::vector<Element*> host_ptr_c;
    std::vector<Element*> host_ptr_d;
    std::vector<int64_t> host_lda;
    std::vector<int64_t> host_ldb;
    std::vector<int64_t> host_ldc;
    std::vector<int64_t> host_ldd;

    host_ptr_a.reserve(counts_.size());
    host_ptr_b.reserve(counts_.size());
    host_ptr_c.reserve(counts_.size());
    host_ptr_d.reserve(counts_.size());
    host_lda.reserve(counts_.size());
    host_ldb.reserve(counts_.size());
    host_ldc.reserve(counts_.size());
    host_ldd.reserve(counts_.size());

    auto* a_base = reinterpret_cast<Element*>(a_.data_ptr());
    auto* weights_base = reinterpret_cast<Element*>(weights_.data_ptr());
    auto* out_base = reinterpret_cast<Element*>(out_.data_ptr());

    for (size_t expert = 0; expert < counts_.size(); ++expert) {
      const int64_t count = counts_[expert];
      if (count == 0) {
        continue;
      }
      active_experts_.push_back(static_cast<int32_t>(expert));
      host_problems_.emplace_back(static_cast<int32_t>(count),
                                  static_cast<int32_t>(n),
                                  static_cast<int32_t>(k));
      host_ptr_a.push_back(a_base + row_offset * k);
      host_ptr_b.push_back(weights_base + static_cast<int64_t>(expert) * n * k);
      auto* out_expert = out_base + row_offset * n;
      host_ptr_c.push_back(out_expert);
      host_ptr_d.push_back(out_expert);
      host_lda.push_back(k);
      host_ldb.push_back(k);
      host_ldc.push_back(n);
      host_ldd.push_back(n);
      row_offset += count;
    }

    if (active_experts_.empty()) {
      return;
    }

    upload_metadata(host_ptr_a, host_ptr_b, host_ptr_c, host_ptr_d,
                    host_lda, host_ldb, host_ldc, host_ldd);

    const int threadblock_count = GemmGrouped::sufficient(
        host_problems_.data(), static_cast<int>(host_problems_.size()));
    TORCH_CHECK(threadblock_count > 0, "CUTLASS could not find a viable grouped launch size");

    auto* metadata_base = static_cast<uint8_t*>(metadata_.data_ptr());
    auto* device_problems = reinterpret_cast<Problem*>(metadata_base + offsets_.problems);
    auto* device_ptr_a = reinterpret_cast<Element**>(metadata_base + offsets_.ptr_a);
    auto* device_ptr_b = reinterpret_cast<Element**>(metadata_base + offsets_.ptr_b);
    auto* device_ptr_c = reinterpret_cast<Element**>(metadata_base + offsets_.ptr_c);
    auto* device_ptr_d = reinterpret_cast<Element**>(metadata_base + offsets_.ptr_d);
    auto* device_lda = reinterpret_cast<int64_t*>(metadata_base + offsets_.lda);
    auto* device_ldb = reinterpret_cast<int64_t*>(metadata_base + offsets_.ldb);
    auto* device_ldc = reinterpret_cast<int64_t*>(metadata_base + offsets_.ldc);
    auto* device_ldd = reinterpret_cast<int64_t*>(metadata_base + offsets_.ldd);

    typename GemmGrouped::EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);
    arguments_ = std::make_unique<GemmGrouped::Arguments>(
        device_problems,
        static_cast<int>(host_problems_.size()),
        threadblock_count,
        epilogue_params,
        device_ptr_a,
        device_ptr_b,
        device_ptr_c,
        device_ptr_d,
        device_lda,
        device_ldb,
        device_ldc,
        device_ldd,
        host_problems_.data());

    const cutlass::Status implement_status = GemmGrouped::can_implement(*arguments_);
    TORCH_CHECK(implement_status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM cannot implement this plan (status ",
                static_cast<int>(implement_status), ")");

    workspace_bytes_ = GemmGrouped::get_workspace_size(*arguments_);
    if (workspace_bytes_ > 0) {
      workspace_ = torch::empty(
          {static_cast<int64_t>(workspace_bytes_)},
          torch::TensorOptions().device(at::Device(at::kCUDA, device_index_)).dtype(at::kByte));
    }

    const auto current_stream = c10::cuda::getCurrentCUDAStream(device_index_);
    const cutlass::Status initialize_status = gemm_.initialize(
        *arguments_, workspace_bytes_ > 0 ? workspace_.data_ptr() : nullptr,
        current_stream.stream());
    TORCH_CHECK(initialize_status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM initialization failed (status ",
                static_cast<int>(initialize_status), ")");
    initialized_ = true;
  }

  void run() {
    if (active_experts_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(run_mutex_);
    c10::cuda::CUDAGuard device_guard(device_index_);
    const auto current_stream = c10::cuda::getCurrentCUDAStream(device_index_);
    c10::cuda::CUDACachingAllocator::recordStream(a_.storage().data_ptr(), current_stream);
    c10::cuda::CUDACachingAllocator::recordStream(weights_.storage().data_ptr(), current_stream);
    c10::cuda::CUDACachingAllocator::recordStream(out_.storage().data_ptr(), current_stream);
    c10::cuda::CUDACachingAllocator::recordStream(metadata_.storage().data_ptr(), current_stream);
    if (workspace_bytes_ > 0) {
      c10::cuda::CUDACachingAllocator::recordStream(workspace_.storage().data_ptr(), current_stream);
    }
    const cutlass::Status status = gemm_.run(current_stream.stream());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS grouped GEMM run failed (status ", static_cast<int>(status), ")");
  }

  const std::vector<int32_t>& active_experts() const { return active_experts_; }
  int device_index() const { return device_index_; }
  size_t workspace_bytes() const { return workspace_bytes_; }

 private:
  void validate_inputs() {
    TORCH_CHECK(a_.is_cuda() && weights_.is_cuda() && out_.is_cuda(),
                "CUTLASS grouped GEMM expects CUDA tensors");
    TORCH_CHECK(a_.dim() == 2 && weights_.dim() == 3 && out_.dim() == 2,
                "expected A[sumM,K], W[E,N,K], and out[sumM,N]");
    TORCH_CHECK(a_.scalar_type() == at::kBFloat16 &&
                    weights_.scalar_type() == at::kBFloat16 &&
                    out_.scalar_type() == at::kBFloat16,
                "CUTLASS grouped GEMM expects BF16 A, W, and output");
    TORCH_CHECK(a_.is_contiguous() && weights_.is_contiguous() && out_.is_contiguous(),
                "CUTLASS grouped GEMM expects contiguous tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a_.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(weights_.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(out_.data_ptr()) % 16 == 0,
                "the align8 BF16 configuration requires 16-byte aligned tensor pointers");
    TORCH_CHECK(a_.device() == weights_.device() && a_.device() == out_.device(),
                "A, W, and output must be on the same CUDA device");
    TORCH_CHECK(!out_.is_alias_of(a_) && !out_.is_alias_of(weights_),
                "output must not share storage with A or W");
    TORCH_CHECK(!counts_.empty() &&
                    static_cast<int64_t>(counts_.size()) == weights_.size(0),
                "counts must contain one host count per expert");

    device_index_ = a_.get_device();
    TORCH_CHECK(device_index_ == 0, "the CUTLASS SM86 baseline requires CUDA device 0");
    int major = 0;
    int minor = 0;
    check_cuda_status(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index_),
                      "cudaDeviceGetAttribute(major)");
    check_cuda_status(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index_),
                      "cudaDeviceGetAttribute(minor)");
    TORCH_CHECK(major == 8 && minor == 6,
                "the CUTLASS baseline is compiled for SM86; device 0 is SM", major, minor);

    const int64_t n = weights_.size(1);
    const int64_t k = weights_.size(2);
    TORCH_CHECK(n > 0 && k > 0, "N and K must be positive");
    TORCH_CHECK(a_.size(1) == k, "A.K must equal W.K");
    TORCH_CHECK(n % 8 == 0 && k % 8 == 0,
                "the align8 CUTLASS configuration requires N and K divisible by 8");
    TORCH_CHECK(weights_.size(0) > 0 && weights_.size(0) <= std::numeric_limits<int32_t>::max(),
                "expert count must be in [1, INT32_MAX]");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max() &&
                    k <= std::numeric_limits<int32_t>::max(),
                "N and K must fit CUTLASS GemmCoord");
    TORCH_CHECK(out_.size(0) == a_.size(0) && out_.size(1) == n,
                "output shape must be [A.rows, W.N]");
    TORCH_CHECK(counts_.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "expert count must fit int32");

    int64_t total_rows = 0;
    for (const int64_t count : counts_) {
      TORCH_CHECK(count >= 0 && count <= std::numeric_limits<int32_t>::max(),
                  "counts must be nonnegative and each count must fit int32");
      TORCH_CHECK(total_rows <= std::numeric_limits<int32_t>::max() - count,
                  "sum(counts) must be less than or equal to INT32_MAX");
      total_rows += count;
    }
    TORCH_CHECK(total_rows == a_.size(0), "sum(counts) must equal A.rows");
  }

  template <typename T>
  static void copy_section(std::vector<uint8_t>& host,
                           size_t offset,
                           const std::vector<T>& source) {
    if (!source.empty()) {
      std::memcpy(host.data() + offset, source.data(), source.size() * sizeof(T));
    }
  }

  void upload_metadata(const std::vector<Element*>& host_ptr_a,
                       const std::vector<Element*>& host_ptr_b,
                       const std::vector<Element*>& host_ptr_c,
                       const std::vector<Element*>& host_ptr_d,
                       const std::vector<int64_t>& host_lda,
                       const std::vector<int64_t>& host_ldb,
                       const std::vector<int64_t>& host_ldc,
                       const std::vector<int64_t>& host_ldd) {
    const size_t groups = host_problems_.size();
    size_t cursor = 0;
    offsets_.problems = reserve_bytes(cursor, groups, sizeof(Problem), alignof(Problem));
    offsets_.ptr_a = reserve_bytes(cursor, groups, sizeof(Element*), alignof(Element*));
    offsets_.ptr_b = reserve_bytes(cursor, groups, sizeof(Element*), alignof(Element*));
    offsets_.ptr_c = reserve_bytes(cursor, groups, sizeof(Element*), alignof(Element*));
    offsets_.ptr_d = reserve_bytes(cursor, groups, sizeof(Element*), alignof(Element*));
    offsets_.lda = reserve_bytes(cursor, groups, sizeof(int64_t), alignof(int64_t));
    offsets_.ldb = reserve_bytes(cursor, groups, sizeof(int64_t), alignof(int64_t));
    offsets_.ldc = reserve_bytes(cursor, groups, sizeof(int64_t), alignof(int64_t));
    offsets_.ldd = reserve_bytes(cursor, groups, sizeof(int64_t), alignof(int64_t));
    offsets_.total_bytes = cursor;

    std::vector<uint8_t> host_metadata(offsets_.total_bytes, uint8_t{0});
    copy_section(host_metadata, offsets_.problems, host_problems_);
    copy_section(host_metadata, offsets_.ptr_a, host_ptr_a);
    copy_section(host_metadata, offsets_.ptr_b, host_ptr_b);
    copy_section(host_metadata, offsets_.ptr_c, host_ptr_c);
    copy_section(host_metadata, offsets_.ptr_d, host_ptr_d);
    copy_section(host_metadata, offsets_.lda, host_lda);
    copy_section(host_metadata, offsets_.ldb, host_ldb);
    copy_section(host_metadata, offsets_.ldc, host_ldc);
    copy_section(host_metadata, offsets_.ldd, host_ldd);

    metadata_ = torch::empty(
        {static_cast<int64_t>(offsets_.total_bytes)},
        torch::TensorOptions().device(at::Device(at::kCUDA, device_index_)).dtype(at::kByte));
    check_cuda_status(cudaMemcpy(metadata_.data_ptr(), host_metadata.data(), offsets_.total_bytes,
                                 cudaMemcpyHostToDevice),
                      "upload grouped GEMM metadata");
  }

  at::Tensor a_;
  at::Tensor weights_;
  at::Tensor out_;
  std::vector<int64_t> counts_;
  std::vector<int32_t> active_experts_;
  std::vector<Problem> host_problems_;
  MetadataOffsets offsets_;
  at::Tensor metadata_;
  at::Tensor workspace_;
  std::unique_ptr<GemmGrouped::Arguments> arguments_;
  GemmGrouped gemm_;
  int device_index_ = -1;
  size_t workspace_bytes_ = 0;
  bool initialized_ = false;
  std::mutex run_mutex_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<PreparedGroupedGemm, std::shared_ptr<PreparedGroupedGemm>>(module, "PreparedGroupedGemm")
      .def("run", &PreparedGroupedGemm::run)
      .def_property_readonly("active_experts", &PreparedGroupedGemm::active_experts)
      .def_property_readonly("device_index", &PreparedGroupedGemm::device_index)
      .def_property_readonly("workspace_bytes", &PreparedGroupedGemm::workspace_bytes);

  module.def("prepare", [](at::Tensor a,
                            at::Tensor weights,
                            at::Tensor out,
                            std::vector<int64_t> counts) {
    return std::make_shared<PreparedGroupedGemm>(
        std::move(a), std::move(weights), std::move(out), std::move(counts));
  });
}
