#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Frozen M26 primitive semantics, reused by new scheduling experiments.
namespace llm_gpu_kernels_sm86 {
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
}  // namespace llm_gpu_kernels_sm86
