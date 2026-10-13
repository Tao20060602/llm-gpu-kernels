# M29: shared-memory to register pipeline

## Hypothesis and controlled comparison

M26 counter samples showed substantial math-pipeline throttling at HMMA instructions. M27 shortened B-fragment batches; M28 added producer warps but did not demonstrate a stable improvement over M27. These observations motivate checking operand supply and scheduling, but do not prove that shared-memory latency is the dominant bottleneck. A matched Torch profile is still missing.

M29 changes only register operand scheduling. It holds the CTA tile at 64×128, BK at 32, four compute warps, XOR shared layout, BF16 operands, FP32 accumulators, and BF16 output. Group 1/2 and shared stages 2/3 give twelve candidates across three modes. This separates operand prefetch from changing the amount of work per CTA or the global-memory pipeline.

## Mapping and two different pipelines

Each warp computes 32×64 outputs: two M16 fragments and eight N8 fragments. Each `mma.sync.m16n8k16` consumes one K16 slice; BK32 therefore has two slices. Each thread owns 64 FP32 accumulator values. A fragments use `ldmatrix.x4`, and B fragments use `ldmatrix.x2` through the existing SM86 primitives.

The existing global → shared pipeline uses `cp.async` and a shared ring. Two stages reserve 24,576 bytes; three reserve 36,864 bytes. The next shared tile is issued before computing the current one. Waits and CTA barriers protect arrival and slot reuse.

M29 adds a separate shared → register scheduling experiment:

| Mode | Register schedule | Intended tradeoff |
|---|---|---|
| 0 | Load a B group, then issue its MMA instructions | Matched serial control |
| 1 | Keep A for the K16 slice; preload the next B group into the other register bank before consuming the current group | Give the scheduler independent loads and computation without doubling A state |
| 2 | Alternate both A/B banks across N groups and the two K16 slices | Prefetch the next K16 A fragments as well; more live state and bookkeeping |

Mode 2 copies the current A registers when the next N group shares the same K16 slice. Only the K16 boundary loads new A. Neither mode prefetches across shared-ring transitions: those still use the original CTA barriers. Each output keeps the original K accumulation order; all 72 validation cases match the paired M27 output bitwise.

Source entry points: [CUDA implementation](../src/llm_gpu_kernels/cuda_bf16_mma_register_pipeline.cu), [prepared wrapper](../src/llm_gpu_kernels/gemm_mma_register_pipeline.py), and [diagnostic runner](../scripts/run_m29_gemm_check.py). The Python plan validates and prepares outside timing, then launches without allocation on the current stream. Its shape/alignment/SM86 restrictions follow the existing MMA contract.

## What the compiler actually produced

All twelve instantiations have zero reported stack/spills. Two-stage versions use 128 registers/thread; three-stage versions use 164. Identical register counts alone do not establish identical code.

The existing binary was inspected using Triton's bundled `cuobjdump`; no GPU run or recompilation was needed. Full output is retained locally at `artifacts/m29-build/m29.sass` (SHA256 `4edd1301f201e85f68f679d4c9a4adcdf994d6f2e55bb190eb96ad54815ef5be`). For G2/S2:

| Mode | Total instruction mnemonics | LDGSTS | LDSM | HMMA |
|---|---:|---:|---:|---:|
| 0 | 1304 | 24 | 20 | 32 |
| 1 | 1312 | 24 | 20 | 32 |
| 2 | 1328 | 24 | 20 | 32 |

The ordered LDGSTS/LDSM/HMMA subsequence is identical for modes 0 and 2, but differs for mode 1. The first mode 0/1 difference is at reported sequence position 39: LDSM versus HMMA. This confirms a visible scheduling change for B-only prefetch. It does not show improved latency hiding, and mode 0/2 have different full instruction sequences, so it does not prove the entire A/B prefetch was eliminated. Mnemonic extraction ignores PCs, register operands, and predicates; it is a narrow static comparison, not a dependency or execution analysis.

## Result and next step

[The same-batch timings](M29_REGISTER_PIPELINE_REPORT.md) show only a 0.35% improvement at 4096³ and a small regression at 2048³. The extra source-level pipeline has not closed the Torch gap. Keep this as a documented unsuccessful optimization hypothesis rather than selecting an observed winner as a new default.

The subsequent `scripts/run_m29_baseline_ncu.sh` capture compares Torch default/strict and M27 at 4096³, serially. It profiles the fourth warmed GEMM under an NVTX range, including every kernel inside the Torch call. The worker records the BF16 reduction flag. [The capture is now available](M29_BASELINE_NCU_METRICS.md). Compare launch resources, tensor utilization, scheduler availability, and memory traffic with their units; use the non-profiled Graph results for performance claims.

After that comparison, BK64 is the next candidate if loop/load overhead remains plausible. Doubling BK also doubles shared storage at fixed stages, potentially reducing resident CTAs; it is a hypothesis to evaluate, not an automatic improvement. Do not add deeper buffers or producer warps solely to increase the feature list.
