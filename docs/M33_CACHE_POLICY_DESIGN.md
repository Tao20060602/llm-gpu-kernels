# M33: isolate L1 bypass and L2 prefetch hints

## Evidence and hypothesis

M32 grouped CTA traversal gives a small improvement at 4096³ but leaves a large Torch gap. Static export of the existing Torch NCU report shows `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` instructions; the exported source table is retained locally at `artifacts/m29-baseline-ncu-20261013T074100Z/torch_default-source.csv`. This is evidence of a SASS load-policy difference, not proof of the original source-level PTX or a single cause of performance.

Keep the M32 GroupM16 traversal, CTA64×128/BK32/S2, four warps, and G2 B-register prefetch. Vary the PTX copy policy only, in a two-by-two comparison:

| Mode | PTX copy | L1 policy | L2 prefetch hint |
|---:|---|---|---|
| 0 | `cp.async.ca.shared.global` | Cache at all levels | None explicit |
| 1 | `cp.async.cg.shared.global` | Bypass L1 | None explicit |
| 2 | `cp.async.ca.shared.global.L2::128B` | Cache at all levels | 128 B |
| 3 | `cp.async.cg.shared.global.L2::128B` | Bypass L1 | 128 B |

CUDA12.8 [PTX documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) defines these as performance hints. The copied payload remains 16 bytes per operation; the L2 hint does not change the number of BF16 elements written to shared memory. It can change cache traffic and may overfetch data. Bypassing L1 may avoid pollution but can also discard useful reuse. Neither choice is automatically faster.

## Controls and scope

All modes retain zero-fill tails, shared arrival/reuse synchronization, MMA arithmetic, and output conversion. Outputs must pass the existing FP32 bounds and match M32 GroupM16 bitwise before timing. GroupM16 was chosen from the prior diagnostic, so this comparison is not a holdout or frozen dispatch-policy validation.

Use the same eight numerical shapes as M32, giving 32 candidates. Benchmark 1024³/2048³/4096³ with eight arms: four cache modes, M32 GroupM16, M29/S2, and Torch strict/default. Use the same three-round Graph protocol and before/after numerical checks. Compile and static SASS evidence establish whether the intended load-policy differences exist; timings establish observed performance. Cache hit/traffic changes for M33 require a separate hardware-counter capture and must not be inferred from speed alone.

Source: [CUDA](../src/llm_gpu_kernels/cuda_bf16_mma_cache_policy.cu), [wrapper](../src/llm_gpu_kernels/gemm_mma_cache_policy.py), [runner](../scripts/run_m33_gemm_check.py).

## Compiler confirmation

CUDA12.8/GCC14 compilation reports 128 registers/thread, 24,576 dynamic shared bytes/CTA, and zero stack/spills for all four modes. Static disassembly of the existing binary contains 24 LDGSTS instructions per specialization:

| Mode | Observed SASS mnemonic |
|---:|---|
| 0 | `LDGSTS.E.128` |
| 1 | `LDGSTS.E.BYPASS.128` |
| 2 | `LDGSTS.E.LTC128B.128` |
| 3 | `LDGSTS.E.BYPASS.LTC128B.128` |

Thus the intended policy differences survived compilation. This still does not measure their cache effects. Full disassembly is retained locally at `artifacts/m33-build/m33.sass` (SHA256 `932b5a2cd21709603014d1d3dbf78a4eae849bc325958ba4d77d08e4b7e281d8`); the build log is at `artifacts/m33-build/build.log`.
