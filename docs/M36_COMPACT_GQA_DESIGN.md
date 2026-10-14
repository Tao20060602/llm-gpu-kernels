# M36: remove unused P shared scratch from direct-P attention

This document and the M36 result/NCU records describe the v1 experiment at commit `3338707`. M37 updates the live compact source and API to v2; see [M37 design](M37_COMPACT_MASK_DESIGN.md). The historical dead-P structure described below is removed from v2, whose caller supplies only `stages`.

M34 reserved P scratch in both modes for a controlled dataflow comparison. DirectP never reads or writes it. M35 rolled traversal reduced registers but did not establish a broad speed benefit, so M36 returns to the M34 unrolled direct-P computation and changes only dynamic shared allocation.

The formula is `(QElements + Stages*KVElements)*2`, removing 4,096 bytes of unused P scratch. Stage1 decreases from 36,864 to 32,768 bytes; stage2 from 53,248 to 49,152. The latter fits the default per-block limit instead of requiring opt-in. Lower allocation can permit more resident CTAs, but actual occupancy is not inferred solely from these bytes and must account for registers, hardware limits, launch size and runtime counters.

The v1 experiment kept the pointer to the end-of-KV region as dead code: DirectP eliminated every P shared access at compile time. Its supported P mode was exclusively 1. QK, softmax, P BF16 packing, PV accumulation, strides, masks, four-segment unrolling and K/V prefetch were unchanged. The same output/ownership contract applied.

Eight numerical shapes × two stages pass independent FP32 checks and record matched M34 direct-P bitwise equality. Benchmark both compact candidates, both matched M34 direct-P stages and raw-layout reusable pack+FA2 in one batch at T32/T96/T256. Keep the prior Graph protocol, padded V and before/after numerical checks. Any gain is local forward latency, not engine TTFT or serving throughput.

Source: [CUDA](../src/llm_gpu_kernels/cuda_gqa_attention_compact.cu), [wrapper](../src/llm_gpu_kernels/attention_cuda_compact.py), [runner](../scripts/run_m36_compact_gqa_check.py).

## Observed same-batch result

All 16 numerical cases pass and match the same-stage M34 direct-P output bitwise. Stage2 Graph p50 decreases from 382.464 to 321.792 µs at T96 (15.86%), and from 594.688 to 514.048 µs at T256 (13.56%). T32 is essentially unchanged. Stage1 is essentially unchanged at all three sizes, providing a useful control rather than a blanket claim that removing scratch always improves speed.

At T256, compact stage2 is 5.68% faster than the same-batch M34 stage1/direct arm, but remains about 9.9% slower than raw-layout reusable pack+FA2. At T96 it is approximately tied with M34 stage1/direct. The allocation change has improved the previously expensive two-stage configuration, not produced a universal winner. See [full results](M36_COMPACT_GQA_REPORT.md).

The T256 profile-only NCU capture for M34/S1, M34/S2 and M36/S2 is complete. For M34/S2 → M36/S2, worker-declared dynamic SMEM changes from 53,248 B to 49,152 B; NCU reports the shared-memory launch limit changing from 1 to 2 blocks while the register limit remains 2 blocks. The captured active-warp metric changes from 8.333455% to 14.273184%. This is one profiled launch per arm; profiler duration is not benchmark speed and these counters do not establish performance causality. See [the NCU report](M36_GQA_NCU_REPORT.md) and [summary JSON](../results/m36-gqa-ncu-summary.json).

## Compiled control

Stage1/2 of the M36 v1 binary remain at 223/230 registers/thread, with zero stack/spills and no local LDL/STL. Each specialization has 256 static HMMA and 288 LDSM instructions, matching the corresponding M34 direct-P entry. The declared shared capacity is 4,096 bytes smaller. These resource/count checks support the controlled allocation experiment; they do not prove full machine-code identity. Runtime occupancy evidence is in the profile-only NCU follow-up linked above. Build and disassembly evidence are retained at `artifacts/m36-build/`.

## M37 v2 follow-up: separate diagnostic batch

M37's single diagnostic and two e2e checks have completed. All 16 numerical cases and both e2e cases passed. The e2e checks updated values in fixed buffers on a non-default stream and verified eager and Graph outputs at S1 and S2. Across all measured finite diagnostic metrics, maximum relative RMS is 0.0022691684 and maximum normalized error is 0.0035540883. Across the two e2e cases, the corresponding maxima are 0.0022706119 and 0.0033076003. See [M37 report](M37_COMPACT_MASK_REPORT.md) and the [S1](../results/provenance/m37-e2e/m37_compact_attention_s1.json)/[S2](../results/provenance/m37-e2e/m37_compact_attention_s2.json) provenance.

M37 adds an internally derived full-history-tile mask branch and removes compact `p_mode`/DirectP and shared-P paths. The compact prepare API exposes `stages` as its only configuration option. In the separate M37 batch, S2 p50 is 0.308992 vs 0.320256 ms at T96 (-3.52%) and 0.497368 vs 0.506624 ms at T256 (-1.83%), comparing v2 to the preserved M36 v1 binary. S1 regresses 1.84% at T32 and 0.96% at T96, with a 1.05% improvement at T256; raw-pack+FA2 remains faster at all three sizes. These are M37 same-batch measurements and must not be mixed with the M36 v1 table above. Retain v2 as an explicit operator experiment with no default change. See [M37 design](M37_COMPACT_MASK_DESIGN.md).
