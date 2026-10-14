# M36: remove unused P shared scratch from direct-P attention

M34 reserved P scratch in both modes for a controlled dataflow comparison. DirectP never reads or writes it. M35 rolled traversal reduced registers but did not establish a broad speed benefit, so M36 returns to the M34 unrolled direct-P computation and changes only dynamic shared allocation.

The formula is `(QElements + Stages*KVElements)*2`, removing 4,096 bytes of unused P scratch. Stage1 decreases from 36,864 to 32,768 bytes; stage2 from 53,248 to 49,152. The latter fits the default per-block limit instead of requiring opt-in. Lower allocation can permit more resident CTAs, but actual occupancy is not inferred solely from these bytes and must account for registers, hardware limits, launch size and runtime counters.

Keep the pointer to the end-of-KV region in source only as dead code: DirectP eliminates every P shared access at compile time. The supported P mode is exclusively 1. QK, softmax, P BF16 packing, PV accumulation, strides, masks, four-segment unrolling and K/V prefetch are unchanged. The same output/ownership contract applies.

Eight numerical shapes × two stages pass independent FP32 checks and record matched M34 direct-P bitwise equality. Benchmark both compact candidates, both matched M34 direct-P stages and raw-layout reusable pack+FA2 in one batch at T32/T96/T256. Keep the prior Graph protocol, padded V and before/after numerical checks. Any gain is local forward latency, not engine TTFT or serving throughput.

Source: [CUDA](../src/llm_gpu_kernels/cuda_gqa_attention_compact.cu), [wrapper](../src/llm_gpu_kernels/attention_cuda_compact.py), [runner](../scripts/run_m36_compact_gqa_check.py).

## Observed same-batch result

All 16 numerical cases pass and match the same-stage M34 direct-P output bitwise. Stage2 Graph p50 decreases from 382.464 to 321.792 µs at T96 (15.86%), and from 594.688 to 514.048 µs at T256 (13.56%). T32 is essentially unchanged. Stage1 is essentially unchanged at all three sizes, providing a useful control rather than a blanket claim that removing scratch always improves speed.

At T256, compact stage2 is 5.68% faster than the same-batch M34 stage1/direct arm, but remains about 9.9% slower than raw-layout reusable pack+FA2. At T96 it is approximately tied with M34 stage1/direct. The allocation change has improved the previously expensive two-stage configuration, not produced a universal winner. See [full results](M36_COMPACT_GQA_REPORT.md).

The next profile captures M34/S2, M36/S2 and M34/S1 at T256 under the same NVTX worker. Runtime launch limits/occupancy must confirm the proposed residency explanation; NCU timings will not replace the non-profiled Graph results. The user must authenticate because profiling counters are restricted and current sudo authentication is unavailable.

## Compiled control

Stage1/2 remain at 223/230 registers/thread, with zero stack/spills and no local LDL/STL. Each specialization has 256 static HMMA and 288 LDSM instructions, matching the corresponding M34 direct-P entry. The declared shared capacity is 4,096 bytes smaller. These resource/count checks support the controlled allocation experiment; they do not prove full machine-code identity or measured occupancy. Build and disassembly evidence are retained at `artifacts/m36-build/`.
