# M35: roll the segment traversal to reduce live-state pressure

M34 produces correct results but remains slower than reusable pack+FA2. Its compiler resource counts are 223–237 registers per thread without spills. This makes adding a permanent Q fragment array risky. The current source completely unrolls four segments, duplicating the attention loop four times; M35 tests a single change: `#pragma unroll 1` on that outer segment loop.

Keep the four configurations, CTA64/key32/four warps, P handoff, K/V pipeline, shared allocation, strides, masks, and mathematical order unchanged. Dynamic indexing of kernel argument fields may itself have a cost or introduce local storage, so less source duplication does not guarantee faster code or fewer registers. Compile resources and existing-binary disassembly must establish what changed before interpreting timing.

Numerical cases match M34 and use the independent FP32 oracle; also record same-config M34 bitwise equality as an observation. Benchmark all four rolled candidates, M34 S1/direct, and reusable pack+FA2 in one batch at T32/T96/T256, preserving padded current V and the same Graph protocol. No cross-batch speed comparisons or default dispatch changes.

References for register-resident Q and tile-local lifetime organisation are recorded in [M34 reference notes](M34_REFERENCE_NOTES.md); this experiment is a local code-organisation hypothesis, not a claim copied from an external speedup.

Source: [kernel](../src/llm_gpu_kernels/cuda_gqa_attention_rolled.cu), [wrapper](../src/llm_gpu_kernels/attention_cuda_rolled.py), [runner](../scripts/run_m35_rolled_gqa_check.py).

## Compiled result and baseline correction

Registers/thread for S1/shared, S1/direct, S2/shared and S2/direct are 149/150/127/128, versus M34's 235/223/237/230. All configurations have zero stack/spills and no local LDL/STL instructions in disassembly. Each entry has 64 static HMMA instructions rather than M34's 256 because the four-segment body is no longer duplicated. Runtime MMA work is unchanged; static instruction counts are not FLOP reductions or cache measurements.

M35 uses the existing `RawPackAttention(..., reusable=True)` directly on padded inputs for the FA2 arm. M34's older contiguous adapter copied padded V to an intermediate before packing, so it had an extra conversion step. M34 records are retained as that declared historical chain; the M35 raw-layout baseline is stronger and comparisons must use same-batch M35 data. It is not valid to subtract times across these batches to claim the loop change's benefit.

The build succeeded, but a shell tee started before its directory existed, so its log was not captured by that tee. The resource output returned by the build tool and the loaded binary's disassembly are retained in `artifacts/m35-build/static-evidence.txt` and `m35.sass`; this logging error did not trigger a second compilation or GPU run.

## Decision after timing

All 32 numerical cases pass and match the paired M34 configurations bitwise. The matched S1/direct candidate nevertheless regresses versus same-batch M34 S1/direct by about 11.9%/6.1%/14.1% at T32/T96/T256. Register reduction alone did not establish a performance gain. Keep the unrolled M34 body for the next shared-capacity experiment; retain the rolled implementation as a failed hypothesis rather than selecting it from compiler resource counts. See [the complete report](M35_ROLLED_GQA_REPORT.md).
