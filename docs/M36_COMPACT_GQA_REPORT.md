# M36 compact-SMEM GQA comparison

Status: **passed**

## Scope and protocol

Fixed BF16 GQA semantics use Q[T,32,128], four K/V[L,8,128] pairs, fully visible history, causal current keys, and the same FP32 softmax/BF16 probability and output arithmetic as M34. M36 exposes only direct-register P mode and stages 1/2. Its shared layout omits M34's unused 4 KiB probability scratch.

The eight functional shapes match M34; each of the two compact configurations is checked against the independent chunked FP32 oracle, and bitwise equality against M34 with the same stage count is recorded as a non-gating comparison.

Timing uses T32/T96/T256 with history lengths (2048,64,512), current-V token stride 6144, and five arms: compact S1/S2, M34 S1/P1 and S2/P1, and reusable raw-layout pack+FA2. Each graph replay contains four forwards; three forward/reverse/forward rounds use three samples per arm per round. FA2 packs the four source segments into reusable K/V buffers on each invocation, so packing is timed; it reads padded current V directly without separate contiguous-V staging.

FP32 relative RMS and normalized max limits are 0.008.

## Functional results

| Case | T | Segment lengths | Stages | Output mode | Padded current V | RMS | Normalized max | Bitwise equal to same-stage M34 | Status |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| T0_history_empty | 0 | [0, 0, 0, 0] | 1 | allocated | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | 2 | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | 1 | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | 2 | allocated | False | 0.0 | 0.0 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | 1 | allocated | False | 0.0020018008071929216 | 0.002693943679332733 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | 2 | provided | False | 0.0020018008071929216 | 0.002693943679332733 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | 1 | provided | True | 0.00204983982257545 | 0.003787652589380741 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | 2 | allocated | True | 0.00204983982257545 | 0.003787652589380741 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | 1 | allocated | False | 0.0021154959686100483 | 0.0017893007025122643 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | 2 | provided | False | 0.0021154959686100483 | 0.0017893007025122643 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | 1 | provided | False | 0.002098108408972621 | 0.0022960547357797623 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | 2 | allocated | False | 0.002098108408972621 | 0.0022960547357797623 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | 1 | allocated | True | 0.0021519591100513935 | 0.002580350963398814 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | 2 | provided | True | 0.0021519591100513935 | 0.002580350963398814 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | 1 | provided | False | 0.002115897834300995 | 0.0026042053941637278 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | 2 | allocated | False | 0.002115897834300995 | 0.0026042053941637278 | True | passed |

## Graph timing

Milliseconds per complete forward; all raw samples are retained in the JSON.

| T | Arm | p50 | p95 |
|---:|---|---:|---:|
| 32 | compact_s1_p1 | 0.22499999403953552 | 0.22742719948291779 |
| 32 | compact_s2_p1 | 0.18969599902629852 | 0.1920896053314209 |
| 32 | m34_s1_p1 | 0.2247679978609085 | 0.2270527958869934 |
| 32 | m34_s2_p1 | 0.18938399851322174 | 0.19342079758644104 |
| 32 | reusable_raw_pack_fa2 | 0.16486400365829468 | 0.16624639928340912 |
| 96 | compact_s1_p1 | 0.32102400064468384 | 0.3226160049438477 |
| 96 | compact_s2_p1 | 0.32179200649261475 | 0.3235327899456024 |
| 96 | m34_s1_p1 | 0.32076799869537354 | 0.3228671967983246 |
| 96 | m34_s2_p1 | 0.3824639916419983 | 0.3831295967102051 |
| 96 | reusable_raw_pack_fa2 | 0.2910720109939575 | 0.29521281123161314 |
| 256 | compact_s1_p1 | 0.5463039875030518 | 0.5468431949615479 |
| 256 | compact_s2_p1 | 0.5140479803085327 | 0.5220639944076538 |
| 256 | m34_s1_p1 | 0.5450239777565002 | 0.5475839972496033 |
| 256 | m34_s2_p1 | 0.5946879982948303 | 0.5980448126792908 |
| 256 | reusable_raw_pack_fa2 | 0.4677119851112366 | 0.46904319524765015 |

## Compact resources and M34 comparison

M36 shared memory is `64*128*2 + 2*32*128*stages*2`; the omitted probability scratch is 4096 bytes. M34 uses the same Q/KV allocation plus that scratch.

| Stages | Shared bytes | Delta vs M34 (B) | Delta vs compact S1 (B) | Registers/thread | Delta regs vs M34 | Spill bytes | Stack bytes | LDL/STL | HMMA/LDSM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 32768 | -4096 | 0 | 223 | 0 | 0 | 0 | 0/0 | 256/288 |
| 2 | 49152 | -4096 | 16384 | 230 | 0 | 0 | 0 | 0/0 | 256/288 |

Per-case arm resources, workspace sizes, raw-pack bytes, source hashes, and runtime versions are recorded in the diagnostic JSON. Static resources do not measure runtime occupancy.

## Artifacts

Diagnostic JSON: `results/m36-compact-gqa-diagnostic.json`.
Raw diagnostic directory: `artifacts/m36-diagnostic`.
CUDA extension build output: `artifacts/m36-build/`.
