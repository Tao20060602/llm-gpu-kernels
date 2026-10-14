# M34 CUDA GQA diagnostic

Status: **passed**

## Scope and protocol

Baseline boundary: this historical FA2 arm uses the contiguous adapter and a separate padded-V conversion before KV packing. It is not the strongest existing raw-layout FA2 entry. M35 corrects that interface to `RawPackAttention(..., reusable=True)`; retain this table as the explicitly measured older chain and do not use cross-batch times to assign optimization gains.

Fixed four-segment GQA inputs use Q[T,32,128], four K/V[L,8,128] pairs, history visible to every query, and current keys visible only when j <= query_i. CUDA variants are stages 1/2 crossed with shared/direct-register P modes. Every correctness row uses the independent chunked FP32 oracle and the 0.008 relative RMS / normalized max limits.

The benchmark uses T=32/96/256 with history lengths (2048,64,512), padded current-V token stride 6144, and six arms: four CUDA variants, M15 Triton G=4 / token tile 32 / key tile 64 / 4 warps / 2 stages, and reusable pack+FA2. Each graph replay contains four forwards. Timing order is forward/reverse/forward with three samples per arm per round; compilation and preparation are excluded. Legacy arms copy padded V into preallocated contiguous staging on every invocation, so that conversion is included in their measured path.

## Numerical cases

| Case | T | Segment lengths | Config | Output mode | Padded current V | RMS | Normalized max | Finite | Status |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| T0_history_empty | 0 | [0, 0, 0, 0] | S1 / P0 (shared) | allocated | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S1 / P1 (direct_register) | provided | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S2 / P0 (shared) | allocated | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S2 / P1 (direct_register) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S1 / P0 (shared) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S1 / P1 (direct_register) | allocated | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S2 / P0 (shared) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S2 / P1 (direct_register) | allocated | False | 0.0 | 0.0 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S1 / P0 (shared) | allocated | False | 0.0020202023442834616 | 0.0025818259455263615 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S1 / P1 (direct_register) | provided | False | 0.0020202023442834616 | 0.0025818259455263615 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S2 / P0 (shared) | allocated | False | 0.0020202023442834616 | 0.0025818259455263615 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S2 / P1 (direct_register) | provided | False | 0.0020202023442834616 | 0.0025818259455263615 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S1 / P0 (shared) | provided | True | 0.0020692222751677036 | 0.002800457878038287 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S1 / P1 (direct_register) | allocated | True | 0.0020692222751677036 | 0.002800457878038287 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S2 / P0 (shared) | provided | True | 0.0020692222751677036 | 0.002800457878038287 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S2 / P1 (direct_register) | allocated | True | 0.0020692222751677036 | 0.002800457878038287 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S1 / P0 (shared) | allocated | False | 0.002119546988978982 | 0.0023560794070363045 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S1 / P1 (direct_register) | provided | False | 0.002119546988978982 | 0.0023560794070363045 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S2 / P0 (shared) | allocated | False | 0.002119546988978982 | 0.0023560794070363045 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S2 / P1 (direct_register) | provided | False | 0.002119546988978982 | 0.0023560794070363045 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S1 / P0 (shared) | provided | False | 0.0020999505650252104 | 0.002651122398674488 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S1 / P1 (direct_register) | allocated | False | 0.0020999505650252104 | 0.002651122398674488 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S2 / P0 (shared) | provided | False | 0.0020999505650252104 | 0.002651122398674488 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S2 / P1 (direct_register) | allocated | False | 0.0020999505650252104 | 0.002651122398674488 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S1 / P0 (shared) | allocated | True | 0.002147645689547062 | 0.0029237421695142984 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S1 / P1 (direct_register) | provided | True | 0.002147645689547062 | 0.0029237421695142984 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S2 / P0 (shared) | allocated | True | 0.002147645689547062 | 0.0029237421695142984 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S2 / P1 (direct_register) | provided | True | 0.002147645689547062 | 0.0029237421695142984 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S1 / P0 (shared) | provided | False | 0.0021093853283673525 | 0.0021465050522238016 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S1 / P1 (direct_register) | allocated | False | 0.0021093853283673525 | 0.0021465050522238016 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S2 / P0 (shared) | provided | False | 0.0021093853283673525 | 0.0021465050522238016 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S2 / P1 (direct_register) | allocated | False | 0.0021093853283673525 | 0.0021465050522238016 | True | passed |

## Graph timing

Milliseconds per complete forward; samples are retained in the fixed JSON result.

| T | Arm | p50 | p95 |
|---:|---|---:|---:|
| 32 | cuda_s1_p0 | 0.24166400730609894 | 0.24383359849452974 |
| 32 | cuda_s1_p1 | 0.22553600370883942 | 0.2300927996635437 |
| 32 | cuda_s2_p0 | 0.20068800449371338 | 0.20572160482406615 |
| 32 | cuda_s2_p1 | 0.19123199582099915 | 0.1941503942012787 |
| 32 | m15_triton_g4_s2 | 0.32179200649261475 | 0.33054720759391787 |
| 32 | reusable_pack_fa2 | 0.16582399606704712 | 0.16778240203857422 |
| 96 | cuda_s1_p0 | 0.3176479935646057 | 0.3185024082660675 |
| 96 | cuda_s1_p1 | 0.32025599479675293 | 0.3297279953956604 |
| 96 | cuda_s2_p0 | 0.4055039882659912 | 0.40714240074157715 |
| 96 | cuda_s2_p1 | 0.3829759955406189 | 0.38430720567703247 |
| 96 | m15_triton_g4_s2 | 0.326911985874176 | 0.3274415910243988 |
| 96 | reusable_pack_fa2 | 0.2930639982223511 | 0.29398239850997926 |
| 256 | cuda_s1_p0 | 0.5703679919242859 | 0.5722112059593201 |
| 256 | cuda_s1_p1 | 0.547327995300293 | 0.5485568046569824 |
| 256 | cuda_s2_p0 | 0.6305279731750488 | 0.6334128141403198 |
| 256 | cuda_s2_p1 | 0.5939199924468994 | 0.597811222076416 |
| 256 | m15_triton_g4_s2 | 0.6668800115585327 | 0.6700831890106201 |
| 256 | reusable_pack_fa2 | 0.46720001101493835 | 0.46945279836654663 |

## CUDA resources

The native CTA is fixed at M=64, key tile 32, and four warps. The shared-memory budget is `(64*128 + 2*32*128*stages + 64*32) * 2` bytes. P mode does not change the retained scratch allocation.

| Stages | P mode | Shared memory (B) | Delta vs S1/P0 (B) | Registers/thread | Spill bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | P0 (shared) | 36864 | 0 | 235 | 0 |
| 1 | P1 (direct_register) | 36864 | 0 | 223 | 0 |
| 2 | P0 (shared) | 53248 | 16384 | 237 | 0 |
| 2 | P1 (direct_register) | 53248 | 16384 | 230 | 0 |

Per-case arm resources and workspace sizes are also included in the JSON. Static shared-memory and compiler resource figures do not measure runtime occupancy or utilization.

## Artifacts and provenance

Fixed diagnostic JSON: `results/m34-cuda-gqa-diagnostic.json`.
Post-run correction: only the output path was corrected and the unchanged JSON was moved from `fixedresults/`; no GPU work was rerun.
Raw compiled artifacts: `artifacts/m34-diagnostic`.
CUDA extension build and ptxas resource evidence: `artifacts/m34-build/`.
Source hashes and runtime versions are recorded in the JSON.
