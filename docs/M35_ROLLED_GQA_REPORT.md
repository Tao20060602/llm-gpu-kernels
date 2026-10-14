# M35 rolled GQA comparison

Status: **passed**

## Scope and protocol

Fixed four-segment BF16 GQA semantics use Q[T,32,128], four K/V[L,8,128] pairs, fully visible history, causal current keys, GQA head mapping q_head//4, and FP32 softmax state with BF16 probabilities/output. The eight functional shapes match M34. All four rolled configurations use stages 1/2 × P mode 0/1 and the same declared CTA/shared-memory budget as M34.

The fixed timing workloads are T32/T96/T256 with history lengths (2048,64,512) and current-V token stride 6144. Six arms are the four rolled variants, M34 S1/P1, and reusable raw-layout pack+FA2. Each graph replay contains four forwards; three forward/reverse/forward rounds use three samples per arm per round. FA2 copies the four segments into reusable packed K/V buffers on each invocation, so packing is included in its timed path; it reads padded current V directly without a separate contiguous-V staging copy. M34's earlier FA2 result used an additional contiguous-V conversion and is not a speed-comparable raw-layout baseline for this run.

FP32 relative RMS and normalized max limits are 0.008. Bitwise comparisons with M34 are recorded for information and do not determine correctness.

## Functional results

| Case | T | Segment lengths | Config | Output mode | Padded current V | RMS | Normalized max | Bitwise equal to same-config M34 | Status |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| T0_history_empty | 0 | [0, 0, 0, 0] | S1 / P0 (shared) | allocated | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S1 / P1 (direct_register) | provided | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S2 / P0 (shared) | allocated | False | 0.0 | 0.0 | True | passed |
| T0_history_empty | 0 | [0, 0, 0, 0] | S2 / P1 (direct_register) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S1 / P0 (shared) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S1 / P1 (direct_register) | allocated | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S2 / P0 (shared) | provided | False | 0.0 | 0.0 | True | passed |
| T1_history_empty | 1 | [0, 0, 0, 1] | S2 / P1 (direct_register) | allocated | False | 0.0 | 0.0 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S1 / P0 (shared) | allocated | False | 0.0020100707188248634 | 0.0022504625376313925 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S1 / P1 (direct_register) | provided | False | 0.0020100707188248634 | 0.0022504625376313925 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S2 / P0 (shared) | allocated | False | 0.0020100707188248634 | 0.0022504625376313925 | True | passed |
| T9_history_17_0_3 | 9 | [17, 0, 3, 9] | S2 / P1 (direct_register) | provided | False | 0.0020100707188248634 | 0.0022504625376313925 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S1 / P0 (shared) | provided | True | 0.0020465366542339325 | 0.0022269180044531822 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S1 / P1 (direct_register) | allocated | True | 0.0020465366542339325 | 0.0022269180044531822 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S2 / P0 (shared) | provided | True | 0.0020465366542339325 | 0.0022269180044531822 | True | passed |
| T17_history_33_1_0 | 17 | [33, 1, 0, 17] | S2 / P1 (direct_register) | allocated | True | 0.0020465366542339325 | 0.0022269180044531822 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S1 / P0 (shared) | allocated | False | 0.002114409115165472 | 0.002438834635540843 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S1 / P1 (direct_register) | provided | False | 0.002114409115165472 | 0.002438834635540843 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S2 / P0 (shared) | allocated | False | 0.002114409115165472 | 0.002438834635540843 | True | passed |
| T64_history_64_16_32 | 64 | [64, 16, 32, 64] | S2 / P1 (direct_register) | provided | False | 0.002114409115165472 | 0.002438834635540843 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S1 / P0 (shared) | provided | False | 0.002096070908010006 | 0.002015734789893031 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S1 / P1 (direct_register) | allocated | False | 0.002096070908010006 | 0.002015734789893031 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S2 / P0 (shared) | provided | False | 0.002096070908010006 | 0.002015734789893031 | True | passed |
| T65_history_31_32_17 | 65 | [31, 32, 17, 65] | S2 / P1 (direct_register) | allocated | False | 0.002096070908010006 | 0.002015734789893031 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S1 / P0 (shared) | allocated | True | 0.002155775437131524 | 0.0025597752537578344 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S1 / P1 (direct_register) | provided | True | 0.002155775437131524 | 0.0025597752537578344 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S2 / P0 (shared) | allocated | True | 0.002155775437131524 | 0.0025597752537578344 | True | passed |
| T96_history_128_64_32 | 96 | [128, 64, 32, 96] | S2 / P1 (direct_register) | provided | True | 0.002155775437131524 | 0.0025597752537578344 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S1 / P0 (shared) | provided | False | 0.0021230606362223625 | 0.002338109305128455 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S1 / P1 (direct_register) | allocated | False | 0.0021230606362223625 | 0.002338109305128455 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S2 / P0 (shared) | provided | False | 0.0021230606362223625 | 0.002338109305128455 | True | passed |
| T129_history_64_0_33 | 129 | [64, 0, 33, 129] | S2 / P1 (direct_register) | allocated | False | 0.0021230606362223625 | 0.002338109305128455 | True | passed |

## Graph timing

Milliseconds per complete forward; raw samples are retained in the JSON result.

| T | Arm | p50 | p95 |
|---:|---|---:|---:|
| 32 | rolled_s1_p0 | 0.2677760124206543 | 0.2764128029346466 |
| 32 | rolled_s1_p1 | 0.2521600127220154 | 0.25507840514183044 |
| 32 | rolled_s2_p0 | 0.24447999894618988 | 0.24540160596370697 |
| 32 | rolled_s2_p1 | 0.24755200743675232 | 0.2503680050373077 |
| 32 | m34_s1_p1 | 0.2252800017595291 | 0.22756160497665406 |
| 32 | reusable_pack_fa2 | 0.16684800386428833 | 0.1753600001335144 |
| 96 | rolled_s1_p0 | 0.31385600566864014 | 0.3169920027256012 |
| 96 | rolled_s1_p1 | 0.3409920036792755 | 0.3424255967140198 |
| 96 | rolled_s2_p0 | 0.48767998814582825 | 0.49031360149383546 |
| 96 | rolled_s2_p1 | 0.49612799286842346 | 0.49734399914741517 |
| 96 | m34_s1_p1 | 0.32153600454330444 | 0.323225599527359 |
| 96 | reusable_pack_fa2 | 0.2908160090446472 | 0.2910720109939575 |
| 256 | rolled_s1_p0 | 0.6392319798469543 | 0.6405471920967102 |
| 256 | rolled_s1_p1 | 0.6204800009727478 | 0.6281216144561768 |
| 256 | rolled_s2_p0 | 0.7603200078010559 | 0.7658303856849671 |
| 256 | rolled_s2_p1 | 0.774399995803833 | 0.7785888075828552 |
| 256 | m34_s1_p1 | 0.5440000295639038 | 0.5467135906219482 |
| 256 | reusable_pack_fa2 | 0.46489599347114563 | 0.46694400906562805 |

## M34 bitwise comparison

The timing arm M34 S1/P1 is compared bitwise with rolled S1/P1 before and after timing. Functional cases compare every rolled configuration with the matching M34 stage/P mode. A mismatch is reported without failing the independent FP32 checks.

| T | API before | Graph before | API after | Graph after |
|---:|---:|---:|---:|---:|
| 32 | True | True | True | True |
| 96 | True | True | True | True |
| 256 | True | True | True | True |

## Shared-memory and compiler resources

The fixed allocation is `(64*128 + 2*32*128*stages + 64*32) * 2` bytes; P mode does not change its size.

| Stages | P mode | Shared memory (B) | Delta vs S1/P0 (B) | Delta vs matching M34 (B) | Registers/thread | Register delta vs M34 | Spill bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | P0 (shared) | 36864 | 0 | 0 | 149 | -86 | 0 |
| 1 | P1 (direct_register) | 36864 | 0 | 0 | 150 | -73 | 0 |
| 2 | P0 (shared) | 53248 | 16384 | 0 | 127 | -110 | 0 |
| 2 | P1 (direct_register) | 53248 | 16384 | 0 | 128 | -102 | 0 |

Per-case output/workspace sizes, conversion bytes, source hashes, and runtime versions are recorded in the JSON. Static resource figures do not measure occupancy or runtime utilization.

## Artifacts

Diagnostic JSON: `results/m35-rolled-gqa-diagnostic.json`.
Raw event/artifact directory: `artifacts/m35-diagnostic`.
CUDA extension build output: `artifacts/m35-build/`.
