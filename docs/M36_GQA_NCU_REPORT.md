# M36 GQA NCU report

## Capture and scope

This completed M36 NCU capture is historical v1 evidence from commit `3338707`, which can reproduce the v1 binary. The current `profile_m36_gqa.py` entry still names its arm `m36_s2` but loads compact v2; newly written worker JSON records the actual `extension_module` from the loaded binary stem. The captured CSVs and worker records below remain unchanged.

Source capture: `artifacts/m36-gqa-ncu-20261014T083723Z`. The `.ncu-repz` reports remain local there. Three raw CSV files and three worker JSON files were copied byte-for-byte to [`results/provenance/m36-gqa-ncu/`](../results/provenance/m36-gqa-ncu/); source hashes are recorded in [`m36-gqa-ncu-summary.json`](../results/m36-gqa-ncu-summary.json). Each CSV has 667 columns and exactly three rows: header, units, and one kernel data row. Values and units below are preserved from those rows without rounding or conversion.

| Arm | Tokens | Seed | History lengths | Stages | p_mode | Worker declared SMEM | Warmups | Profiled runs |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `m34_s1` | 256 | 436 | `[2048, 64, 512]` | 1 | 1 | 36864 B | 3 | 1 |
| `m34_s2` | 256 | 436 | `[2048, 64, 512]` | 2 | 1 | 53248 B | 3 | 1 |
| `m36_s2` | 256 | 436 | `[2048, 64, 512]` | 2 | 1 | 49152 B | 3 | 1 |

## Kernel and launch resources

| Field | m34_s1 | m34_s2 | m36_s2 |
|---|---:|---:|---:|
| `Kernel Name` | `void <unnamed>::gqa_forward<1, 1>(const __nv_bfloat16 *, <unnamed>::Segments, __nv_bfloat16 *, int, long)` (unit blank) | `void <unnamed>::gqa_forward<2, 1>(const __nv_bfloat16 *, <unnamed>::Segments, __nv_bfloat16 *, int, long)` (unit blank) | `void <unnamed>::compact_gqa_forward<2, 1>(const __nv_bfloat16 *, <unnamed>::Segments, __nv_bfloat16 *, int, long)` (unit blank) |
| `Block Size` | `(128, 1, 1)` (unit blank) | `(128, 1, 1)` (unit blank) | `(128, 1, 1)` (unit blank) |
| `Grid Size` | `(4, 32, 1)` (unit blank) | `(4, 32, 1)` (unit blank) | `(4, 32, 1)` (unit blank) |
| `launch__registers_per_thread` | `223` register/thread | `230` register/thread | `230` register/thread |
| `launch__registers_per_thread_allocated` | `224.000000` register/thread | `232.000000` register/thread | `232.000000` register/thread |
| `launch__shared_mem_per_block (total)` | `37.888000` Kbyte/block | `54.272000` Kbyte/block | `50.176000` Kbyte/block |
| `launch__shared_mem_per_block_dynamic` | `36.864000` Kbyte/block | `53.248000` Kbyte/block | `49.152000` Kbyte/block |
| `launch__shared_mem_per_block_static` | `0` byte/block | `0` byte/block | `0` byte/block |
| `launch__shared_mem_per_block_allocated` | `37.888000` Kbyte/block | `54.272000` Kbyte/block | `50.176000` Kbyte/block |
| `launch__occupancy_limit_registers` | `2.000000` block | `2.000000` block | `2.000000` block |
| `launch__occupancy_limit_shared_mem` | `2.000000` block | `1.000000` block | `2.000000` block |

## Runtime counters

| Field | m34_s1 | m34_s2 | m36_s2 |
|---|---:|---:|---:|
| `gpu__time_duration.sum (NCU profiler duration)` | `561.024000` us | `605.440000` us | `527.520000` us |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | `14.050778` % | `8.333455` % | `14.273184` % |
| `sm__warps_active.avg.per_cycle_active` | `6.744374` warp | `4.000059` warp | `6.851128` warp |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | `33.831614` % | `29.240082` % | `35.808595` % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `28.700719` % | `25.601398` % | `30.947114` % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | `16.915807` % | `14.620041` % | `17.904298` % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed` | `14.350359` % | `12.800699` % | `15.473557` % |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.343311` warp | `0.269417` warp | `0.407558` warp |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `29.109737` % | `26.941731` % | `32.983465` % |

The profiler duration is NCU output, not an unprofiled benchmark speed result. The active-warp field is reported as captured; it is not used to estimate latency.

## PC sampling

These are raw sampled counts, not percentages or time shares. `not_issued` values are shown beside their matching fields.

| PC-sampling raw field | m34_s1 | m34_s2 | m36_s2 |
|---|---:|---:|---:|
| `barrier (count / not_issued)` | `1328` / `1103` warp | `753` / `753` warp | `1137` / `961` warp |
| `math_pipe_throttle (count / not_issued)` | `13425` / `11756` warp | `15518` / `15477` warp | `14833` / `12897` warp |
| `short_scoreboard (count / not_issued)` | `603` / `537` warp | `792` / `784` warp | `532` / `462` warp |
| `smsp__pcsamp_sample_count` | `32928` (unit blank) | `38084` (unit blank) | `31071` (unit blank) |

## Reading limits

M36 S2 reports a shared-memory launch occupancy limit of 2 blocks versus 1 for M34 S2; the register launch limit is 2 blocks for both. The captured `sm__warps_active.avg.pct_of_peak_sustained_active` values are `8.333455`% for M34 S2 and `14.273184`% for M36 S2. Each arm has one profiled launch. These measurements describe the captured launches; they do not establish benchmark speed or isolate a performance cause. PC-sampling counts should not be compared as time percentages, and sample counts differ by arm.
