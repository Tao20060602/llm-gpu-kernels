# M26 NCU 原始指标提取

来源：`artifacts/m26-ncu-20261013T142633`。本页和 [`m26-ncu-summary.json`](../results/m26-ncu-summary.json) 仅转录 CSV/worker JSON/日志内容，不含性能根因或架构判断。CSV 值按原始字符串保存，单位逐字取自 CSV 第二行；不做单位换算或推算。

## 采样记录

| case | CSV 数据行 | worker shape M×N×K | worker runs / profile_only | CSV kernel | block | grid | NCU passes |
|---|---:|---|---|---|---|---|---:|
| m25 | 1 | 4096×4096×4096 | 4 / true | `void <unnamed>::bf16_mma<2>(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int, int)` | `(128, 1, 1)` | `(128, 64, 1)` | 13 |
| m26_padding | 1 | 4096×4096×4096 | 4 / true | `void <unnamed>::tiled_mma<64, 128, 2, 0>(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int, int)` | `(128, 1, 1)` | `(64, 32, 1)` | 13 |
| m26_swizzle | 1 | 4096×4096×4096 | 4 / true | `void <unnamed>::tiled_mma<64, 128, 2, 1>(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int, int)` | `(128, 1, 1)` | `(64, 32, 1)` | 13 |

三个 CSV 都各有 **1 条 kernel 数据行**；对应 profile 日志均显示完成 13 passes，worker JSON 的 `runs` 为 4、`profile_only` 为 true。

## 核心指标

| case | duration (ms) | registers/thread (register/thread) | shared/block (Kbyte/block) | active occupancy metric (%) | active warps (warp/cycle) | eligible warps (warp/cycle) | issue active (%) | tensor pipe active (% elapsed) | HMMA instruction rate (% elapsed) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m25 | 7.222880 | 58 | 16.384000 | 49.241811 | 5.940950 | 0.354783 | 25.131191 | 31.365118 | 15.682559 |
| m26_padding | 5.221824 | 126 | 31.744000 | 24.640988 | 2.952909 | 0.276399 | 20.497437 | 39.102096 | 19.551048 |
| m26_swizzle | 5.141600 | 157 | 25.600000 | 24.478720 | 2.935041 | 0.324326 | 23.894419 | 40.525473 | 20.262737 |

字段名依次为 `gpu__time_duration.sum`、`launch__registers_per_thread`、`launch__shared_mem_per_block`、`sm__warps_active.avg.pct_of_peak_sustained_active`、`smsp__warps_active.avg.per_cycle_active`、`smsp__warps_eligible.avg.per_cycle_active`、`smsp__issue_active.avg.pct_of_peak_sustained_active`、`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`、`sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed`。表中单位直接来自 CSV；warp 值是每 active cycle 的 warp 数。

### 吞吐与流量字段

| case | SM instruction throughput (%) | GPU compute-memory access throughput (%) | GPU compute-memory request throughput (%) | SM memory throughput (%) | DRAM rate (Gbyte/s) | L2 sector hit rate (%) | global L1 tag requests (sectors) | L2 theoretical global sectors (sectors) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m25 | 31.365118 | 75.388239 | 74.122852 | 29.711098 | 38.376049 | 66.652358 | 104857600 | 205520896 |
| m26_padding | 39.102096 | 48.086743 | 49.536503 | 20.486596 | 27.716399 | 66.570526 | 54525952 | 104857600 |
| m26_swizzle | 40.525473 | 49.708195 | 51.197099 | 21.232340 | 28.645471 | 65.976994 | 54525952 | 104857600 |

以上列名原样对应：`sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed`、`gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed`、`gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed`、`sm__memory_throughput.avg.pct_of_peak_sustained_elapsed`、`dram__bytes.avg.per_second`、`lts__t_sector_hit_rate.pct`、`memory_l1_tag_requests_global`、`memory_l2_theoretical_sectors_global`。DRAM 字段是速率，CSV 未提供 DRAM 总字节数；没有计算总字节数。没有 `lts__*throughput` 字段。

### Occupancy 相关原始字段

| case | `launch__occupancy_limit_registers` (block) | `launch__occupancy_limit_shared_mem` (block) | `launch__occupancy_limit_warps` (block) | `launch__occupancy_limit_blocks` (block) | `smsp__warps_active.avg.peak_sustained` (warp) | `derived__pct_occupancy_per_block_size` (%) | `derived__pct_occupancy_per_register_count` (%/register) | `derived__pct_occupancy_per_shared_mem_size` (%/Kbyte) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m25 | 8.000000 | 6.000000 | 12.000000 | 16.000000 | 12.000000 | 3280.000000 | 8170.000000 | 3.637000 |
| m26_padding | 4.000000 | 3.000000 | 12.000000 | 16.000000 | 12.000000 | 774.000000 | 5633.000000 | 2.563000 |
| m26_swizzle | 3.000000 | 4.000000 | 12.000000 | 16.000000 | 12.000000 | 476.000000 | 6665.000000 | 2.155000 |

派生 occupancy 字段保留 CSV 原始文本：其中原始字符可能缺少公式分隔符，不能据此解释成 occupancy 百分比。JSON 还保留三个 `launch__occupancy_per_*` 无单位字段及 allocated shared/registers 字段。

### Shared-memory 相关字段

| case | `derived__memory_l1_conflicts_shared_nway` (CSV unit blank) | `memory_l1_wavefronts_shared` (sectors) | `memory_l1_wavefronts_shared_ideal` (sectors) | `derived__memory_l1_wavefronts_shared_excessive` (CSV unit blank) |
|---|---:|---:|---:|---:|
| m25 | 84 | 251658240 | 150994944 | 100663296 |
| m26_padding | 180 | 125829120 | 75497472 | 50331648 |
| m26_swizzle | 136 | 100663296 | 75497472 | 25165824 |

共享冲突字段原样保存；CSV 单位为空，数值不另行解释为 conflict count。

## PC-sampling stalls

单位和值均按 CSV 原样列出；这些是 PC-sampling `warp`/`branches`/`inst` 数值，不是 runtime 百分比。`not_issued` 对应源字段 `_not_issued`。

| stall reason | unit | m25 base / not_issued | m26_padding base / not_issued | m26_swizzle base / not_issued |
|---|---|---:|---:|---:|
| `barrier` | warp | 85519 / 66902 | 60024 / 52214 | 28792 / 23633 |
| `branch_resolving` | branches | 5363 / 3933 | 5149 / 3881 | 5317 / 4181 |
| `dispatch_stall` | warp | 2799 / 1938 | 3154 / 918 | 3042 / 1046 |
| `drain` | warp | 453 / 395 | 386 / 353 | 412 / 378 |
| `imc_miss` | warp | 47 / 44 | 54 / 52 | 42 / 36 |
| `lg_throttle` | warp | 705 / 648 | 4558 / 4402 | 4618 / 4461 |
| `long_scoreboard` | warp | 38037 / 33064 | 12738 / 10710 | 9810 / 7866 |
| `math_pipe_throttle` | warp | 82079 / 57541 | 139953 / 122658 | 161302 / 137045 |
| `membar` | warp | 0 / 0 | 0 / 0 | 0 / 0 |
| `mio_throttle` | warp | 55408 / 51474 | 19668 / 18295 | 15107 / 13321 |
| `misc` | warp | 0 / 0 | 0 / 0 | 0 / 0 |
| `no_instructions` | inst | 1729 / 1224 | 3122 / 2596 | 2890 / 2404 |
| `not_selected` | warp | 7322 / 0 | 7863 / 0 | 9000 / 0 |
| `selected` | warp | 18071 / 0 | 23286 / 0 | 26390 / 0 |
| `short_scoreboard` | warp | 70360 / 54002 | 1849 / 1636 | 5278 / 4566 |
| `sleeping` | warp | 0 / 0 | 0 / 0 | 0 / 0 |
| `tex_throttle` | warp | 0 / 0 | 0 / 0 | 0 / 0 |
| `wait` | warp | 49687 / 34564 | 49591 / 37642 | 48126 / 37266 |

PC sampling 元数据：`smsp__pcsamp_sample_count` (unit blank) 为 417579 / 331395 / 320126；`smsp__pcsamp_aggregated_passes` (pass) 为 2 / 2 / 2。CSV 未包含非 PC `smsp__warp_issue_stalled_*_per_warp_active.pct`、`smsp__warps_issued*` 字段，因而没有归一化的 stall percentage 或 issued-warp 指标可转录。

## 输入文件与缺失项

- 三个 worker JSON 均记录 shape `4096×4096×4096`、`runs: 4`、`profile_only: true` 和各自 binary SHA-256；完整原始字段见 JSON。
- 三个 `.log` 均记录 NCU 完成 13 passes 并写出对应 `.ncu-repz`；三个 `.import.log` 均为空。
- CSV 未找到 DRAM 总字节字段、明确的 L2-throughput 字段、非 PC normalized stall 百分比或 issued-warps 字段；缺失项保持缺失，没有补算。
