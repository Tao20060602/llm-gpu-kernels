# M29 Baseline NCU metrics

Capture source directory (local artifacts): `artifacts/m29-baseline-ncu-20261013T074100Z`. Each arm CSV contains a header row, a units row, and one kernel data row (667 columns).

## Capture metadata

The worker used BF16 inputs and output with a 4096 × 4096 × 4096 GEMM, seed 429, three warmup calls, then profiled run 4 inside the <code>profile_gemm</code> NVTX range. Worker JSON has no timing field.

| Arm | BF16 reduced-precision reduction | Baseline flag | Accumulator metadata | NCU report | Raw CSV | Worker JSON |
|---|---|---|---|---|---|---|
| torch_default | True | True | null | Local artifact retained at `artifacts/m29-baseline-ncu-20261013T074100Z/torch_default.ncu-repz` (not tracked) | [CSV](../results/provenance/m29-baseline-ncu/torch_default.csv) | [worker JSON](../results/provenance/m29-baseline-ncu/torch_default.worker.json) |
| torch_strict | False | True | null | Local artifact retained at `artifacts/m29-baseline-ncu-20261013T074100Z/torch_strict.ncu-repz` (not tracked) | [CSV](../results/provenance/m29-baseline-ncu/torch_strict.csv) | [worker JSON](../results/provenance/m29-baseline-ncu/torch_strict.worker.json) |
| m27 | null (not a torch.mm arm) | False | native_fp32 | Local artifact retained at `artifacts/m29-baseline-ncu-20261013T074100Z/m27.ncu-repz` (not tracked) | [CSV](../results/provenance/m29-baseline-ncu/m27.csv) | [worker JSON](../results/provenance/m29-baseline-ncu/m27.worker.json) |

## Kernel and launch resources

| Arm | Reported kernel name | Block Size | Grid Size | Registers/thread (requested / allocated) | SMEM/block (total; dynamic; static) | Occupancy limit (registers; shared memory) |
|---|---|---|---|---|---|---|
| torch_default | <code>ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn</code> | <code>(256, 1, 1)</code> | <code>(16, 32, 1)</code> | 222 / 224.000000 register/thread | 74.752000 Kbyte/block; 24.576000 Kbyte/block; 49.152000 Kbyte/block | 1.000000 block; 1.000000 block |
| torch_strict | <code>ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn</code> | <code>(256, 1, 1)</code> | <code>(16, 32, 1)</code> | 222 / 224.000000 register/thread | 74.752000 Kbyte/block; 24.576000 Kbyte/block; 49.152000 Kbyte/block | 1.000000 block; 1.000000 block |
| m27 | <code>void &lt;unnamed&gt;::scheduled_mma&lt;2, 2, 1&gt;(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int, int)</code> | <code>(128, 1, 1)</code> | <code>(64, 32, 1)</code> | 128 / 128.000000 register/thread | 25.600000 Kbyte/block; 24.576000 Kbyte/block; 0 byte/block | 4.000000 block; 4.000000 block |

In this capture, torch_default and torch_strict report the same kernel name and the same block/grid/register/SMEM fields. That equality does not establish identical internal execution paths or instruction sequences.

## Tensor, occupancy, issue, DRAM, and L2 counters

Metric values and units below are copied from the raw CSV. A blank unit is shown as such.

### Core counters

| Raw NCU field | torch_default | torch_strict | m27 |
|---|---|---|---|
| <code>gpu__time_duration.sum</code> | <code>3.854688</code> ms | <code>3.845248</code> ms | <code>5.002208</code> ms |
| <code>sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active</code> | <code>48.935161</code> % | <code>48.929875</code> % | <code>43.281904</code> % |
| <code>sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed</code> | <code>47.367341</code> % | <code>47.345400</code> % | <code>42.904804</code> % |
| <code>sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active</code> | <code>24.467581</code> % | <code>24.464937</code> % | <code>21.640952</code> % |
| <code>sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed</code> | <code>23.683670</code> % | <code>23.672700</code> % | <code>21.452402</code> % |
| <code>sm__warps_active.avg.pct_of_peak_sustained_active</code> | <code>16.663055</code> % | <code>16.662960</code> % | <code>32.467061</code> % |
| <code>sm__warps_active.avg.per_cycle_active</code> | <code>7.998266</code> warp | <code>7.998221</code> warp | <code>15.584189</code> warp |
| <code>smsp__warps_active.avg.peak_sustained</code> | <code>12.000000</code> warp | <code>12.000000</code> warp | <code>12.000000</code> warp |
| <code>smsp__warps_active.avg.per_cycle_active</code> | <code>1.999869</code> warp | <code>1.999870</code> warp | <code>3.898785</code> warp |
| <code>smsp__warps_eligible.avg.per_cycle_active</code> | <code>0.090195</code> warp | <code>0.090199</code> warp | <code>0.338588</code> warp |
| <code>smsp__issue_active.avg.pct_of_peak_sustained_active</code> | <code>5.892838</code> % | <code>5.893025</code> % | <code>25.361531</code> % |
| <code>smsp__issue_active.avg.per_cycle_active</code> | <code>0.06</code> (unit blank in CSV) | <code>0.06</code> (unit blank in CSV) | <code>0.25</code> (unit blank in CSV) |
| <code>sm__inst_issued.avg.pct_of_peak_sustained_elapsed</code> | <code>5.703410</code> % | <code>5.700768</code> % | <code>25.125256</code> % |
| <code>sm__inst_issued.avg.per_cycle_active</code> | <code>0.235688</code> inst/cycle | <code>0.235662</code> inst/cycle | <code>1.013843</code> inst/cycle |
| <code>dram__bytes.avg.per_second</code> | <code>8.692862</code> Gbyte/s | <code>8.709343</code> Gbyte/s | <code>28.490561</code> Gbyte/s |
| <code>dram__bytes.min.per_second</code> | <code>8.690592</code> Gbyte/s | <code>8.708132</code> Gbyte/s | <code>28.488795</code> Gbyte/s |
| <code>dram__bytes.max.per_second</code> | <code>8.695672</code> Gbyte/s | <code>8.710629</code> Gbyte/s | <code>28.493273</code> Gbyte/s |
| <code>dram__bytes.sum.per_second</code> | <code>69.542898</code> Gbyte/s | <code>69.674745</code> Gbyte/s | <code>227.924488</code> Gbyte/s |
| <code>lts__t_sector_hit_rate.pct</code> | <code>92.853685</code> % | <code>92.858372</code> % | <code>66.936481</code> % |
| <code>lts__t_sectors_lookup_miss.sum</code> | <code>7340595.000000</code> sector | <code>7345428.000000</code> sector | <code>34538149.000000</code> sector |
| <code>lts__t_sectors_lookup_miss_data_promoted.sum</code> | <code>3670016.000000</code> sector | <code>3670016.000000</code> sector | <code>0</code> sector |
| <code>lts__cycles_active.avg</code> | <code>5573188.312500</code> cycle | <code>5562386.812500</code> cycle | <code>6181370.031250</code> cycle |
| <code>lts__cycles_elapsed.avg</code> | <code>5635386.000000</code> cycle | <code>5634196.000000</code> cycle | <code>6188085.000000</code> cycle |

### Launch occupancy fields as exported

| Raw NCU field | torch_default | torch_strict | m27 |
|---|---|---|---|
| <code>launch__occupancy_limit_blocks</code> | <code>16.000000</code> block | <code>16.000000</code> block | <code>16.000000</code> block |
| <code>launch__occupancy_limit_registers</code> | <code>1.000000</code> block | <code>1.000000</code> block | <code>4.000000</code> block |
| <code>launch__occupancy_limit_shared_mem</code> | <code>1.000000</code> block | <code>1.000000</code> block | <code>4.000000</code> block |
| <code>launch__occupancy_limit_warps</code> | <code>6.000000</code> block | <code>6.000000</code> block | <code>12.000000</code> block |
| <code>launch__occupancy_per_block_size</code> | <code>73</code> (unit blank in CSV) | <code>73</code> (unit blank in CSV) | <code>398</code> (unit blank in CSV) |
| <code>launch__occupancy_per_register_count</code> | <code>2056</code> (unit blank in CSV) | <code>2056</code> (unit blank in CSV) | <code>3248</code> (unit blank in CSV) |
| <code>launch__occupancy_per_shared_mem_size</code> | <code>1592</code> (unit blank in CSV) | <code>1592</code> (unit blank in CSV) | <code>1248</code> (unit blank in CSV) |
| <code>derived__pct_occupancy_per_block_size</code> | <code>146.000000</code> % | <code>146.000000</code> % | <code>818.000000</code> % |
| <code>derived__pct_occupancy_per_register_count</code> | <code>4112.000000</code> %/register | <code>4112.000000</code> %/register | <code>6665.000000</code> %/register |
| <code>derived__pct_occupancy_per_shared_mem_size</code> | <code>3.184000</code> %/Kbyte | <code>3.184000</code> %/Kbyte | <code>2.563000</code> %/Kbyte |
| <code>derived__lts__t_sectors_lookup_miss_data_promoted_pct</code> | <code>49.996165</code> % | <code>49.963270</code> % | <code>0</code> % |

The NCU <code>gpu__time_duration.sum</code> values are profiler measurements and are not treated here as formal GEMM speed results.

## Static SASS observation

The imported Torch-default report's source/SASS export contains 952 instruction rows for the reported kernel. Within that view, the observed static instruction-row counts are: 64 <code>HMMA.16816.F32.BF16</code>, 24 <code>LDSM.16.M88.4</code>, 21 <code>LDGSTS.E.BYPASS.LTC128B.128.CONSTANT</code>, 6 <code>BAR.SYNC*</code>, and 4 global <code>STG.E.EF.64</code> rows. These are rows in the exported SASS view, not runtime proportions. The exported source CSV is available as [torch_default-source.csv](../results/provenance/m29-baseline-ncu/torch_default-source.csv).

## DRAM and L1/L2 counter availability

The CSV headers contain <code>dram__bytes.{avg,min,max,sum}.per_second</code> rate metrics, but no raw DRAM read-byte, write-byte, or total-byte counter. No global L1 load-sector field matching <code>l1tex__t_sectors_pipe_lsu_mem_global_op_ld</code> is present. No L2 byte or throughput field is present. The available L2 sector fields are lookup-miss and lookup-miss-data-promoted counters; they are not total L2 traffic.

## PC sampling stall counters

The table includes each exported <code>smsp__pcsamp_warps_issue_stalled_*</code> field, including <code>_not_issued</code> variants. Values and units are shown as reported.

| Raw NCU field | Unit | torch_default | torch_strict | m27 |
|---|---|---|---|---|
| <code>smsp__pcsamp_warps_issue_stalled_barrier</code> | warp | <code>674</code> | <code>662</code> | <code>33045</code> |
| <code>smsp__pcsamp_warps_issue_stalled_barrier_not_issued</code> | warp | <code>598</code> | <code>605</code> | <code>26391</code> |
| <code>smsp__pcsamp_warps_issue_stalled_branch_resolving</code> | branches | <code>416</code> | <code>477</code> | <code>4008</code> |
| <code>smsp__pcsamp_warps_issue_stalled_branch_resolving_not_issued</code> | branches | <code>363</code> | <code>414</code> | <code>3207</code> |
| <code>smsp__pcsamp_warps_issue_stalled_dispatch_stall</code> | warp | <code>10</code> | <code>15</code> | <code>2782</code> |
| <code>smsp__pcsamp_warps_issue_stalled_dispatch_stall_not_issued</code> | warp | <code>10</code> | <code>14</code> | <code>1338</code> |
| <code>smsp__pcsamp_warps_issue_stalled_drain</code> | warp | <code>17</code> | <code>14</code> | <code>283</code> |
| <code>smsp__pcsamp_warps_issue_stalled_drain_not_issued</code> | warp | <code>16</code> | <code>14</code> | <code>262</code> |
| <code>smsp__pcsamp_warps_issue_stalled_imc_miss</code> | warp | <code>46</code> | <code>54</code> | <code>67</code> |
| <code>smsp__pcsamp_warps_issue_stalled_imc_miss_not_issued</code> | warp | <code>46</code> | <code>54</code> | <code>66</code> |
| <code>smsp__pcsamp_warps_issue_stalled_lg_throttle</code> | warp | <code>270</code> | <code>291</code> | <code>1721</code> |
| <code>smsp__pcsamp_warps_issue_stalled_lg_throttle_not_issued</code> | warp | <code>251</code> | <code>270</code> | <code>1646</code> |
| <code>smsp__pcsamp_warps_issue_stalled_long_scoreboard</code> | warp | <code>272</code> | <code>276</code> | <code>5642</code> |
| <code>smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued</code> | warp | <code>262</code> | <code>269</code> | <code>4631</code> |
| <code>smsp__pcsamp_warps_issue_stalled_math_pipe_throttle</code> | warp | <code>225907</code> | <code>225644</code> | <code>131617</code> |
| <code>smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued</code> | warp | <code>221997</code> | <code>221819</code> | <code>104211</code> |
| <code>smsp__pcsamp_warps_issue_stalled_membar</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_membar_not_issued</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_mio_throttle</code> | warp | <code>637</code> | <code>597</code> | <code>3736</code> |
| <code>smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued</code> | warp | <code>566</code> | <code>537</code> | <code>3507</code> |
| <code>smsp__pcsamp_warps_issue_stalled_misc</code> | warp | <code>2</code> | <code>2</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_misc_not_issued</code> | warp | <code>2</code> | <code>2</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_no_instructions</code> | inst | <code>230</code> | <code>232</code> | <code>2425</code> |
| <code>smsp__pcsamp_warps_issue_stalled_no_instructions_not_issued</code> | inst | <code>212</code> | <code>214</code> | <code>2018</code> |
| <code>smsp__pcsamp_warps_issue_stalled_not_selected</code> | warp | <code>4170</code> | <code>4192</code> | <code>6506</code> |
| <code>smsp__pcsamp_warps_issue_stalled_not_selected_not_issued</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_selected</code> | warp | <code>7941</code> | <code>7870</code> | <code>20066</code> |
| <code>smsp__pcsamp_warps_issue_stalled_selected_not_issued</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_short_scoreboard</code> | warp | <code>145</code> | <code>167</code> | <code>37020</code> |
| <code>smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued</code> | warp | <code>123</code> | <code>142</code> | <code>29038</code> |
| <code>smsp__pcsamp_warps_issue_stalled_sleeping</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_sleeping_not_issued</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_tex_throttle</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_tex_throttle_not_issued</code> | warp | <code>0</code> | <code>0</code> | <code>0</code> |
| <code>smsp__pcsamp_warps_issue_stalled_wait</code> | warp | <code>27084</code> | <code>27338</code> | <code>53898</code> |
| <code>smsp__pcsamp_warps_issue_stalled_wait_not_issued</code> | warp | <code>26333</code> | <code>26623</code> | <code>40655</code> |

| Raw NCU field | Unit | torch_default | torch_strict | m27 |
|---|---|---|---|---|
| <code>smsp__pcsamp_sample_count</code> | (unit blank in CSV) | <code>267821</code> | <code>267831</code> | <code>302816</code> |

The complete selected counter set, including exact raw values, units, worker JSON, report paths, CSV shape, and availability fields, is stored in [m29-baseline-ncu-summary.json](../results/m29-baseline-ncu-summary.json).

## Design interpretation

The captured Torch kernel's name, grid, thread count, and shared allocation are consistent with a 256×128 output tile and BK32/S3. M27 computes a 64×128 tile. Torch has lower achieved occupancy but higher tensor-pipe activity and L2 sector hit rate. This makes increasing output-tile reuse a stronger next hypothesis than treating occupancy or BK alone as the target. The next controlled experiment is [M31](M31_OUTPUT_TILE_DESIGN.md).

Raw PC sampling counts are not percentages of execution time, and different warp counts/schedules affect their comparison. They do not establish that removing the largest stall category will improve performance. The exported `derived__pct_occupancy_*` fields contain anomalous values and are retained only as raw evidence, not interpreted as achieved occupancy. Torch's lower issue-active/eligible-warp metrics are likewise not evidence that it is slower; performance is established by the separate Graph timings. This capture narrows the hypotheses but does not isolate one cause or expose cuBLAS's full implementation.
