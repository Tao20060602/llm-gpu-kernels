# M37 compact API sweep

Searches:

```text
rg -n --hidden -g '!.git/**' 'prepare_cuda_gqa_compact|CompactCudaGqaPlan|p_mode' src scripts tests docs results artifacts
rg -n --hidden -g '!.git/**' 'prepare_cuda_gqa_compact|CompactCudaGqaPlan' src scripts tests docs
```

## Current hits and dispositions

| Hit | Disposition |
|---|---|
| `src/llm_gpu_kernels/attention_cuda_compact.py` public prepare wrapper | `_P_MODES`, its `p_mode` argument/check, and the native argument are removed. It loads only `llm_gpu_kernels_m36_compact_gqa_sm86_v2` from `artifacts/m36-build-v2`; native construction takes five arguments. |
| `src/llm_gpu_kernels/cuda_gqa_attention_compact.cu` native plan | Root removed native `p_mode` state/argument, DirectP selection, and shared-P path. Its constructor now matches the five-argument wrapper. |
| `scripts/run_m36_compact_gqa_check.py` compact calls | Both public compact calls pass `stages` only. The compact `p_mode=1` config/output field remains fixed measurement metadata; its other `p_mode` uses select the M34 comparison arm. New runs use `m37-compact-v2` output paths and record the loaded module name. |
| `scripts/profile_m36_gqa.py` compact worker | Native call takes five arguments and loads v2. Worker JSON records `extension_module` from the loaded binary stem; `p_mode: 1` remains fixed measurement metadata. `scripts/run_m36_gqa_ncu.sh` now checks the v2 binary. |
| `scripts/run_m37_compact_mask_check.py` public calls and native call | Public v2 calls use the prepare wrapper without `p_mode`. The six-argument native call is confined to `_load_v1_control()`, which loads the preserved M36 v1 binary explicitly for benchmark comparison; it is not a product path or fallback. |
| `tests/e2e/test_compact_attention.py` | Calls the public wrapper with stages only; no compact `p_mode` choice. |
| M34/M35 code | `attention_cuda.py`, `cuda_gqa_attention.cu`, `run_m34_attention_check.py`, `attention_cuda_rolled.py`, `cuda_gqa_attention_rolled.cu`, and `run_m35_rolled_gqa_check.py` keep their genuine two-mode experiments. |
| Historical M34/M35/M36 result and build records | `results/m34-cuda-gqa-diagnostic.json`, `results/m35-rolled-gqa-diagnostic.json`, `results/m36-compact-gqa-diagnostic.json`; `artifacts/m34-*`, `artifacts/m35-*`, and `artifacts/m36-*` manifests/static evidence remain historical records. They were not rewritten. |
| Historical NCU records | `artifacts/m36-gqa-ncu-20261014T083723Z/*`, `results/provenance/m36-gqa-ncu/*`, and the captured M36 table retain their original fields and values. The completed M36 capture is v1 evidence from commit `3338707`; current profiling loads v2. |
| Docs mentioning compact `p_mode` or `prepare_cuda_gqa_compact` | M37 proposal/report, current-status and M36 design prose describe the API change or historical experiment; they are not caller sites. The M36 reports identify commit `3338707` for old measurements. |

The preserved v1 binary at `artifacts/m37-control/llm_gpu_kernels_m36_compact_gqa_sm86_v1.so` has the same SHA-256 as the M36 build artifact: `99970108a0256485c92aa6b4009d8cad722961ffe00823a60d841ad0c5028122`. Existing M36 result JSON and source fingerprints were left untouched.

## CPU build evidence

The wrapper loader built and loaded the sole v2 module after sourcing `scripts/activate-cutlass.sh`. The SM86 cubin was extracted and inspected locally. Per-stage resources are: S1 221 registers, 32768 dynamic shared bytes, no stack/spills, 256 HMMA, 288 LDSM, 0 LDL/STL; S2 231 registers, 49152 dynamic shared bytes, no stack/spills, 256 HMMA, 288 LDSM, 0 LDL/STL. HMMA/LDSM/LDL/STL counts were read from cuobjdump SASS and cross-checked with nvdisasm.

`results/provenance/m37-build/` contains the loader/build log, cuobjdump resource output, and concise resource summary JSON. Full cubin and SASS dumps remain under local `artifacts/m37-build/`. This subtask ran CPU build/load and static inspection only; root ran the M37 GPU benchmark and e2e verification separately.

## Net lines by area

Counts cover the M37 task files changed in this worktree; new text files count as additions. The self-counting `docs/M37_API_SWEEP.md` is excluded from the docs total.

| Area | Added | Removed | Net |
|---|---:|---:|---:|
| Kernel (`cuda_gqa_attention_compact.cu`) | 46 | 35 | +11 |
| Wrapper/callers (compact wrapper, M36 runner, profile worker, NCU driver) | 38 | 35 | +3 |
| Verification (M37 runner and e2e source) | 886 | 0 | +886 |
| Docs (README, status/design/report notes, new M37 design and report, historical NCU report) | 196 | 13 | +183 |
| Evidence (`results/provenance/m37-build/`, three text files) | 96 | 0 | +96 |

The net counts exclude ignored `.so`, cubin, full SASS dumps and generated Python bytecode. No tests or GPU kernels were run by this subtask; no commit was made.
