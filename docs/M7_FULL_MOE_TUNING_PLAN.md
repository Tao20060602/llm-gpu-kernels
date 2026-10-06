# M7: Training-Only Full-MoE Tile Search

## Purpose and scope

M7 searches a small set of shared GEMM tile configurations for the complete
ranked fixed-route MoE forward. Every candidate uses
`prepare_configured_moe(...)`, including the same ranked dispatch, prefix,
fused gate/up plus SwiGLU, down GEMM, and combine path. The original M4/M5/M6
entry points remain unchanged. The scan measures complete forwards; it is not
a pregrouped-only GEMM result, a production claim, or a comparison against a
mature backend.

The supported training domain is the current RTX 3080 Laptop GPU / SM86,
PyTorch 2.7.1+cu128, CUDA runtime 12.8, Triton 3.3.1, with
`E=128, H=2048, I=768, topk=8`. The inputs are synthetic BF16 tensors with
finite FP32 routing weights. The experiment does not use a model checkpoint,
router, serving stack, or expert-parallel communication.

## Fixed training matrix

Use only `T={64,512,4096}` and the two route patterns `balanced` and `hot16`,
all generated with seed `409`. For a given T, both route cases use identical
X, gate/up weights, down weights, and routing weights; only IDs differ.

- `balanced`: `arange(T * topk) % E`, reshaped to `[T, topk]`. At these T
  values each expert receives exactly 4, 32, or 256 route rows respectively.
- `hot16`: the existing `moe_benchmark.make_inputs(pattern="hot")` helper,
  which selects 16 experts for this E and chooses distinct top-k IDs per token.

This gives six training cases. Route counts are not copied from the GPU or
used to choose a tile. The later policy uses only `mu = T * topk / E`, a
shape-derived estimate. Balanced and hot16 cases at the same T therefore
share one policy bucket and are scored together.

## Candidate configurations

The stable candidate index is the tie-break order. Gate/up and down initially
share each configuration so the scan has one controlled tile choice per
candidate.

| Index | BM | BN | BK | Warps | Stages | Purpose |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 32 | 64 | 32 | 4 | 3 | Existing baseline |
| 1 | 16 | 64 | 32 | 4 | 3 | Smaller M tile |
| 2 | 64 | 64 | 32 | 4 | 3 | Larger M tile; vLLM large-tile dimensions |
| 3 | 16 | 32 | 64 | 4 | 3 | vLLM small-tile dimensions |
| 4 | 32 | 32 | 32 | 4 | 3 | Smaller N tile |
| 5 | 32 | 128 | 32 | 4 | 3 | Larger N tile |
| 6 | 32 | 256 | 32 | 4 | 3 | Largest N tile in this scan |
| 7 | 32 | 64 | 64 | 4 | 3 | Larger K tile |
| 8 | 32 | 128 | 64 | 4 | 3 | Larger N and K tiles |
| 9 | 64 | 128 | 32 | 4 | 3 | Larger M and N tiles |
| 10 | 32 | 64 | 32 | 8 | 3 | More warps |
| 11 | 32 | 64 | 32 | 4 | 2 | Fewer stages |

Candidate compilation, resource failures, and numerical failures are retained
in the scan. Each candidate records fused gate/up and down registers, spills,
and shared memory when compilation exposes that metadata. No candidate is
silently removed. Resource metadata and full-forward latency are observations;
without hardware counters, they do not establish occupancy or memory-traffic
causality.

## Correctness and timing protocol

For each training case, construct one full FP32 reference with
`moe_reference` and retain it through all 12 candidate comparisons. Validate
the baseline and each candidate against that same reference before timing and
again after Graph timing. Require finite output, relative RMS at most `0.008`,
and normalized max error at most `0.008`. A candidate failing numerical
validation is invalid for selection and its failure is recorded. The reference
and checks are never timed.

Each baseline/candidate pair uses separate prepared workspaces over the same
fixed input tensors. Capture one Graph per path, each with eight complete
forwards per replay. Use four balanced `AB, BA, BA, AB` rounds, six samples per
round, and two Graph replays per sample. Normalize event and wall samples by
both calls per sample and forwards per replay. Preparation, compilation,
warmup, Graph capture, correctness checks, hashing, telemetry, and allocator
cleanup stay outside the measured event windows. The reported quantity is
milliseconds per complete forward; derived FLOP/s is mathematical work divided
by latency, not a hardware counter.

Record all `12 × 6 = 72` candidate rows, including errors and resource failures.
Record the full input and route hashes before and after each training case,
per-case telemetry, configuration, launch count, workspace bytes, pair peak
allocated bytes, correctness metrics, and raw per-round samples. The reported
peak includes other live allocations such as inputs, the FP32 reference, and
Graph pools; it is not pure operator workspace.

## Selection and longer recheck

Use three fixed shape-only buckets for this domain:

| Bucket | `mu = T * topk / E` |
| --- | ---: |
| small | `mu <= 8` |
| medium | `8 < mu <= 64` |
| large | `mu > 64` |

For each candidate and bucket, score the two corresponding route cases by the
geometric mean of candidate p50 divided by paired baseline p50. A candidate is
eligible only if it passed correctness and measurement for all six training
cases, its bucket score is at most `0.95`, and neither balanced nor hot16 case
has a ratio above `1.05`. Select the eligible candidate with the lowest score;
on an exact tie, select the lower candidate index. If no candidate qualifies,
select the baseline.

Recheck only the initially selected candidate for each bucket on that bucket's
balanced and hot16 cases. The recheck uses the same seed and inputs, verifies
the hashes match the scan, and uses eight samples per round and four Graph
replays per sample, still with eight forwards per replay. Apply the same
correctness and ratio criteria. If a non-baseline choice fails recheck, freeze
the baseline for that bucket; do not try the next-ranked candidate. A baseline
failure prevents policy creation.

## Frozen policy and holdout boundary

After all scan and recheck records are complete, write
`results/m7-policy.json` with status `frozen_before_holdout`, the exact domain,
thresholds, selected configurations, source run, relative training-record
path, explicit source hashes, and the seed/token/route protocol. The loader
uses only `(T,E,topk,H,I)` and falls back to the baseline outside the exact
`E/H/I/topk` domain. It does not inspect IDs or counts.

The training record includes the policy file SHA256. To avoid a circular hash,
`training_record_sha256` in the policy hashes canonical compact JSON for the
complete final training record after removing only its top-level
`policy_sha256` field. The policy SHA is the SHA256 of the exact policy file
bytes. This scope is fixed in both artifacts and checked by the policy loader.

The M5 random holdout cases (seeds 73/991, T193/1024/2049) must not be opened,
scored, or used to alter the candidate set, bucket thresholds, or selection
rule before freeze. Run the frozen policy on them only as confirmation. Report
holdout results as measured even if they regress; do not retune against them.

## Run and outputs

Run on the already validated project environment with:

```bash
python -m llm_gpu_kernels.moe_full_tuning
```

The runner refuses to overwrite an existing `results/m7-policy.json`. It saves
incremental scan results to a unique
`results/m7-full-moe-tuning-<UTC timestamp>.json` file and writes the frozen
policy only after successful baseline checks and rechecks. The source hash list
is explicit and covers only the experiment plan, runner, policy loader,
reference/measurement helpers, and launcher/kernel files used by M7.
