# M8: Segmented-KV GQA Prefill Cost Baseline

## Goal and boundary

M8 measures the GPU-resident cost of packing four fixed K/V segments and calling
the unmodified public FlashAttention 2 API. It establishes the strong packed
baseline before any segmented attention kernel is considered. It does not run
NanoKV, a model, a router, or H2D transfers, and it does not implement a custom
attention kernel.

The contract follows the native Linux operator handoff at
`/home/tmz/.codex/worktrees/nanokv-native-linux/nanokv/docs/SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md`:
query `Q[T,32,128]` and
four ordered K/V segments `[selected history, sink, previous recent, current]`,
each `K/V[L,8,128]`, all contiguous BF16 on the same GPU. The current segment
has length `T`. All three history segments are visible to every query; current
key position `j` is visible only when `j <= query position i`. GQA maps
`kv_head = q_head // 4`; softmax scale is `1/sqrt(128)`. Inputs are assumed to
have any model-required positional processing already applied.

## Backend and timed paths

The backend is the official unmodified FlashAttention 2.8.3.post1 public
`flash_attn_func` from the verified isolated overlay recorded in
`results/provenance/flash-attention-2.8.3.post1-runtime.json`. It receives
`[1,T,32,128]` Q and `[1,S,8,128]` K/V views, with `dropout_p=0.0`, explicit
scale, and `causal=True`. The public API's bottom-right causal alignment makes
the first `S-T` keys visible to every Q while applying the required causal mask
to the current T keys.

Three timings are kept separate:

1. **Full chain:** `torch.cat` the four K and V segments into reusable
   preallocated buffers, then call public FA2.
2. **Prepacked FA2:** the same public FA2 call with the same already-packed K/V;
   packing is outside this timing.
3. **Pack-only diagnostic:** reusable-buffer K and V concatenation with no
   attention call.

Pack-only is not subtracted from the full-chain time. Their independent
measurements provide context; the full chain is the direct cost baseline.
Public FA2 returns a newly allocated eager output. That behavior is included in
the eager call timing. CUDA Graph output-pool behavior is measured separately.

## Fixed performance matrix

Seed `819`; `T={64,256,1024,4096}` and selected-history length
`{0,512,2048}`. Sink is 64, previous recent is 512, and current length is T.
This yields 12 cases. Q and the common sink/recent/current data are reproducible
for a given T; selected history varies only with its requested length. Inputs
are GPU-resident before any timed call. There is no repeat-KV expansion on the
timed path.

For each case, report CUDA-event and synchronized wall samples for full-chain
and prepacked FA2. Use four balanced `AB, BA, BA, AB` rounds, eight samples per
round, and two calls per sample. Also report eight-forward CUDA Graphs for each
arm; time two graph replays per sample and normalize by both replays and eight
forwards. Pack-only is separately measured with the same API sample schedule.
Preparation, input creation and hashes, warmup, graph capture, correctness,
telemetry, cleanup, and allocator inspection remain outside timed windows.

## Correctness and stream checks

The small FP32 oracle explicitly repeats each KV head four times, applies the
bottom-right mask, computes FP32 scores/softmax/value accumulation, and is used
only for `T<=65` and history length `<=91`. Require finite output,
`relative_rms <= 0.008`, and `normalized_max <= 0.008`. Functional cases cover
empty history, non-aligned segment tails, extreme logits, explicit GQA head
mapping, `T=0`, and a Q65/history91 boundary case. For the performance matrix, full-chain output is compared
to prepacked public FA2 on the identical Q/K/V input; both use the same public
operator and exact packed values, so this is an arm-consistency check rather
than an independent numerical oracle.

A designated `T=9` small case captures fixed-address full-chain and prepacked
graphs, updates Q and every K/V segment on a non-default stream, explicitly
orders that stream before graph replay, checks both graph outputs against the
FP32 oracle, restores the original inputs, replays again, and verifies the
original input hash. No updates or hashing occur in timed regions.

## Records and limits

The unique `results/m8-gqa-baseline-*.json` record contains exact source hashes
for this plan, wrapper, and runner; FA2 wheel/runtime provenance and the actual
imported module path; input hashes; all raw timing samples; per-case telemetry;
and allocator peaks. Allocator totals include live inputs, outputs, graph pools,
and runtime state, so they are not reported as operator-only workspace.

This benchmark isolates GPU-resident packing plus FA2. It does not include
selected-history CPU gather, H2D, NanoKV store/update, model execution, or
serving. Results answer the cost-baseline question for the listed shapes and
runtime only; they do not establish the value of a later kernel or a system-wide
speedup.

## Run

From the operator repository root, source the project environment and official
FA2 overlay, then run:

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_baseline_benchmark
```
