# M9: Minimal Four-Segment GQA Attention Prototype

## Goal and scope

M9 adds one prepared Triton forward kernel that consumes four GPU-resident K/V
segments directly. It is a correctness-first prototype for the selected-history,
sink, previous-recent, current layout. It does not run NanoKV, a router, a model,
CPU gather, H2D transfer, or FlashAttention inside the candidate kernel. The
unmodified public FA2 wrapper and M8 records remain the strong packed baseline.

The kernel is fixed to one batch, contiguous BF16
`Q[T,32,128]`, and four contiguous `(K[L,8,128],V[L,8,128])` segments in the
order selected history, sink, previous recent, current. Current K and V lengths
must equal T. The three history segments are visible to every Q row; current
position `j` is visible only for `j <= query_position`. GQA maps each group of
four contiguous Q heads to one KV head. No KV duplication or final KV packing is
performed by this kernel.

## Prepared API and first configuration

`prepare_segmented_kernel(q, segments, config=AttentionConfig())` validates
metadata, allocates the BF16 output, compiles and warms the fixed launch, and
returns an owner with `.run()`, `.output`, `.kernel`, `.config`, `.grid`, and
`.workspace_bytes`. `AttentionConfig` has `token_tile=32`, `key_tile=64`,
`head_group=1`, `warps=4`, and `stages=2` by default. The first launch is thus
token tile 32, key tile 64, head group 1, four warps, and two stages. Head
groups 1, 2, and 4 are supported for correctness and exploratory comparison;
changing the group changes the effective Triton row block to
`token_tile * head_group`. This is a small fixed comparison set, not an
autotuning claim. Configured effective row and key tiles must each be at least
16, and tile sizes must be powers of two.

The prepared object retains Q, every segment, output, compiled kernel, and
pointer-cached runner arguments. Input values may change in place, but tensor
shape/address and segment order stay fixed. `.run()` launches on the caller's
current stream and makes no host data readback or new workspace allocation.
Callers order cross-stream writes before running and must not concurrently use
one output object. For T=0, preparation returns an empty output and `.run()`
launches no kernel, regardless of history lengths. The only owned device storage
is the output, `T * 32 * 128 * 2` bytes; there is no KV or softmax scratch buffer.

## Softmax recurrence and four-segment traversal

The online-softmax recurrence is the FlashAttention v2 method shown in the
official Triton 3.3.1 fused-attention tutorial. M9 applies one shared state to
the concatenation's mathematical sequence while reading each physical segment
in order; the method itself is not presented as new. For each Q row, maintain
FP32 `m`, `l`, and unnormalized output accumulator `acc` across all blocks in
all four segments. Scores use `1/sqrt(128)` converted to base-2 scale; the
probability tile is cast to BF16 for the BF16 V dot, which accumulates into
FP32 `acc`. Divide by `l` and write BF16 once at the end.

For a block of scores `s` after scale and `log2(e)` conversion, the stable
update is:

```text
m_next = max(m, max(s over visible keys))
p      = 2 ** (s - m_next), zero for masked keys
alpha  = 2 ** (m - m_next)
l      = alpha * l + sum(p)
acc    = alpha * acc + p @ V
m      = m_next
```

The state is not reinitialized at segment boundaries, and partial segment
outputs are not averaged or summed after separate normalizations. Empty
segments have zero loop iterations. The current segment loop ends after the
query tile's largest position; each current key still receives the per-row
`j <= i` mask. All history blocks are processed for every tile.

## Safe tails and races

Each CTA owns one token tile and one contiguous query-head group. Its output
token/head coordinates do not overlap another CTA, so the forward path needs
no atomics, inter-CTA reduction, or workspace. K/V are read-only. All pointer
offset arithmetic is widened to int64.

Mask Q rows beyond T on both load and store, and mask key lanes beyond each
segment's actual length. For the current segment, intersect those masks with
`key_position <= query_position`. Rows with no visible keys in a block leave
their online state unchanged. This avoids `-inf - -inf` when early current
blocks are fully future-masked, when a history segment is empty, or when Q
lanes are padded. The final divisor is guarded; valid rows with T>0 always see
their own current key, while padded rows are not stored. T=0 is a host-side
no-launch case.

## Preregistered correctness checks

Before any performance claim, run every correctness case for group sizes
`G={1,2,4}` against an independent small FP32 reference. Require finite output,
relative RMS at most `0.008`, and normalized max at most `0.008`; do not raise
the M0-M8 tolerance. Retain actual errors for every case and G value.

| Case | Query and segment lengths `(selected,sink,recent,current)` | Purpose |
| --- | --- | --- |
| Empty history, short | `T=7`, `(0,0,0,7)` | Current-only causal normalization |
| Segment tails | `T=17`, `(5,4,7,17)` | Non-aligned segment and query tails |
| Extreme logits, short | `T=5`, `(2,3,4,5)` | Stable max/exp with large finite Q/K |
| GQA mapping, six rows | `T=6`, `(3,2,5,6)`, zero Q/K and distinct V per KV head | Each contiguous four-Q-head group maps to its KV head |
| Empty query | `T=0`, `(3,4,5,0)` | Empty output, zero launch |
| Query and key tile boundary | `T=65`, `(33,17,41,65)` | Crosses query tiles 32/64 and key tile 64 |
| Empty history, 17 rows | `T=17`, `(0,0,0,17)` | Additional empty-history tail case |
| Short tails, 63 rows | `T=63`, `(1,5,7,63)` | Non-aligned query and segment tails |
| Cross key tile, short history | `T=65`, `(3,17,5,65)` | Segment loop across key tile 64 |
| Causal boundary | `T=9`, `(2,3,5,9)`, position-coded V | Check first/current-last visibility |
| Extreme logits, nine rows | `T=9`, `(2,3,5,9)` | Stable max/exp on another tail layout |
| GQA mapping, five rows | `T=5`, `(2,2,2,5)`, zero Q/K and distinct V per KV head | Check GQA mapping on a second shape |

Keep the FP32 reference small; it may explicitly repeat KV heads for clarity,
but that expansion is correctness-only and never enters a timed candidate path.
Compare each G separately both to the oracle and to the packed FA2 output on the
same inputs. Bitwise equality between Triton and FA2 is not required. Include
every listed functional case and all three G values in the result, including
failure metadata rather than dropping failed cases.

The fixed M8 performance matrix is also the M9 exploratory comparison matrix:
`T={64,256,1024,4096}` crossed with selected-history length
`{0,512,2048}`, with sink length 64, recent length 512, and current length T.
Run all 12 cases for each `head_group` in `{1,2,4}` against the unmodified
reusable-pack plus public FA2 path. Use the independent chunked dense FP32
reference before and after API and captured-graph measurements. Capture graph
replay and fixed-address in-place updates are functional checks; update Q and
all four K/V segments twice on a non-default stream, replay the graph after each
update, then restore and hash-check the original inputs. This G comparison is
exploratory, is not a holdout, and does not authorize changing the preregistered
configuration or making an optimization-selection claim.

## Narrow follow-up hypotheses after M8

The first M9 fixed configuration is only a correctness baseline. M8's measured
pack and FA2 costs must be available before choosing follow-up work.

Grouping several Q heads for one KV head can let a CTA reuse a loaded KV tile,
but it also multiplies the row, score, and FP32 accumulator state when the token
tile stays fixed. With fixed token tile 32, effective BM and state grow with G,
so CTA count falls; for example T=64 gives 64, 32, and 16 CTAs for G=1, 2, and
4. If token tile is instead reduced to hold effective BM fixed, CTA count and
the total logical KV tile loads are approximately unchanged, so grouping alone
does not imply less KV traffic. Any reuse benefit is a hypothesis and depends
on the compiler layout and cache behavior. Compare the same inputs and report
per-configuration registers/spills and latency, without inferring occupancy or
DRAM traffic without counters. The three fixed G values are not a claim of
fourfold traffic reduction.

If long-KV latency is dominated by serial work within too few query CTAs, a
separate split-KV experiment could divide the K/V ranges, write partial FP32
`(m,l,acc)` states, and merge them with the same stable recurrence in a second
kernel. That adds a launch and approximately
`splits * T * 32 * (128 + 2) * 4` bytes of FP32 partial-state storage. At
T=4096 and four splits this is about 260 MiB, so it is not part of this minimal
prototype. It should be considered only after M8 and M9 timings show enough
benefit to justify the workspace and merge cost.

The online recurrence is the established FlashAttention v2 method described in
the [official Triton 3.3.1 fused-attention tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/06-fused-attention.py).
M9's scope is the four direct segment pointers, the current-chunk mask, and a
GQA CTA layout for this fixed shape. The [official FA2 v2.8.3.post1 interface](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post1/flash_attn/flash_attn_interface.py)
is the M8 comparison path and documents GQA and bottom-right causal semantics.
