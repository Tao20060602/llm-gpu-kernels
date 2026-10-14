# M34: CUDA BF16 four-segment GQA forward

## Goal and contract

This stage connects the existing SM86 MMA work to the existing attention semantics. It is a new CUDA implementation, not a replacement of the frozen Triton policy or the engine default. Q is `[T,32,128]`; four K/V pairs are `[L,8,128]` in selected/sink/recent/current order. History is fully visible; current uses key position `j <= query_i`. Current length is T, GQA head mapping is `q_head/4`, and the scale is `1/sqrt(128)`. Inputs are already position-processed; no dropout/backward is included.

BF16 Q/K feed FP32-accumulating QK MMA. Softmax maxima, denominators and output state are FP32. Exponentials are rounded to BF16 for PV MMA, and output is BF16, matching the prior attention contract. Different reduction/accumulation order from Triton need not produce bitwise equality; retain the independent FP32 finite/RMS/normalized-max limits of 0.008.

## Mapping

Each CTA handles 64 query rows of one Q head and a 32-key tile, using four warps. Each warp owns 16 query rows. QK has four N8 score fragments, each with four FP32 values per lane; PV has sixteen N8 output fragments. A lane owns two query rows and two columns per fragment. Four-lane XOR shuffles compute each row's max and sum, without an output reduction across warps.

Q/K shared storage uses a BF16 pitch of 128 with XOR row-dependent swizzle. QK uses `ldmatrix.x4` for Q and `ldmatrix.x2` for K, then `mma.sync.m16n8k16`. V is physically key-major with output columns contiguous, so PV B operands use `ldmatrix.x2.trans`. This transposed load is essential: the K operand and V operand have different mathematical roles despite both starting as `[L,8,128]` tensors.

K/V are staged directly from each segment. The final tile is zero-filled and masked. Current tiles beyond the last query in the CTA are skipped. All-masked rows in an intermediate tile retain zero probabilities and safe empty softmax state; active rows ultimately see at least their own current key. Empty T returns without launching.

## Two independently controlled comparisons

| Axis | Control | Candidate | Cost to observe |
|---|---|---|---|
| P handoff | Write BF16 P to shared memory, barrier, `ldmatrix.x4` | Pack adjacent score fragments directly into BF16 A registers for PV | Less shared traffic/barrier, potentially longer register lifetimes |
| K/V staging | Single buffer, issue after consumption | Two buffers, issue next tile before current QK/softmax/PV | Loading can overlap computation, but shared capacity increases |

Register P packing follows the MMA accumulator/A fragment mapping: for each K16 slice, take the low/high query-row pairs from two adjacent N8 score fragments to form four packed A registers. The published [CUDA forward dataflow](https://github.com/Bias92/flashattn-cuda/blob/main/cuda/flash_fwd_kernel.h) was consulted as a mapping reference; this implementation adapts BF16 conversion, fixed four-segment GQA, current masking and token-padded source strides. The algorithm follows [FlashAttention-2](https://arxiv.org/abs/2307.08691); further reference comparisons are recorded in [M34 notes](M34_REFERENCE_NOTES.md).

Both P modes reserve the same P scratch area to isolate the dataflow change from changing declared shared allocation. Declared bytes are `(64*128 + stages*2*32*128 + 64*32)*2`: 36,864 at one stage and 53,248 at two stages. Two stages use shared-memory opt-in during prepare. Q remains in shared memory in this first implementation; register-resident Q is a separate next hypothesis.

## Real-layout support and ownership

Q and K/V may have token padding with contiguous inner strides `(128,1)`, token stride at least `heads*128` and divisible by eight BF16 elements, and a 16-byte-aligned base. Each segment carries its actual K and V token stride; current V stride6144 is read directly rather than copied into a contiguous global intermediate. Output is contiguous and must not share storage with any input.

The prepared wrapper retains references to the input and output owners so their storage stays alive. Callers must keep tensor shape, stride, and storage fixed for the prepared object's lifetime; `run` does not promise runtime metadata or storage revalidation. Values may change in place while those caller preconditions hold. The kernel launches on the current stream and allocates no tensors in `run`. Callers establish cross-stream dependencies and avoid overlapping writes to one output. The wrapper subclasses the existing owner holder because its output is the final owner rather than the GEMM holder's third owner.

## Evaluation scope

Eight functional cases × four configurations cover T0/T1, empty history, uneven lengths, T tails, and padded current V. Benchmark T32/T96/T256 with history `[2048,64,512]`, against the existing Triton implementation including any required layout conversion, and reusable pack + official FA2. Graphs contain four calls per replay; three forward/reverse/forward rounds with three samples per arm. Compile/prepare are outside timing; every arm is checked before and after timing. Preserve all candidates and failures.

The first measurements establish this CUDA baseline and the two controlled changes. They do not imply a new best implementation or engine TTFT gain.

Source: [kernel](../src/llm_gpu_kernels/cuda_gqa_attention.cu), [wrapper](../src/llm_gpu_kernels/attention_cuda.py), [runner](../scripts/run_m34_attention_check.py).

## Observed first result

All 32 numerical cases passed. CUDA best-case Graph p50 is 191.23/317.65/547.33 µs for T32/T96/T256, versus reusable pack+FA2 at 165.82/293.06/467.20 µs. These are synthetic inputs with the production-shaped padded V layout, not captured model activations. FA2 is still faster in all three cases. Direct P helps most arms but S1/shared is marginally fastest at T96; two stages helps T32 but hurts T96/T256. No universal configuration is selected.

Compiled registers for S1/shared, S1/direct, S2/shared and S2/direct are 235/223/237/230, with zero stack/spills. The unexpectedly high register budget motivates a separate rolled segment-loop experiment (M35), rather than adding register-resident Q immediately. The first build failed due to a missing math-constants include; its log is preserved, and the explicit include fixed the build before GPU validation. Timing and errors are in [the report](M34_CUDA_GQA_REPORT.md).

## Profile-only NCU follow-up

A T256 profile-only capture of M34 S1, M34 S2 and M36 v1 S2 is complete (three warmups and one profiled launch per arm). For M34 S2 → M36 S2, worker-declared dynamic shared memory is 53,248 → 49,152 B; the CSV reports total/dynamic/static per-block shared memory as 54.272/53.248 Kbyte and 0 byte versus 50.176/49.152 Kbyte and 0 byte. The NCU shared-memory launch occupancy limit changes from 1 to 2 blocks while the register limit stays at 2 blocks. The captured `sm__warps_active.avg.pct_of_peak_sustained_active` changes from 8.333455% to 14.273184%. This measures the launch-resource limit transition for these specializations; one profiled launch does not establish performance causality. NCU duration is profiler output, not unprofiled benchmark speed. See [the M36 GQA NCU report](M36_GQA_NCU_REPORT.md) and [summary JSON](../results/m36-gqa-ncu-summary.json).
