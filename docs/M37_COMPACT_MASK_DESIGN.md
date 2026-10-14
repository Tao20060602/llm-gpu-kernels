# M37: simplify compact attention and separate complete history tiles

## Before the change: minimal design gate

1. Structural: compact attention has only register-direct P, so its P mode and shared-P implementation are unreachable; remove that retired path end to end rather than validating a constant at each layer.
2. Burden: remove the caller's `p_mode=1` argument and add no new runtime choice for the full-tile optimisation.
3. Determinable: complete query/key tile status is derived from shape, segment length and tile position inside the kernel.
4. Gratuitous: modify the existing compact implementation, reusing current ownership, loader and verification entry points instead of adding another CUDA/wrapper clone.
5. Overfitted: apply the fast path to every complete history tile; retain the existing masks for any partial query/key tile and for current causal tiles, without special-casing measured lengths.

The smallest change removes the unused P structure/parameter and uses a uniform branch to scale a complete history score tile without per-element boundary checks. Parameter removal alone cannot eliminate those repeated checks, while adding a policy/configuration for choosing the path would burden callers with a fact the kernel already knows. Historical M34/M35 experiments and results are retained because the project explicitly preserves failed and controlled comparisons; they are not compatibility fallbacks of the compact API.

## Structural defect and rule

The defect class is a retired algorithm option still represented in a live caller interface, native state and unreachable workspace logic. A fixed behaviour has one implementation, and callers should not supply a constant to select a path the implementation cannot support.

Remove compact `p_mode`, the native member/constructor argument, `DirectP` template choice, P shared pointer/size and shared-P read/write branches. Update every current compact call site in the same change. M34/M35 retain their genuine two-mode experiments. No alias or compatibility reader is added. Record the exact search and dispositions in the delivery note.

## Performance hypothesis

FA2's complete-tile organisation motivates distinguishing full historical tiles from masked boundaries. On complete historical tiles, every query and key is valid; only score scaling is needed before row max. Derive `segment<3 && start_q+64<=T && tile_start+32<=length` once per tile. On all other tiles, use the existing query tail, key tail and current causal masks. K/V staging and zero fill remain unchanged. Outputs should match the previous compact kernel bitwise because score scaling and valid-element arithmetic are unchanged.

Preserve the existing v1 compiled compact binary locally as a measured control, and build the new ABI as v2. The product loader selects only v2. A benchmark's explicit old-binary control is a historical measurement, not a runtime product fallback. Compare both stages to matching v1 controls and raw-layout reusable pack+FA2 in one batch; compilation/preparation excluded.

## Behaviour-first validation

Use the public `prepare_cuda_gqa_compact` entry on the GPU. Before implementing, list failures: T0/empty history; complete tiles; partial query/key tiles; zero-length middle segment; current causal boundary; padded V; changed values on fixed buffers; current-stream dispatch; fixed-address Graph replay. Reuse existing numerical cases plus a value-update/Graph scenario through this entry; add no source-text/hash/recipe assertions. Independent FP32 attention determines the expected result; hashes remain optional evidence bookkeeping, not a behavioural gate. Save a normalised manifest with actual errors, shape/stride and checks, leaving timestamps/source fingerprints outside assertions.

LLM-agent-surface applies to the work briefs and evidence handoff: raw values/units are measured, the primary agent chooses the next hypothesis, and no language heuristics or agent-framework settings are added to this operator package.

## Delivery review scope

The primary agent has read the full compact CUDA source and Python wrapper before design. Record all subsequently reviewed changed files and net lines by kernel, wrapper/call sites, verification and documentation in the delivery report; do not extend that review verdict to unrelated historical kernels or repository tests.
