"""Fixed-launch entry points for the M4 MoE implementations.

This module keeps Triton 3.3.1's compiled launchers and the complete Python
argument tuples ready before the timed call. Tensor pointers are captured once;
the bound tensors and their shape/address metadata must remain unchanged, but
their values (including IDs and routing weights) may be updated in place.
"""
from dataclasses import dataclass

import torch
import triton

from .moe import PreparedMoE, prepare_moe
from .moe_fused import prepare_fused_moe


@dataclass(frozen=True)
class _PreparedLaunch:
    """One CompiledKernel runner with its complete, pointer-cached arguments."""

    name: str
    grid: tuple
    kernel: object
    runner: object
    args: tuple

    def run(self):
        # The Triton runner resolves the current stream when it is called.
        self.runner(*self.args)


@dataclass(frozen=True)
class PreparedMoELauncher:
    """A prepared M4 launch sequence owning its base workspace and kernels.

    Keep all buffers at their prepared shapes and addresses. Input values may
    change in place. The caller must establish stream dependencies and must not
    concurrently write or run this workspace.
    """

    base: PreparedMoE
    fused: bool
    launches: tuple

    def run(self):
        if self.base.x.shape[0] == 0:
            return self.base.output

        # Avoid device-context entry on the common path. On the exceptional
        # path, make the base workspace's device current before getting the
        # stream used by each cached Triton runner.
        if torch.cuda.current_device() == self.base.x.device.index:
            for launch in self.launches:
                launch.run()
        else:
            with torch.cuda.device(self.base.x.device):
                for launch in self.launches:
                    launch.run()
        return self.base.output

    def check_inputs(self):
        """Perform the base workspace's explicit synchronizing validation."""
        return self.base.check_inputs()


def _launch(name, kernel, grid, args):
    """Bind one compiled kernel after checking its full Python signature.

    Triton 3.3.1 keeps constexpr placeholders in ``src.signature`` and expects
    them at the Python launcher even though they are omitted from the device
    kernel parameter list. Therefore validate against and cache all arguments.
    """
    grid = tuple(grid) + (1,) * (3 - len(grid))
    signature = kernel.src.signature
    if len(signature) != len(args):
        raise RuntimeError(
            f"unexpected Triton {name} launcher signature: "
            f"expected {len(args)} Python arguments, found {len(signature)}"
        )
    return _PreparedLaunch(name, grid, kernel, kernel[grid], tuple(args))


def _build_launches(base, fused_kernel=None):
    tokens, hidden = base.x.shape
    experts, _, intermediate = base.down.shape
    topk = base.ids.shape[1]
    routed_rows = tokens * topk
    config = base.config
    be = triton.next_power_of_2(experts)

    if routed_rows == 0:
        return ()

    # Cache every Tensor.data_ptr once. The base object owns all of these
    # tensors for at least as long as this launch sequence is alive.
    counts = base.counts.data_ptr()
    status = base.status.data_ptr()
    ids = base.ids.data_ptr()
    routing_weights = base.routing_weights.data_ptr()
    offsets = base.offsets.data_ptr()
    cursors = base.cursors.data_ptr()
    tasks = base.tasks.data_ptr()
    mapping = base.mapping.data_ptr()
    x = base.x.data_ptr()
    grouped_input = base.grouped_input.data_ptr()
    projected = base.projected.data_ptr()
    activated = base.activated.data_ptr()
    down_output = base.down_output.data_ptr()
    output = base.output.data_ptr()

    launches = [
        _launch("reset", base.kernels["reset"], (1,),
                (counts, status, experts, be)),
        _launch("histogram", base.kernels["histogram"],
                (triton.cdiv(routed_rows, 256),),
                (ids, routing_weights, counts, status, routed_rows, experts, 256)),
        _launch("prefix", base.kernels["prefix"], (1,),
                (counts, offsets, cursors, tasks, experts, config.block_m, be)),
        _launch("positions", base.kernels["positions"],
                (triton.cdiv(routed_rows, 256),),
                (ids, offsets, cursors, mapping, routed_rows, experts, 256)),
        _launch("dispatch", base.kernels["dispatch"],
                (triton.cdiv(routed_rows, 16), triton.cdiv(hidden, 128)),
                (x, mapping, grouped_input, routed_rows, hidden, topk, 16, 128)),
    ]

    if fused_kernel is None:
        gate_up_n = base.gate_up.shape[1]
        gate_up_k = base.gate_up.shape[2]
        gate_up_tasks = triton.cdiv(routed_rows, config.block_m) + experts - 1
        launches.extend((
            _launch("gate_up", base.kernels["gate_up"],
                    (gate_up_tasks, triton.cdiv(gate_up_n, config.block_n)),
                    (grouped_input, base.gate_up.data_ptr(), projected,
                     offsets, counts, tasks, experts, gate_up_n, gate_up_k, be,
                     config.block_m, config.block_n, config.block_k)),
            _launch("swiglu", base.kernels["swiglu"],
                    (triton.cdiv(routed_rows * intermediate, 256),),
                    (projected, activated, routed_rows, intermediate, 256)),
        ))
    else:
        fused_tasks = triton.cdiv(routed_rows, config.block_m) + experts - 1
        launches.append(
            _launch("gate_up_swiglu", fused_kernel,
                    (fused_tasks, triton.cdiv(intermediate, config.block_n)),
                    (grouped_input, base.gate_up.data_ptr(), activated,
                     offsets, counts, tasks, experts, intermediate, hidden, be,
                     config.block_m, config.block_n, config.block_k))
        )

    down_n = base.down.shape[1]
    down_k = base.down.shape[2]
    down_tasks = triton.cdiv(routed_rows, config.block_m) + experts - 1
    launches.extend((
        _launch("down", base.kernels["down"],
                (down_tasks, triton.cdiv(down_n, config.block_n)),
                (activated, base.down.data_ptr(), down_output,
                 offsets, counts, tasks, experts, down_n, down_k, be,
                 config.block_m, config.block_n, config.block_k)),
        _launch("combine", base.kernels["combine"],
                (tokens, triton.cdiv(hidden, 128)),
                (down_output, mapping, routing_weights, output,
                 tokens, hidden, topk, 128)),
    ))
    return tuple(launches)


def prepare_moe_launcher(x, gate_up, down, ids, routing_weights, fused=True):
    """Prepare a cached 8-launch fused or 9-launch baseline M4 entry point.

    Compilation, workspace construction, and the base implementation's
    preparation run all happen here. The returned ``run`` only dispatches the
    fixed cached sequence and does no host counts readback or tensor allocation.
    Triton 3.3.1 is required because the cached CompiledKernel launcher contract
    and its full Python signatures are version-specific.
    """
    if triton.__version__ != "3.3.1":
        raise RuntimeError("prepared MoE launcher is validated only with Triton 3.3.1")

    if fused:
        prepared_fused = prepare_fused_moe(x, gate_up, down, ids, routing_weights)
        base = prepared_fused.base
        if base.x.shape[0] != 0:
            # Populate/compile the fused path and the downstream launch kernels.
            prepared_fused.run()
        fused_kernel = prepared_fused.kernel
    else:
        base = prepare_moe(x, gate_up, down, ids, routing_weights)
        if base.x.shape[0] != 0:
            # Explicitly ensure the complete unfused launch set has been compiled.
            base.run()
        fused_kernel = None

    launches = _build_launches(base, fused_kernel)
    return PreparedMoELauncher(base=base, fused=bool(fused), launches=launches)
