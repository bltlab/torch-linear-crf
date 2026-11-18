"""CUDA graphs around a `LinearChainCRF`'s likelihoods and decodes.

On CUDA the CRF's recursions are launch-bound: each sequence position costs
about 15 kernel launches forward and 32 with the backward, whatever the batch
or tag count. `GraphedCRF` captures each call's kernels once per input
signature and replays them as one graph launch, so the GPU runs the same
kernels on the same data as the CRF's own method, and the results are
bitwise equal to it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import torch

from torch_linear_crf.crf import LinearChainCRF, _bucket_length, _check_bucketing

# Eager calls before each capture, as torch.cuda.make_graphed_callables does,
# so lazy CUDA initialization (handles, workspaces) stays out of the graph.
_WARMUP_CALLS = 3

_Key = tuple[Any, ...]


class _Graphs:
    """The graphs and static tensors captured for one input signature."""

    __slots__ = ("backward", "forward", "grad_output", "grads", "inputs", "outputs", "written")

    def __init__(self, inputs: tuple[torch.Tensor, ...]) -> None:
        # The user's inputs, padded with zeros (False) to the bucket length.
        # Allocated outside the graph pool, so no replay ever writes them.
        self.inputs = inputs
        # Beyond this prefix length every static input is zero.
        self.written = 0
        self.forward = torch.cuda.CUDAGraph()
        self.backward: torch.cuda.CUDAGraph | None = None
        self.outputs: tuple[torch.Tensor, ...] = ()
        self.grad_output: torch.Tensor | None = None
        # Gradients of the static inputs that require grad, then of the parameters.
        self.grads: tuple[torch.Tensor | None, ...] = ()

    def load(self, tensors: tuple[torch.Tensor, ...], length: int) -> None:
        """Copy `tensors` into the static inputs and zero what the last call left past `length`."""
        with torch.no_grad():
            for static, x in zip(self.inputs, tensors, strict=True):
                static[:, :length].copy_(x)
                if self.written > length:
                    static[:, length : self.written].zero_()
        self.written = length


class _Replay(torch.autograd.Function):
    """Autograd node for a graphed likelihood: forward and backward are graph replays."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, helper: GraphedCRF, graphs: _Graphs, length: int, count: int, *args: torch.Tensor
    ) -> torch.Tensor:
        helper._replay(graphs, args[:count], length)
        ctx.helper = helper
        ctx.graphs = graphs
        ctx.length = length
        ctx.replay = helper._replays
        return graphs.outputs[0].clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        helper: GraphedCRF = ctx.helper
        graphs: _Graphs = ctx.graphs
        if helper._replays != ctx.replay:
            raise RuntimeError(
                "GraphedCRF: another graphed call ran between this likelihood and its "
                "backward and may have overwritten the memory the backward reads. Call "
                "backward on each graphed likelihood before the next graphed call of any "
                "method, or use the CRF's own method for the call in between."
            )
        assert graphs.grad_output is not None and graphs.backward is not None
        graphs.grad_output.copy_(grad_output)
        graphs.backward.replay()
        helper._replays += 1
        grads = iter(graphs.grads)
        result: list[torch.Tensor | None] = [None, None, None, None]
        for static in graphs.inputs:
            grad = next(grads) if static.requires_grad else None
            # A clone, so no caller ever holds memory the next replay overwrites.
            result.append(None if grad is None else grad[:, : ctx.length].clone())
        result.extend(None if grad is None else grad.clone() for grad in grads)
        return tuple(result)


class GraphedCRF:
    """Runs a `LinearChainCRF`'s likelihoods and decodes as CUDA graph replays.

        graphed = GraphedCRF(model.crf)
        loss = -graphed.log_likelihood(emissions, tags, mask).sum()
        loss.backward()  # gradients reach emissions and model.crf's parameters

    Covers `log_likelihood` and `partial_log_likelihood` (forward and
    backward), `decode_to_tensor` and `decode`, with the CRF's own signatures.
    Values and gradients are bitwise those of the CRF's method on the same
    inputs padded by `pad_to_length_bucket(..., min_length=min_length,
    multiple=multiple)`: so bitwise those of the unpadded call for every
    gradient, `partial_log_likelihood` and the decodes, and for
    `log_likelihood`'s value to rounding (at most 1.6 units in the last place
    measured; see `pad_to_length_bucket`). With `parallel_scan`, the
    likelihoods match the padded call bitwise and the unpadded one to
    rounding. `min_length=1, multiple=1` captures one graph per exact length,
    bitwise equal to the unpadded call throughout.

    One graph (forward and, when gradients are needed, backward) is captured
    per signature: method, batch size, bucket length, dtypes, which inputs
    and parameters require grad, and grad mode. The batch is never padded,
    since dummy rows would change the order of the parameter gradients'
    batch sums, so a smaller last batch gets its own graphs. The first call
    of each signature runs the method 3 times eagerly, then captures, which
    synchronizes the device; every later call copies the inputs into static
    buffers, replays, and clones the outputs, with no host sync.

    Every graph of one `GraphedCRF` shares one memory pool, so each graph's
    temporaries reuse the same memory. That makes one rule necessary, which
    the backward checks: between a graphed likelihood and its backward, no
    other graphed call of this object may run (RuntimeError otherwise). A
    training step's forward, backward, forward, backward order satisfies it;
    calling the likelihood twice before one backward, or decoding in between,
    does not.

    Memory: the pool holds about the largest signature's working set, close
    to one eager call's peak (`log_likelihood` with backward at B128 T256
    C100: 1356 MB reserved against an eager peak of 1323 MB), and each
    further signature adds its static inputs, outputs and gradients (about
    8 * B * T * C bytes for the emissions and their gradient) plus what its
    working set needs beyond the pool's free blocks (B32 C37: 68 MB for
    length 256 alone, 70 MB with lengths 128 down to 16 added after it, 72
    MB for 16 up to 256). Each capture also synchronizes the device and
    empties the CUDA caching allocator's cache, as `torch.cuda.graph` does.

    Off CUDA, and for inputs the CRF would reject on shape, each method calls
    the CRF's own method directly, so the same code runs on CPU and raises
    the CRF's own errors. The CRF must use `validate="async"` or None on
    CUDA: "sync" reads each check back to the host, which a graph cannot
    capture (ValueError). Under autocast nothing is needed: the CRF disables
    autocast around its recursions and casts explicitly, so the capture runs
    with autocast off and records the same kernels as an eager call under
    autocast. Changing the CRF's `constrain_loss` or `parallel_scan`, or
    moving or recasting its parameters or buffers (`.to`, `.double()`),
    drops every graph; in-place updates such as an optimizer step are read
    by the next replay.
    """

    def __init__(
        self, crf: LinearChainCRF, min_length: int = 16, multiple: int | None = None
    ) -> None:
        _check_bucketing(min_length, multiple)
        self.crf = crf
        self.min_length = min_length
        self.multiple = multiple
        self._graphs: dict[_Key, _Graphs] = {}
        self._state: _Key | None = None
        self._pool: Any = None
        # One capture stream per device: the caching allocator reuses a freed
        # block only on the stream it was allocated on, so captures on one
        # stream are what lets the graphs share the pool's memory.
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        # Counts replays, so a backward can tell whether another replay ran
        # since its forward.
        self._replays = 0

    @property
    def num_graphs(self) -> int:
        """How many input signatures have been captured."""
        return len(self._graphs)

    def reset(self) -> None:
        """Drop every captured graph and the memory pool."""
        self._graphs.clear()
        self._pool = None
        self._streams.clear()
        self._state = None
        self._replays += 1

    def log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`LinearChainCRF.log_likelihood`, replayed from a CUDA graph."""
        tensors = (emissions, tags, mask)
        if not self._graphable(tensors):
            return self.crf.log_likelihood(emissions, tags, mask)
        return self._likelihood("log_likelihood", tensors, self.crf.log_likelihood)

    def partial_log_likelihood(
        self,
        emissions: torch.Tensor,
        allowed: torch.Tensor,
        mask: torch.Tensor,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`LinearChainCRF.partial_log_likelihood`, replayed from a CUDA graph."""
        tensors = (emissions, allowed, mask)
        if log_weights is not None:
            tensors += (log_weights,)
        if not self._graphable(tensors):
            return self.crf.partial_log_likelihood(emissions, allowed, mask, log_weights)
        crf = self.crf

        def call(*s: torch.Tensor) -> torch.Tensor:
            return crf.partial_log_likelihood(s[0], s[1], s[2], s[3] if len(s) > 3 else None)

        return self._likelihood("partial_log_likelihood", tensors, call)

    def decode_to_tensor(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allowed: torch.Tensor | None = None,
        pad_tag: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`LinearChainCRF.decode_to_tensor`, replayed from a CUDA graph."""
        tensors = (emissions, mask) if allowed is None else (emissions, mask, allowed)
        if not self._graphable(tensors):
            return self.crf.decode_to_tensor(emissions, mask, allowed, pad_tag)
        return self._decode(tensors, pad_tag)

    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None = None
    ) -> list[list[int]]:
        """`LinearChainCRF.decode`: graphed `decode_to_tensor`, then a host copy."""
        tensors = (emissions, mask) if allowed is None else (emissions, mask, allowed)
        if not self._graphable(tensors):
            return self.crf.decode(emissions, mask, allowed)
        paths, _ = self._decode(tensors, -1)
        rows: list[list[int]] = paths.tolist()
        lengths: list[int] = mask.to(torch.bool).sum(1).tolist()
        return [row[:n] for row, n in zip(rows, lengths, strict=True)]

    @staticmethod
    def _graphable(tensors: tuple[torch.Tensor, ...]) -> bool:
        """Whether the inputs are CUDA tensors of a shape the CRF accepts, so far as graphs need."""
        emissions = tensors[0]
        if not emissions.is_cuda or emissions.dim() != 3:
            return False
        batch, length = emissions.size(0), emissions.size(1)
        return (
            batch > 0
            and length > 0
            and all(
                x.device == emissions.device
                and x.dim() >= 2
                and x.size(0) == batch
                and x.size(1) == length
                for x in tensors
            )
        )

    def _likelihood(
        self, name: str, tensors: tuple[torch.Tensor, ...], call: Callable[..., torch.Tensor]
    ) -> torch.Tensor:
        params = tuple(p for p in self.crf.parameters(recurse=False) if p.requires_grad)
        needs_grad = torch.is_grad_enabled() and (
            bool(params) or any(x.requires_grad for x in tensors)
        )
        graphs, length = self._lookup(name, tensors, call, needs_grad)
        if not needs_grad:
            self._replay(graphs, tensors, length)
            return graphs.outputs[0].clone()
        out = _Replay.apply(self, graphs, length, len(tensors), *tensors, *params)
        assert isinstance(out, torch.Tensor)
        return out

    def _decode(
        self, tensors: tuple[torch.Tensor, ...], pad_tag: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        crf = self.crf

        def call(*s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return crf.decode_to_tensor(s[0], s[1], s[2] if len(s) > 2 else None, pad_tag)

        graphs, length = self._lookup(("decode_to_tensor", pad_tag), tensors, call, False)
        self._replay(graphs, tensors, length)
        paths, scores = graphs.outputs
        return paths[:, :length].clone(), scores.clone()

    def _replay(self, graphs: _Graphs, tensors: tuple[torch.Tensor, ...], length: int) -> None:
        graphs.load(tensors, length)
        graphs.forward.replay()
        self._replays += 1

    def _check_state(self) -> None:
        """Drop every graph if the CRF changed in a way a replay would not see."""
        crf = self.crf
        if crf._check_values and not crf._async_checks:
            raise ValueError(
                'GraphedCRF needs the CRF\'s validate to be "async" or None on CUDA; "sync" '
                "reads each check back to the host, which a CUDA graph cannot capture"
            )
        state = (
            crf.constrain_loss,
            crf.parallel_scan,
            crf._check_values,
            tuple(
                (x.data_ptr(), x.dtype, x.device)
                for x in itertools.chain(crf.parameters(), crf.buffers())
            ),
        )
        if state != self._state:
            if self._state is not None:
                self.reset()
            self._state = state

    def _lookup(
        self,
        name: Any,
        tensors: tuple[torch.Tensor, ...],
        call: Callable[..., Any],
        needs_grad: bool,
    ) -> tuple[_Graphs, int]:
        """The graphs for these inputs' signature, captured now if new, and the input length."""
        self._check_state()
        emissions = tensors[0]
        length = emissions.size(1)
        bucket = _bucket_length(length, self.min_length, self.multiple)
        key = (
            name,
            bucket,
            needs_grad,
            emissions.device,
            tuple(p.requires_grad for p in self.crf.parameters(recurse=False))
            if needs_grad
            else (),
            tuple(
                (x.dtype, x.requires_grad and needs_grad, x.shape[:1] + x.shape[2:])
                for x in tensors
            ),
        )
        graphs = self._graphs.get(key)
        if graphs is None:
            graphs = self._capture(tensors, bucket, call, needs_grad)
            self._graphs[key] = graphs
        return graphs, length

    def _capture(
        self,
        tensors: tuple[torch.Tensor, ...],
        bucket: int,
        call: Callable[..., Any],
        needs_grad: bool,
    ) -> _Graphs:
        device = tensors[0].device
        statics = []
        for x in tensors:
            shape = list(x.shape)
            shape[1] = bucket
            statics.append(torch.zeros(shape, dtype=x.dtype, device=device))
        graphs = _Graphs(tuple(statics))
        graphs.load(tensors, tensors[0].size(1))
        for x, static in zip(tensors, statics, strict=True):
            if needs_grad and x.requires_grad:
                static.requires_grad_(True)
        # The capture differentiates with respect to fresh leaves sharing the
        # parameters' storage, so a replay reads every in-place update of the
        # parameters, and the autograd nodes of the caller's own graphs (which
        # may still hold the parameters' gradient accumulators, created on
        # another stream) take no part in the capture.
        crf = self.crf
        aliases = {
            name: p.detach().requires_grad_(True)
            for name, p in crf.named_parameters(recurse=False)
            if needs_grad and p.requires_grad
        }
        targets = tuple(s for s in statics if s.requires_grad) + tuple(aliases.values())
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()

        def run() -> tuple[torch.Tensor, ...]:
            originals = {name: crf._parameters[name] for name in aliases}
            crf._parameters.update(aliases)  # type: ignore[arg-type]
            try:
                out = call(*statics)
            finally:
                crf._parameters.update(originals)
            return out if isinstance(out, tuple) else (out,)

        # Autocast off: the CRF turns it off around its recursions itself, and
        # nothing outside them is on an autocast list, so the kernels are those
        # of an eager call under autocast, without autocast's weight cache.
        with (
            torch.cuda.device(device),
            torch.autocast("cuda", enabled=False),
            torch.set_grad_enabled(needs_grad),
        ):
            stream = self._streams.get(device)
            if stream is None:
                stream = self._streams[device] = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(_WARMUP_CALLS):
                    out = run()
                    if needs_grad:
                        torch.autograd.grad(
                            out[0], targets, torch.ones_like(out[0]), allow_unused=True
                        )
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graphs.forward, pool=self._pool, stream=stream):
                out = run()
            if needs_grad:
                # Outside the pool, like the static inputs.
                graphs.grad_output = torch.empty_like(out[0])
                graphs.backward = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs.backward, pool=self._pool, stream=stream):
                    graphs.grads = torch.autograd.grad(
                        out[0], targets, graphs.grad_output, allow_unused=True
                    )
        graphs.outputs = tuple(o.detach() for o in out)
        return graphs
