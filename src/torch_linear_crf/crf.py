"""Linear-chain CRF layer with optional hard transition constraints.

The algorithm is the textbook linear-chain CRF (Lafferty et al. 2001): forward
algorithm for the partition function, Viterbi for decoding, both fully
vectorized over the batch dimension (the only Python loop is over the
sequence-length dimension, since each step depends on the previous one).

The module shape -- separate start/end/transition parameters, batch-first
API, and the `log_likelihood` / `decode` split -- follows pytorch-crf
(https://github.com/kmkurn/pytorch-crf, MIT). Behavior to know about:

* **bf16 autocast.** Every method casts emissions and parameters to fp32 (or
  higher: an fp64 model stays fp64) before the recursion and runs it inside
  `torch.autocast(enabled=False)`. Running the recursion in bf16 was
  measured 6.9 nats off full fp32 at realistic logit scale, against 0.36 for
  bf16 emissions fed into an fp32 recursion. The explicit cast is what keeps
  the numbers right; no op in the recursion is on an autocast cast list
  today, so the guard covers one joining later and a caller's own nested
  autocast region.
* **Hard transition constraints** use finite penalties (`FORBIDDEN`) rather
  than `-inf`, so a fully-constrained row cannot produce nan from
  `(-inf) - (-inf)`. `torch_linear_crf.constraints` derives the masks from a
  tag encoding; this module only applies whatever mask it is given.
* **-inf emissions are safe to train on.** The likelihoods clamp emissions
  from below at `FORBIDDEN`, so masking a tag out with -inf gives finite
  gradients instead of nan; see `LinearChainCRF` for exactly what changes.
* **`include_start_end_transitions=False`** makes the start/end parameters
  fixed zero buffers rather than removing them, so every downstream
  computation stays branch-free.

License
-------
MIT (see LICENSE). No code copied verbatim from pytorch-crf or AllenNLP.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

# Additive penalty for a forbidden transition. Finite rather than -inf so that a
# fully-constrained row can never produce nan from (-inf) - (-inf); large enough
# that a forbidden path is never competitive at realistic logit scale.
FORBIDDEN = -1e4


def _autocast_device_type(device: torch.device) -> str:
    """The device type to disable autocast for, or "cpu" where there is none.

    Eager and `torch.compile` can name any device autocast supports (cuda,
    xpu, mps, hpu, ...), so the guard holds on every accelerator, not just
    CUDA. TorchScript cannot: it requires a constant device_type ("Autocast
    device argument must be a constant"), and its autocast pass recognizes
    only "cuda" and "cpu" -- a branch naming any other device fails an
    internal assert as soon as the scripted module runs. Hence the
    `torch.jit.is_scripting()` split at both call sites; a scripted module on
    XPU/MPS/HPU falls back to the cpu branch, which leaves that device's
    autocast alone and relies on this module's explicit fp32 casting instead.
    """
    device_type = device.type
    if not torch.amp.is_autocast_available(device_type):
        # e.g. meta tensors: no autocast state exists to disable.
        return "cpu"
    return device_type


def _floor_emissions(emissions: torch.Tensor) -> torch.Tensor:
    """Raise every emission below FORBIDDEN to FORBIDDEN, for the likelihoods.

    A tag whose emission is -inf at a valid position makes that forward
    entry a logsumexp over all -inf, whose backward is nan for every
    parameter. Clamped, it costs the same finite FORBIDDEN as a forbidden
    transition. Emissions at or above FORBIDDEN pass through unchanged,
    value and gradient; a clamped one gets zero gradient.
    """
    # -1e4 is FORBIDDEN, written out: TorchScript cannot read a module global.
    # masked_fill on strictly-below, not clamp(min=...): clamp's backward
    # is zero at the boundary itself, which would change the gradient of
    # an emission a caller already set to FORBIDDEN.
    return emissions.masked_fill(emissions < -1e4, -1e4)


def _check_value(ok: torch.Tensor, message: str, async_checks: bool) -> None:
    """Raise ValueError(message) unless the 0-d bool tensor `ok` is true.

    Reading `ok` back is a host sync. With `async_checks` and a CUDA tensor,
    `torch._assert_async` checks it on the device instead, so the caller
    never waits on the GPU; a failure is then a device-side assert, which is
    fatal to the CUDA context.
    """
    if async_checks and ok.is_cuda:
        torch._assert_async(ok, message)
    elif not bool(ok):
        raise ValueError(message)


def forward_backward(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    transitions: torch.Tensor,
    end: torch.Tensor,
    temperature: float = 1.0,
    validate: str | None = "async",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-backward over any linear chain: (alpha, beta, log_z).

    For a chain that is not a `LinearChainCRF`'s own, such as per-token
    log-softmax emissions with constraint-only transitions; for a CRF's own
    potentials, use `LinearChainCRF.forward_backward`. `emissions` `(B, T, C)`
    and `mask` `(B, T)` follow `LinearChainCRF`'s layout (a contiguous valid
    prefix per row, at least one valid position); `start`, `transitions`,
    `end` are `(C,)`, `(C, C)`, `(C,)`, with `transitions[i, j]` the score of
    tag i followed by tag j.

    `alpha[b, t, c]` is the log total score of every prefix over positions
    0..t ending in tag c, emission at t included; `beta[b, t, c]` is the log
    total score of every continuation after t given tag c at t, end score
    included and emission at t excluded; `log_z` `(B,)` is the log partition.
    `alpha + beta - log_z.view(-1, 1, 1)` is the log posterior marginal of
    each tag at each position. Values at padding positions are finite and
    meaningless.

    `temperature` divides every score, potentials included, giving the
    distribution proportional to p(y) ** (1 / temperature). The recursion
    runs in fp32, or fp64 for fp64 emissions, with autocast off, and is
    differentiable in the emissions and all three potentials. Emissions are
    clamped from below at `FORBIDDEN` after the temperature, as in
    `LinearChainCRF.log_likelihood`. `validate` is as in `LinearChainCRF`;
    shape checks always run.
    """
    if emissions.dim() != 3:
        raise ValueError(
            "emissions must have 3 dimensions (batch, length, num_tags), got "
            + str(emissions.dim())
        )
    num_tags = emissions.size(2)
    if mask.dim() != 2 or mask.size(0) != emissions.size(0) or mask.size(1) != emissions.size(1):
        raise ValueError("mask must have shape (batch, length) matching emissions")
    if (
        start.dim() != 1
        or start.size(0) != num_tags
        or end.dim() != 1
        or end.size(0) != num_tags
        or transitions.dim() != 2
        or transitions.size(0) != num_tags
        or transitions.size(1) != num_tags
    ):
        raise ValueError(
            "start, transitions and end must have shapes (C,), (C, C) and (C,) for "
            "C = emissions.size(2) = " + str(num_tags)
        )
    if not temperature > 0.0:
        raise ValueError("temperature must be > 0, got " + str(temperature))
    if validate is not None:
        if validate != "async" and validate != "sync":
            raise ValueError('validate must be "async", "sync" or None, got ' + validate)
        _check_value(
            mask[:, 0].to(torch.bool).all(),
            "every row must have at least one valid position, with the valid "
            "prefix starting at position 0",
            validate == "async",
        )
    return _guarded_forward_backward(emissions, mask, start, transitions, end, temperature, False)


def _guarded_forward_backward(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    transitions: torch.Tensor,
    end: torch.Tensor,
    temperature: float,
    clamped: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.jit.is_scripting():
        # See _autocast_device_type: only the scripted path is restricted
        # to cuda/cpu.
        if emissions.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return _forward_backward_impl(
                    emissions, mask, start, transitions, end, temperature, clamped
                )
        else:
            with torch.autocast(device_type="cpu", enabled=False):
                return _forward_backward_impl(
                    emissions, mask, start, transitions, end, temperature, clamped
                )
    else:
        with torch.autocast(device_type=_autocast_device_type(emissions.device), enabled=False):
            return _forward_backward_impl(
                emissions, mask, start, transitions, end, temperature, clamped
            )


def _forward_backward_impl(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    transitions: torch.Tensor,
    end: torch.Tensor,
    temperature: float,
    clamped: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.promote_types(emissions.dtype, torch.float32)
    emissions = _floor_emissions(emissions.to(dtype) / temperature)
    start = start.to(dtype) / temperature
    transitions = transitions.to(dtype) / temperature
    end = end.to(dtype) / temperature
    mask = mask.to(torch.bool)
    batch = emissions.size(0)
    # A padding position's emission may be non-finite and is never part of a
    # path; zero it so neither recursion's backward sees it (see _log_partition).
    emissions = torch.where(mask.unsqueeze(2), emissions, torch.zeros_like(emissions))
    # unbind rather than per-step indexing; see LinearChainCRF._log_partition.
    steps = emissions.unbind(1)
    valid_steps = mask.unbind(1)

    alphas: list[torch.Tensor] = [start.unsqueeze(0) + steps[0]]
    for t in range(1, len(steps)):
        step = _logsumexp(alphas[-1].unsqueeze(2) + transitions.unsqueeze(0), 1, clamped)
        # Past a row's length, carry alpha forward unchanged.
        alphas.append(torch.where(valid_steps[t].unsqueeze(1), step + steps[t], alphas[-1]))

    # beta at a row's last valid position is the end score. Walking backwards,
    # a position whose successor is padding is a last position (or padding),
    # so it takes the end score too; every other position recurses.
    final = end.unsqueeze(0).expand(batch, -1)
    betas: list[torch.Tensor] = [final]
    for t in range(len(steps) - 2, -1, -1):
        following = (steps[t + 1] + betas[-1]).unsqueeze(1)  # (B, 1, C_next)
        step = _logsumexp(transitions.unsqueeze(0) + following, 2, clamped)
        betas.append(torch.where(valid_steps[t + 1].unsqueeze(1), step, final))
    betas.reverse()

    log_z = _logsumexp(alphas[-1] + end.unsqueeze(0), 1, clamped)
    return torch.stack(alphas, dim=1), torch.stack(betas, dim=1), log_z


def pad_to_length_bucket(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    *others: torch.Tensor,
    min_length: int = 16,
    multiple: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Pad the length dimension (dim 1) of every tensor up to a bucket length.

    For `torch.compile`, which builds one graph per distinct length (the
    recursion's loop over positions is unrolled) and, after
    `torch._dynamo.config.recompile_limit` lengths (8 by default), runs any
    further length eagerly. Padding lengths to a few buckets keeps the number
    of graphs small. Call it eagerly, before the compiled function.

    The bucket is the next power of two that is at least `min_length` (16,
    32, 64, ... by default), or, with `multiple`, the next multiple of it.
    Powers of two give fewest graphs but up to twice the steps; under the
    default loop each padded step costs as much as a real one, under
    `parallel_scan` almost nothing.

    Returns `(emissions, mask, *others)` in order, padded with zeros (False for
    bool tensors such as `mask` and `allowed`; tag 0 for `tags`), all on their
    own devices with no host sync. Padded positions fall outside `mask`, so
    results match the unpadded input's within each row's valid prefix
    (`marginals`, `decode_to_tensor` and the like return the padded length;
    slice it off): bitwise for every gradient, `partial_log_likelihood`, the
    decodes and the marginals; `log_likelihood`'s value to rounding (measured
    at most 1.6 units in the last place), since its gold-path score sums over
    the padded length. Under `parallel_scan` the tree pairs steps differently,
    so the likelihoods agree to rounding. Raises ValueError if `min_length` or
    `multiple` is below 1 or the tensors disagree on dim 1.
    """
    _check_bucketing(min_length, multiple)
    tensors = (emissions, mask, *others)
    length = emissions.size(1)
    for x in tensors:
        if x.dim() < 2 or x.size(1) != length:
            raise ValueError(
                f"every tensor needs dim 1 equal to emissions' length {length}, got shape "
                f"{tuple(x.shape)}"
            )
    target = _bucket_length(length, min_length, multiple)
    if target == length:
        return tensors
    padded = []
    for x in tensors:
        shape = list(x.shape)
        shape[1] = target - length
        padded.append(torch.cat([x, x.new_zeros(shape)], dim=1))
    return tuple(padded)


def _check_bucketing(min_length: int, multiple: int | None) -> None:
    """Raise ValueError unless `pad_to_length_bucket`'s bucket arguments are valid."""
    if min_length < 1:
        raise ValueError(f"min_length must be at least 1, got {min_length}")
    if multiple is not None and multiple < 1:
        raise ValueError(f"multiple must be at least 1, got {multiple}")


def _bucket_length(length: int, min_length: int, multiple: int | None) -> int:
    """The length `pad_to_length_bucket` pads `length` to."""
    if multiple is None:
        return max(min_length, 1 << max(length - 1, 0).bit_length())
    return max(min_length, -(-length // multiple) * multiple)


def _clamped_exp(y: torch.Tensor, exact: bool) -> torch.Tensor:
    """exp(y), computed with every input below -87 kept off exp's slow path.

    PyTorch's vectorized CPU exp leaves its fast path below about -87.3,
    where float32 exp underflows, and takes 16-35x longer there (measured on a
    Threadripper 3970X). Inputs below -87 are clamped and their result set to 0, which
    is exp's own value below -103.98. Between that and -87 exp is at most
    1.65e-38 (a denormal below -87.34): with `exact`, those entries are
    recomputed with exp, when there are any, so the result is bitwise exp(y);
    otherwise they are 0.
    """
    low = y < -87.0
    out = y.clamp(min=-87.0).exp_().masked_fill_(low, 0.0)
    if exact:
        band = low & (y > -110.0)
        if bool(band.any()):
            out[band] = y[band].exp()
    return out


class _ClampedLogSumExp(torch.autograd.Function):
    """torch.logsumexp with `_clamped_exp` in the forward and the backward.

    The same operations as torch.logsumexp and its backward, so values and
    gradients match it bitwise in every test. The forward drops the terms
    `exact` would restore: each row's sum includes exp(0) = 1 from its
    maximum, against which a term under 1.65e-38 is far below rounding. The
    backward uses each term on its own, so it keeps them exact. A backward
    with `create_graph` runs torch.logsumexp's own backward, which is
    differentiable, and `jvp` is torch.logsumexp's forward-mode formula with
    the exact clamped exp.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:  # type: ignore[override]
        maxes = x.amax(dim, keepdim=True)
        maxes = maxes.masked_fill(maxes.abs() == float("inf"), 0.0)
        result = _clamped_exp(x - maxes, False).sum(dim).log_().add_(maxes.squeeze(dim))
        ctx.save_for_backward(x, result)
        ctx.save_for_forward(x)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x, result = ctx.saved_tensors
        shifted = x - result.unsqueeze(ctx.dim)
        # Grad mode is on in a backward only under create_graph.
        terms = shifted.exp() if torch.is_grad_enabled() else _clamped_exp(shifted, True)
        return grad.unsqueeze(ctx.dim) * terms, None

    @staticmethod
    def jvp(ctx, x_t: torch.Tensor, _dim_t: None) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        terms = _clamped_exp(x - x.amax(ctx.dim, keepdim=True), True)
        return (terms * x_t).sum(ctx.dim) / terms.sum(ctx.dim)


@torch.jit.unused
def _clamped_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return _ClampedLogSumExp.apply(x, dim)


@torch.jit.unused
def _is_plain_eager() -> bool:
    """Whether nothing is tracing or transforming the call.

    In order: torch.compile and torch.export, torch.jit.trace, torch.func
    transforms, and dispatch modes, such as make_fx's (pre-dispatch too),
    FakeTensorMode and FlopCounterMode.
    """
    return (
        not torch.compiler.is_compiling()
        and not torch.jit.is_tracing()
        and torch._C._functorch.peek_interpreter_stack() is None  # type: ignore[attr-defined]
        and torch._C._len_torch_dispatch_stack() == 0
        and torch._ops._len_torch_dispatch_stack_pre_dispatch() == 0
    )


def _logsumexp(x: torch.Tensor, dim: int, clamped: bool) -> torch.Tensor:
    """torch.logsumexp, or `_ClampedLogSumExp` when `clamped` and `x` is float32."""
    if clamped and not torch.jit.is_scripting() and x.dtype == torch.float32:
        return _clamped_logsumexp(x, dim)
    return torch.logsumexp(x, dim=dim)


def _log_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Log-space matrix product, exactly: out[..., i, j] = logsumexp_k a[..., i, k] + b[..., k, j].

    Through a broadcast (..., C, C, C) tensor rather than exp, matmul and
    log, which would underflow to log(0) for entries far below the row's
    best (a forbidden transition, a FORBIDDEN-masked tag).
    """
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def _log_partition_scan(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    transitions: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """log Z by a balanced tree of `_log_matmul`s over the per-step (C, C) score matrices.

    `emissions` must already be finite at padding positions. Each step t >= 1
    contributes transitions + emissions[t] as a matrix; a padding step, and
    the filler that evens out a level, is the log-space identity, with a
    large finite value off the diagonal rather than -inf so that no
    logsumexp ever sees an all -inf row (whose backward is nan).
    """
    batch = emissions.size(0)
    num_tags = emissions.size(2)
    # Off-diagonal "zero": exp of it is exactly 0, and twice it is still finite.
    identity = torch.full(
        (num_tags, num_tags), -1e30, dtype=emissions.dtype, device=emissions.device
    )
    identity.fill_diagonal_(0.0)
    mats = transitions.view(1, 1, num_tags, num_tags) + emissions[:, 1:].unsqueeze(2)
    mats = torch.where(mask[:, 1:].view(batch, -1, 1, 1), mats, identity)
    while mats.size(1) > 1:
        if mats.size(1) % 2 == 1:
            mats = torch.cat([mats, identity.expand(batch, 1, num_tags, num_tags)], dim=1)
        mats = _log_matmul(mats[:, 0::2], mats[:, 1::2])
    alpha = (start + emissions[:, 0]).unsqueeze(1)  # (B, 1, C)
    if mats.size(1) == 1:
        alpha = _log_matmul(alpha, mats[:, 0])
    return torch.logsumexp(alpha.squeeze(1) + end, dim=1)


class LinearChainCRF(nn.Module):
    """Linear-chain CRF over emissions with a contiguous valid prefix per row.

    Emissions are `(B, T, C)`, tags `(B, T)`, mask `(B, T)` with a contiguous
    valid prefix per row (real tokens first, padding after) and at least one
    valid position per row.

    `constraints`, when given, is a `(start_ok, transition_ok, end_ok)` triple
    of boolean tensors -- shapes `(C,)`, `(C, C)`, `(C,)` -- and forbidden
    transitions get a `FORBIDDEN` additive penalty. Decoding always applies
    them, so Viterbi cannot emit a transition you marked invalid.
    `constrain_loss` additionally applies them to the training likelihood,
    which restricts the partition function to valid sequences; it defaults to
    False, so the training objective is unchanged from the standard CRF and
    only the decode is constrained.

    `include_start_end_transitions` (default True) controls whether the
    model has learned start/end bias parameters at all. When False,
    `start_transitions`/`end_transitions` are fixed zero buffers instead of
    parameters -- present so every computation stays branch-free, but frozen
    and contributing nothing.

    `parallel_scan` (default False) computes the log partition in
    `log_likelihood` and `partial_log_likelihood` by a log-depth tree of
    exact log-space matrix products instead of a loop over positions: log2(T)
    sequential steps instead of T, at O(B * T * C^3) work and about
    8 * B * T * C^3 bytes of peak memory with the backward. It pays on a GPU
    with few tags (measured on an RTX 2080 Ti: 4-11x at 9 tags, 4.5-5.6x at
    37 tags and batch 8, slower or out of memory at 37 tags and batch 128)
    and not on CPU. Results agree with the loop to rounding, not bitwise,
    hence opt-in. `decode`, `decode_top_k` and the marginals always use the
    loop.

    `fast_cpu_logsumexp` (default True) speeds up `log_likelihood`,
    `partial_log_likelihood` and the marginals on CPU when their scores hold
    entries far below the rest: forbidden transitions (in the likelihoods
    under `constrain_loss`, in the marginals and `forward_backward` under
    their `constrained` argument), or the tags `partial_log_likelihood`
    masks out. PyTorch's CPU exp takes 16-35x longer on arguments below
    about -87, which such entries produce at every step. The fast path
    clamps those arguments first and recomputes exp's results there
    exactly, so values, gradients and forward-mode tangents matched
    torch.logsumexp bitwise in every test. It runs only in plain eager mode
    (not under TorchScript, torch.compile, torch.export, torch.jit.trace, a
    torch.func transform, or a dispatch mode such as make_fx's or
    FlopCounterMode), in float32, on at most 4 threads, from
    16384 * 2 ** (threads - 1) entries per step (batch * num_tags ** 2),
    where it measured faster on a Threadripper 3970X. The
    likelihoods under `parallel_scan`, `partial_log_likelihood` on up to
    16384 / num_tags ** 2 sequences, and the module-level `forward_backward`
    never use it. Pass False to always use torch.logsumexp.

    `validate` sets how the input checks that read tensor values run (a row
    starting unmasked, tag ids in range, an allowed tag at each valid
    position). "async" (the default) checks CUDA inputs on the device with
    `torch._assert_async`, so no call waits on the GPU; a failure is a
    device-side assert, which is fatal to the CUDA context. "sync" reads each
    check back and raises ValueError, one host sync per check. None skips
    them: a mask that does not start at position 0, or a position that
    allows no tag, then gives a wrong result with no error. On CPU, "async"
    raises ValueError like "sync". Shape checks always run and never sync.

    `log_likelihood` and `partial_log_likelihood` clamp emissions (and, in
    the latter, emissions plus `log_weights`) from below at `FORBIDDEN`
    (-1e4), so a caller may mask a tag out with -inf and still get finite
    gradients; unclamped, an emission that is -inf at a valid position makes
    every gradient nan. Emissions at or above -1e4 are untouched, in value
    and gradient. Below it, an emission is scored as -1e4 and gets zero
    gradient, so a gold path through a -inf emission scores about -1e4
    rather than -inf. `decode` and `decode_with_scores` do not clamp: -inf
    there still rules a tag out absolutely.

    `torch_linear_crf.constraints.encoding_constraints` builds the
    `constraints` triple from a tag encoding (BIO, BIOES, BILOU, IO, IOB,
    IOB-lenient), or build your own mask.

    The penalties are registered buffers, so a checkpoint carries the label
    set it was trained with. Loading a checkpoint under a different label
    ordering would apply the wrong constraints; save your vocab alongside the
    checkpoint.
    """

    if TYPE_CHECKING:
        # Buffers registered in __init__, typed so type checkers see Tensor
        # rather than nn.Module's `Tensor | Module`. Behind TYPE_CHECKING
        # because TorchScript cannot resolve these as string annotations.
        start_penalty: torch.Tensor
        transition_penalty: torch.Tensor
        end_penalty: torch.Tensor

    def __init__(
        self,
        num_tags: int,
        constraints: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        constrain_loss: bool = False,
        include_start_end_transitions: bool = True,
        validate: str | None = "async",
        parallel_scan: bool = False,
        fast_cpu_logsumexp: bool = True,
    ) -> None:
        super().__init__()
        if validate not in ("async", "sync", None):
            raise ValueError(f'validate must be "async", "sync" or None, got {validate!r}')
        self.num_tags = num_tags
        self.validate = validate
        # Resolved once here, so a call pays one attribute read per check.
        self._check_values = validate is not None
        self._async_checks = validate == "async"
        self.constrain_loss = constrain_loss
        self.include_start_end_transitions = include_start_end_transitions
        self.parallel_scan = parallel_scan
        self.fast_cpu_logsumexp = fast_cpu_logsumexp

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        else:
            # Fixed at zero, not learned: every downstream computation reads
            # self.start_transitions/end_transitions unconditionally, so this
            # keeps _scores/_sequence_score/_log_partition/decode branch-free
            # rather than needing an Optional[Tensor] special case throughout.
            # persistent=False so a state_dict from an instance where these were
            # real parameters can't silently overwrite the fixed zero here --
            # loading one raises an unexpected-key error instead.
            self.register_buffer("start_transitions", torch.zeros(num_tags), persistent=False)
            self.register_buffer("end_transitions", torch.zeros(num_tags), persistent=False)

        if constraints is None:
            start_ok = torch.ones(num_tags, dtype=torch.bool)
            transition_ok = torch.ones(num_tags, num_tags, dtype=torch.bool)
            end_ok = torch.ones(num_tags, dtype=torch.bool)
        else:
            start_ok, transition_ok, end_ok = constraints
            if (
                start_ok.shape != (num_tags,)
                or transition_ok.shape != (num_tags, num_tags)
                or end_ok.shape != (num_tags,)
            ):
                raise ValueError(
                    f"constraints do not match num_tags={num_tags}: "
                    f"{tuple(start_ok.shape)}, {tuple(transition_ok.shape)}, {tuple(end_ok.shape)}"
                )

        self.register_buffer(
            "start_penalty", torch.zeros(num_tags).masked_fill(~start_ok, FORBIDDEN)
        )
        self.register_buffer(
            "transition_penalty",
            torch.zeros(num_tags, num_tags).masked_fill(~transition_ok, FORBIDDEN),
        )
        self.register_buffer("end_penalty", torch.zeros(num_tags).masked_fill(~end_ok, FORBIDDEN))

    def _check_shapes(self, emissions: torch.Tensor, mask: torch.Tensor) -> None:
        """Shape/rank checks with clear messages, kept TorchScript-safe.

        Plain `.size(i)` calls and string concatenation, no f-string over a
        sliced `.shape`: TorchScript cannot statically size a List slice.
        """
        if emissions.dim() != 3:
            raise ValueError(
                "emissions must have 3 dimensions (batch, length, num_tags), got "
                + str(emissions.dim())
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                "emissions' last dimension must equal num_tags="
                + str(self.num_tags)
                + ", got "
                + str(emissions.size(2))
            )
        if mask.dim() != 2:
            raise ValueError("mask must have 2 dimensions (batch, length), got " + str(mask.dim()))
        if emissions.size(0) != mask.size(0) or emissions.size(1) != mask.size(1):
            raise ValueError(
                "emissions and mask must agree on (batch, length); got emissions "
                + str(emissions.size(0))
                + "x"
                + str(emissions.size(1))
                + " vs mask "
                + str(mask.size(0))
                + "x"
                + str(mask.size(1))
            )
        if self._check_values:
            self._check(
                mask[:, 0].to(torch.bool).all(),
                "every row must have at least one labeled position, with the valid "
                "prefix starting at position 0; drop rows where mask.sum(1) == 0 "
                "before calling",
            )

    def _check(self, ok: torch.Tensor, message: str) -> None:
        """`_check_value` under this module's `validate` mode."""
        _check_value(ok, message, self._async_checks)

    def _scores(
        self, constrained: bool, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transition parameters cast to `dtype` (fp32 by default), with
        constraint penalties if asked. Callers pass a higher-precision dtype
        (e.g. fp64) to avoid silently downcasting an fp64 model to fp32."""
        if self.transitions.dtype == torch.float16 or self.transitions.dtype == torch.bfloat16:
            # stacklevel=1: the depth below the public method varies, and a fixed
            # location is what makes Python's default filter show it once per process.
            warnings.warn(
                "LinearChainCRF's parameters are "
                + str(self.transitions.dtype)
                + ", which keeps about 3 significant digits, so small training updates "
                "to them round away. The CRF computes in fp32 regardless; keep its "
                "parameters in fp32, e.g. call .float() on the CRF after casting the "
                "rest of the model to 16-bit.",
                stacklevel=1,
            )
        start = self.start_transitions.to(dtype)
        transitions = self.transitions.to(dtype)
        end = self.end_transitions.to(dtype)
        if constrained:
            start = start + self.start_penalty
            transitions = transitions + self.transition_penalty
            end = end + self.end_penalty
        return start, transitions, end

    @torch.jit.export
    def potentials(
        self, constrained: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The (start, transitions, end) scores, shapes (C,), (C, C), (C,).

        In fp32, or the parameters' dtype where that is wider (an fp64 model
        gives fp64). With `constrained` (the default) the constraint penalties
        are added, so these are exactly the scores `decode` maximizes over;
        without it, the learned parameters alone. They keep their gradient, so
        a caller can train through them (e.g. to distill one CRF's path
        distribution into another); detach them to use them as fixed
        potentials.
        """
        return self._scores(constrained, torch.promote_types(self.transitions.dtype, torch.float32))

    @torch.jit.export
    def forward_backward(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        constrained: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(alpha, beta, log_z) over this CRF's potentials; see `forward_backward`.

        With `constrained` (the default) the constraint penalties are
        included, so this is the distribution `decode` maximizes over. That
        differs from the likelihoods, which include them only with
        `constrain_loss`. Gradients reach the emissions and the parameters.
        """
        self._check_forward_backward(emissions, mask, temperature)
        return self._forward_backward(emissions, mask, constrained, temperature)

    def _check_forward_backward(
        self, emissions: torch.Tensor, mask: torch.Tensor, temperature: float
    ) -> None:
        self._check_shapes(emissions, mask)
        if not temperature > 0.0:
            raise ValueError("temperature must be > 0, got " + str(temperature))

    def _forward_backward(
        self, emissions: torch.Tensor, mask: torch.Tensor, constrained: bool, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unchecked, so each public method checks once at its own level: a
        # check nested a method deeper costs torch.compile an extra graph break.
        dtype = torch.promote_types(emissions.dtype, torch.float32)
        start, transitions, end = self._scores(constrained, dtype)
        clamped = self._clamps_logsumexp(emissions, constrained, False)
        return _guarded_forward_backward(
            emissions, mask, start, transitions, end, temperature, clamped
        )

    @torch.jit.export
    def log_marginals(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        constrained: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """log P(y_t = c | emissions), shape (B, T, C); padding positions are meaningless.

        Arguments as in `forward_backward`.
        """
        self._check_forward_backward(emissions, mask, temperature)
        alpha, beta, log_z = self._forward_backward(emissions, mask, constrained, temperature)
        return alpha + beta - log_z.view(-1, 1, 1)

    @torch.jit.export
    def marginals(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        constrained: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """P(y_t = c | emissions), shape (B, T, C), zero at padding positions.

        Arguments as in `forward_backward`.
        """
        self._check_forward_backward(emissions, mask, temperature)
        alpha, beta, log_z = self._forward_backward(emissions, mask, constrained, temperature)
        probs = (alpha + beta - log_z.view(-1, 1, 1)).exp()
        return torch.where(mask.to(torch.bool).unsqueeze(2), probs, torch.zeros_like(probs))

    @torch.jit.export
    def log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-sequence log p(tags | emissions), shape (B,)."""
        self._check_shapes(emissions, mask)
        if (
            tags.dim() != 2
            or tags.size(0) != emissions.size(0)
            or tags.size(1) != emissions.size(1)
        ):
            raise ValueError("tags must have shape (batch, length) matching emissions")
        if self._check_values:
            self._check(
                ((tags >= 0) & (tags < self.num_tags)).all(),
                "tags must be valid tag ids in [0, num_tags="
                + str(self.num_tags)
                + ") at every position, including padding; an ignore-index "
                "convention like -100 is not supported -- pack padded "
                "positions with any in-range id (e.g. 0), since mask already "
                "excludes them from the score",
            )
        if torch.jit.is_scripting():
            # See _autocast_device_type: only the scripted path is restricted
            # to cuda/cpu.
            if emissions.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    return self._log_likelihood_impl(emissions, tags, mask)
            else:
                with torch.autocast(device_type="cpu", enabled=False):
                    return self._log_likelihood_impl(emissions, tags, mask)
        else:
            with torch.autocast(device_type=_autocast_device_type(emissions.device), enabled=False):
                return self._log_likelihood_impl(emissions, tags, mask)

    def _log_likelihood_impl(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # fp32 floor, not a fp32 ceiling: promote_types leaves an already-fp64
        # model at fp64 instead of silently downcasting it to fp32.
        dtype = torch.promote_types(emissions.dtype, torch.float32)
        emissions = _floor_emissions(emissions.to(dtype))
        mask = mask.to(torch.bool)
        start, transitions, end = self._scores(self.constrain_loss, dtype)
        numerator = self._sequence_score(emissions, tags, mask, start, transitions, end)
        clamped = not self.parallel_scan and self._clamps_logsumexp(
            emissions, self.constrain_loss, False
        )
        denominator = self._log_partition(emissions, mask, start, transitions, end, clamped)
        return numerator - denominator

    @torch.jit.export
    def partial_log_likelihood(
        self,
        emissions: torch.Tensor,
        allowed: torch.Tensor,
        mask: torch.Tensor,
        log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sequence log P(y_t in allowed[t] for every t | emissions), shape (B,).

        The partial-label ("fuzzy") CRF likelihood, for training on partially
        annotated sequences: the probability mass of every tag path that stays
        inside the allowed sets,

            log sum_{y : y_t in allowed[t]} exp(score(y)) - log Z,

        computed as a forward pass with every disallowed emission replaced by
        `FORBIDDEN`, minus the ordinary log partition. With exactly one allowed
        tag per valid position it equals `log_likelihood` of those tags.
        `allowed` is `(B, T, C)` bool, and every valid position must allow at
        least one tag. Uses the same scores as `log_likelihood` (the constraint
        penalties iff `constrain_loss`), in fp32 or higher, with autocast off.
        With `constrain_loss`, at least one allowed path must also be valid
        under the constraints: otherwise disallowed tags and forbidden
        transitions cost the same finite `FORBIDDEN`, and the result is not a
        meaningful probability.

        `log_weights` `(B, T, C)` weights each allowed path by the
        product of its per-position weights: the numerator becomes
        log sum_{y allowed} exp(score(y)) * prod_t w_t(y_t), computed by adding
        log w to the emissions of the numerator's forward pass only. That is
        how a teacher's per-token distribution can weight the paths a
        partial label allows. A weight of zero (log weight -inf) is legal:
        the sum is clamped at `FORBIDDEN` like the emissions (see the class
        docstring), which leaves that tag allowed at weight exp(-1e4), 0 in
        fp32.

        `FORBIDDEN` rather than -inf: a disallowed tag's forward entry would
        be a logsumexp over all -inf, whose backward is nan even though the
        entry never reaches the result. exp(FORBIDDEN) is exactly 0 in fp32,
        so the value is unchanged at realistic logit scale.
        """
        self._check_shapes(emissions, mask)
        allowed = self._checked_allowed(allowed, emissions, mask)
        if log_weights is not None and not self._same_shape(log_weights, emissions):
            raise ValueError(
                "log_weights must have shape (batch, length, num_tags) matching emissions"
            )
        if torch.jit.is_scripting():
            # See _autocast_device_type: only the scripted path is restricted
            # to cuda/cpu.
            if emissions.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    return self._partial_log_likelihood_impl(emissions, allowed, mask, log_weights)
            else:
                with torch.autocast(device_type="cpu", enabled=False):
                    return self._partial_log_likelihood_impl(emissions, allowed, mask, log_weights)
        else:
            with torch.autocast(device_type=_autocast_device_type(emissions.device), enabled=False):
                return self._partial_log_likelihood_impl(emissions, allowed, mask, log_weights)

    def _checked_allowed(
        self, allowed: torch.Tensor, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`allowed` as bool, after checking its shape and that each valid position allows a tag."""
        if not self._same_shape(allowed, emissions):
            raise ValueError("allowed must have shape (batch, length, num_tags) matching emissions")
        allowed = allowed.to(torch.bool)
        if self._check_values:
            self._check(
                (allowed.any(2) | ~mask.to(torch.bool)).all(),
                "every valid position must allow at least one tag",
            )
        return allowed

    @staticmethod
    def _same_shape(x: torch.Tensor, emissions: torch.Tensor) -> bool:
        return (
            x.dim() == 3
            and x.size(0) == emissions.size(0)
            and x.size(1) == emissions.size(1)
            and x.size(2) == emissions.size(2)
        )

    def _partial_log_likelihood_impl(
        self,
        emissions: torch.Tensor,
        allowed: torch.Tensor,
        mask: torch.Tensor,
        log_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = torch.promote_types(emissions.dtype, torch.float32)
        emissions = _floor_emissions(emissions.to(dtype))
        mask = mask.to(torch.bool)
        start, transitions, end = self._scores(self.constrain_loss, dtype)
        weighted = (
            emissions
            if log_weights is None
            else _floor_emissions(emissions + log_weights.to(dtype))
        )
        # -1e4 is FORBIDDEN, written out: TorchScript cannot read a module global.
        restricted = weighted.masked_fill(~allowed, -1e4)
        batch = emissions.size(0)
        num_tags = emissions.size(2)
        # Numerator and denominator as one 2B-row recursion: the same per-row
        # arithmetic (bitwise equal), half the steps. The recursion is
        # launch-bound on accelerators, so this always pays there; on CPU it
        # pays only while each step's (2B, C, C) temporary stays small
        # (measured on a Threadripper 2990WX: 1.2-2x up to B * C^2 of about 11k, 0.7-1x
        # from about 44k), hence the cutoff.
        if emissions.device.type != "cpu" or batch * num_tags * num_tags <= 16384:
            stacked = torch.cat([restricted, emissions])
            # No clamped logsumexp: it measured slower at these sizes on CPU.
            both = self._log_partition(stacked, mask.repeat(2, 1), start, transitions, end, False)
            return both[:batch] - both[batch:]
        scan = self.parallel_scan
        clamped = not scan and self._clamps_logsumexp(emissions, self.constrain_loss, True)
        numerator = self._log_partition(restricted, mask, start, transitions, end, clamped)
        clamped = not scan and self._clamps_logsumexp(emissions, self.constrain_loss, False)
        denominator = self._log_partition(emissions, mask, start, transitions, end, clamped)
        return numerator - denominator

    def _sequence_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        start: torch.Tensor,
        transitions: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """Unnormalized score of the given tag sequence."""
        # A masked select, not emissions.gather: on torch 2.4-2.12, inductor
        # drops gather's gradient when it is summed with _log_partition's, so
        # a compiled log_likelihood trained on a wrong gradient. Values and
        # gradients are bitwise those of gather; `where` keeps a non-finite
        # emission at a non-gold tag out of the sum. The mask is a bool
        # comparison rather than one_hot, which would build an int64 tensor first.
        gold = tags.unsqueeze(2) == torch.arange(emissions.size(2), device=tags.device)
        emit = torch.where(gold, emissions, torch.zeros_like(emissions)).sum(2)  # (B, T)
        score = start[tags[:, 0]] + emit[:, 0]
        if emissions.size(1) > 1:
            step = transitions[tags[:, :-1], tags[:, 1:]] + emit[:, 1:]
            # torch.where, not step * mask: a masked position can carry a
            # non-finite emission (e.g. -inf padding upstream), and
            # (-inf) * False is nan, which would poison the whole row's score
            # and gradient rather than being correctly dropped.
            score = score + torch.where(mask[:, 1:], step, torch.zeros_like(step)).sum(1)
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + end[last]

    def _clamps_logsumexp(self, emissions: torch.Tensor, constrained: bool, masked: bool) -> bool:
        """Whether a loop recursion over `emissions` uses `_ClampedLogSumExp`.

        Only where it measured faster than torch.logsumexp on a Threadripper 3970X: in
        plain eager mode on CPU in float32, with entries far below their row's
        maximum (forbidden transitions under `constrained`, or emissions
        masked to FORBIDDEN when `masked`), on at most 4 threads, from
        16384 * 2 ** (threads - 1) entries per step. torch.logsumexp's slow
        path spreads over threads, so the crossover rises with them.
        `_logsumexp` also requires the scores themselves to be float32, which
        an fp64 CRF's under `constrain_loss` are not.
        """
        if torch.jit.is_scripting():
            return False
        # Before the thread count: Dynamo cannot trace torch.get_num_threads.
        if not _is_plain_eager():
            return False
        threads = torch.get_num_threads()
        if (
            not self.fast_cpu_logsumexp
            or emissions.device.type != "cpu"
            or torch.promote_types(emissions.dtype, torch.float32) != torch.float32
            or threads > 4
            or emissions.size(0) * self.num_tags * self.num_tags < 16384 << (threads - 1)
        ):
            return False
        return masked or (constrained and self._has_forbidden())

    def _has_forbidden(self) -> bool:
        """Whether any constraint penalty is set."""
        return bool(
            (self.transition_penalty != 0).any()
            or (self.start_penalty != 0).any()
            or (self.end_penalty != 0).any()
        )

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        start: torch.Tensor,
        transitions: torch.Tensor,
        end: torch.Tensor,
        clamped: bool,
    ) -> torch.Tensor:
        """log sum over all tag sequences (all *valid* ones, if constrained).

        `clamped` selects `_ClampedLogSumExp` for the loop (see `_clamps_logsumexp`).
        """
        # A padding position's emission can be non-finite (e.g. -inf padding
        # fed in upstream); logsumexp over an all -inf row gives a finite
        # forward result once the `where` below discards it, but its *local*
        # backward Jacobian is still nan, and `where`'s backward is
        # 0 * nan = nan, not 0. So substitute a finite placeholder first;
        # padding values are only ever used through that `where`.
        safe = torch.where(mask.unsqueeze(2), emissions, torch.zeros_like(emissions))
        if self.parallel_scan:
            return _log_partition_scan(safe, mask, start, transitions, end)
        # unbind, not emissions[:, t] per step: each select's backward
        # materializes a zero (B, T, C) gradient, which made the backward
        # O(B * T^2 * C).
        steps = safe.unbind(1)
        valid_steps = mask.unbind(1)
        alpha = start + steps[0]  # (B, C)
        for t in range(1, len(steps)):
            valid = valid_steps[t].unsqueeze(1)
            scores = alpha.unsqueeze(2) + transitions.unsqueeze(0) + steps[t].unsqueeze(1)
            # Past a row's length, carry alpha forward unchanged.
            alpha = torch.where(valid, _logsumexp(scores, 1, clamped), alpha)
        return _logsumexp(alpha + end, 1, clamped)

    def _viterbi(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared Viterbi core for `decode`/`decode_with_scores`.

        Returns (paths, best_scores, lengths): paths is (B, T) with the full
        (unsliced) decoded tag ids, best_scores is (B,), lengths is (B,).
        Callers cut `paths[i]` to `lengths[i]`.
        """
        if torch.jit.is_scripting():
            # See _autocast_device_type: only the scripted path is restricted
            # to cuda/cpu.
            if emissions.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    return self._viterbi_impl(emissions, mask, allowed)
            else:
                with torch.autocast(device_type="cpu", enabled=False):
                    return self._viterbi_impl(emissions, mask, allowed)
        else:
            with torch.autocast(device_type=_autocast_device_type(emissions.device), enabled=False):
                return self._viterbi_impl(emissions, mask, allowed)

    def _sorts_rows(self, emissions: torch.Tensor, entries: int) -> bool:
        """Whether a decode sorts rows by length and steps only the running ones.

        On CPU from 8192 `entries` per step (below that the slicing cost more
        than it saved on a Threadripper 3970X, 1 and 4 threads), in eager mode or
        TorchScript. The running row counts are read to the host, which would
        break a torch.compile graph and be fixed into a torch.jit.trace.
        """
        if emissions.device.type != "cpu" or entries < 8192:
            return False
        if torch.jit.is_scripting():
            return True
        return _is_plain_eager()

    def _viterbi_impl(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = torch.promote_types(emissions.dtype, torch.float32)
        emissions = emissions.to(dtype)
        if allowed is not None:
            # -inf, not FORBIDDEN: decode takes no gradient, and an excluded tag
            # must lose even to a path through a forbidden transition.
            emissions = emissions.masked_fill(~allowed, float("-inf"))
        mask = mask.to(torch.bool)
        lengths = mask.sum(1)
        start, transitions, end = self._scores(True, dtype)
        batch, length, num_tags = emissions.shape

        # On CPU, sort rows by length so each step computes only the rows
        # still running (a prefix), as in `_viterbi_top_k_impl`.
        order: torch.Tensor | None = None
        running: list[int] = [batch] * length
        if self._sorts_rows(emissions, batch * num_tags * num_tags):
            order = lengths.argsort(descending=True)
            emissions = emissions.index_select(0, order)
            mask = mask.index_select(0, order)
            counts: list[int] = mask.sum(0).tolist()  # rows running at each step
            running = counts

        # Scores are laid out (B, next tag, previous tag), so the max reduces
        # over contiguous memory. Slicing every step up front costs one op
        # instead of two per step, which is most of the time on short input.
        transitions_t = transitions.t().contiguous()
        steps = emissions.unsqueeze(3).unbind(1)  # T x (B, C, 1)
        valid = mask.unsqueeze(2).unbind(1)  # T x (B, 1)
        score = start + emissions[:, 0]
        backpointers: list[torch.Tensor] = []
        for t in range(1, length):
            n = running[t]
            if n == 0:
                break
            if order is None:
                best, best_previous = (score.unsqueeze(1) + transitions_t + steps[t]).max(2)
                score = torch.where(valid[t], best, score)
            elif n == batch:
                score, best_previous = (score.unsqueeze(1) + transitions_t + steps[t]).max(2)
            else:
                # Sorted by length, the first n rows are exactly those still running.
                best, best_previous = (score[:n].unsqueeze(1) + transitions_t + steps[t][:n]).max(2)
                score = torch.cat([best, score[n:]])
            backpointers.append(best_previous)
        final = score + end
        best_score, current = final.max(1, keepdim=True)

        # Vectorized backtrace on (B, 1) tags. Past a row's length its tag is
        # frozen, so the tag chosen at its true last position is what lands
        # there. The path is collected and concatenated once.
        path: list[torch.Tensor] = [current]
        if order is None and (emissions.device.type != "cpu" or batch * length * num_tags <= 65536):
            if length > 1:
                # Mask all pointers at once: at a padding position every tag
                # points to itself, so each step is one gather. On CPU the two
                # passes over the stacked (B, T, C) pointers cost more than
                # they save past about 65k entries (measured on a Threadripper 2990WX).
                identity = torch.arange(num_tags, device=emissions.device).view(1, 1, num_tags)
                pointers = torch.where(
                    mask[:, 1:].unsqueeze(2), torch.stack(backpointers, dim=1), identity
                ).unbind(1)
                for t in range(length - 1, 0, -1):
                    current = pointers[t - 1].gather(1, current)
                    path.append(current)
        elif order is None:
            for t in range(length - 1, 0, -1):
                current = torch.where(valid[t], backpointers[t - 1].gather(1, current), current)
                path.append(current)
        else:
            for t in range(length - 1, 0, -1):
                if t <= len(backpointers):
                    pointer = backpointers[t - 1]
                    n = pointer.size(0)
                    if n == batch:
                        current = pointer.gather(1, current)
                    else:
                        current = torch.cat([pointer.gather(1, current[:n]), current[n:]])
                path.append(current)
        path.reverse()
        tags = torch.cat(path, dim=1)
        best_score = best_score.squeeze(1)
        if order is not None:
            inverse = order.argsort()
            tags = tags.index_select(0, inverse)
            best_score = best_score.index_select(0, inverse)
        return tags, best_score, lengths

    @torch.jit.export
    @torch.no_grad()
    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None = None
    ) -> list[list[int]]:
        """Constrained Viterbi decode; returns one tag-id list per row, cut to length.

        Takes padded input like every method here (`mask` marks each row's
        valid prefix) and returns host lists, so the call waits for the device;
        `decode_to_tensor` returns the same paths as device tensors instead.

        `allowed`, `(B, T, C)` bool as in `partial_log_likelihood`, restricts
        each valid position to its allowed tags, absolutely: a disallowed tag
        is never chosen, even where every path through the allowed tags
        crosses a forbidden transition. Every valid position must allow at
        least one tag.
        """
        self._check_shapes(emissions, mask)
        if allowed is not None:
            allowed = self._checked_allowed(allowed, emissions, mask)
        out, _, lengths = self._viterbi(emissions, mask, allowed)
        out_list: list[list[int]] = out.tolist()
        lengths_list: list[int] = lengths.tolist()
        return [out_list[i][: lengths_list[i]] for i in range(len(out_list))]

    @torch.jit.export
    @torch.no_grad()
    def decode_to_tensor(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allowed: torch.Tensor | None = None,
        pad_tag: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`decode_with_scores` as device tensors: (tags `(B, T)` long, scores `(B,)`).

        The same paths and scores, left on the emissions' device with no host
        sync, for callers that keep predictions there (pseudo-labels, input to
        another model, metrics computed with tensor ops). Positions outside
        `mask` hold `pad_tag`. Input and `allowed` are as in `decode`.
        """
        self._check_shapes(emissions, mask)
        if allowed is not None:
            allowed = self._checked_allowed(allowed, emissions, mask)
        return self._padded_viterbi(emissions, mask, allowed, pad_tag)

    def _padded_viterbi(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allowed: torch.Tensor | None,
        pad_tag: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Undecorated: TorchScript cannot resolve `torch` inside a no_grad body.
        out, best_score, _ = self._viterbi(emissions, mask, allowed)
        return out.masked_fill(~mask.to(torch.bool), pad_tag), best_score

    @torch.jit.export
    @torch.no_grad()
    def decode_with_scores(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None = None
    ) -> list[tuple[list[int], float]]:
        """Like `decode`, but pairs each row's path with its Viterbi score."""
        self._check_shapes(emissions, mask)
        if allowed is not None:
            allowed = self._checked_allowed(allowed, emissions, mask)
        out, best_score, lengths = self._viterbi(emissions, mask, allowed)
        out_list: list[list[int]] = out.tolist()
        lengths_list: list[int] = lengths.tolist()
        scores_list: list[float] = best_score.tolist()
        return [(out_list[i][: lengths_list[i]], scores_list[i]) for i in range(len(out_list))]

    @torch.jit.export
    @torch.no_grad()
    def decode_top_k(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        k: int,
        allowed: torch.Tensor | None = None,
    ) -> list[list[tuple[list[int], float]]]:
        """The `k` best paths per row with their scores, best first.

        Exact: the same constrained scores `decode_with_scores` maximizes,
        and its first entry is `decode_with_scores`' result up to ties. A
        row with fewer than `k` paths of finite score (a short row, or one
        restricted by `allowed`, which is as in `decode`) returns only those.
        Paths through forbidden transitions have finite scores and are
        included, after every path that avoids them.
        """
        self._check_shapes(emissions, mask)
        if k < 1:
            raise ValueError("k must be at least 1, got " + str(k))
        if allowed is not None:
            allowed = self._checked_allowed(allowed, emissions, mask)
        paths, scores, lengths = self._viterbi_top_k(emissions, mask, allowed, k)
        paths_list: list[list[list[int]]] = paths.tolist()
        scores_list: list[list[float]] = scores.tolist()
        lengths_list: list[int] = lengths.tolist()
        result: list[list[tuple[list[int], float]]] = []
        for i in range(len(paths_list)):
            row: list[tuple[list[int], float]] = []
            for j in range(len(paths_list[i])):
                if scores_list[i][j] > float("-inf"):
                    # A loop: TorchScript rejects comprehensions with an `if`.
                    row.append((paths_list[i][j][: lengths_list[i]], scores_list[i][j]))  # noqa: PERF401
            result.append(row)
        return result

    def _viterbi_top_k(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`_viterbi_top_k_impl` with autocast off, as in `_viterbi`."""
        if torch.jit.is_scripting():
            # See _autocast_device_type: only the scripted path is restricted
            # to cuda/cpu.
            if emissions.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    return self._viterbi_top_k_impl(emissions, mask, allowed, k)
            else:
                with torch.autocast(device_type="cpu", enabled=False):
                    return self._viterbi_top_k_impl(emissions, mask, allowed, k)
        else:
            with torch.autocast(device_type=_autocast_device_type(emissions.device), enabled=False):
                return self._viterbi_top_k_impl(emissions, mask, allowed, k)

    def _viterbi_top_k_impl(
        self, emissions: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor | None, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(paths (B, k, T), scores (B, k), lengths (B,)), best first; -inf marks a missing path.

        `score[b, c, r]` is the r-th best score of a prefix ending in tag c.
        Each step takes the k best of the C * k (previous tag, rank)
        candidates for every tag, laid out last so `topk` reduces over
        contiguous memory; a backpointer stores the winner's flat index
        previous_tag * k + rank. Slots that start at -inf stand for paths
        that do not exist and stay -inf.

        Where `_sorts_rows`, rows are sorted by length and each step computes
        only the rows still running (a prefix); otherwise every step computes
        every row.
        """
        dtype = torch.promote_types(emissions.dtype, torch.float32)
        emissions = emissions.to(dtype)
        if allowed is not None:
            emissions = emissions.masked_fill(~allowed, float("-inf"))
        mask = mask.to(torch.bool)
        lengths = mask.sum(1)
        start, transitions, end = self._scores(True, dtype)
        batch, length, num_tags = emissions.shape
        device = emissions.device

        order: torch.Tensor | None = None
        running: list[int] = [batch] * length
        if self._sorts_rows(emissions, batch * num_tags * num_tags * k):
            order = lengths.argsort(descending=True)
            emissions = emissions.index_select(0, order)
            mask = mask.index_select(0, order)
            counts: list[int] = mask.sum(0).tolist()  # rows running at each step
            running = counts

        score = torch.full((batch, num_tags, k), float("-inf"), dtype=dtype, device=device)
        score[:, :, 0] = start + emissions[:, 0]
        # At a padding position every slot points back to itself.
        identity = (
            torch.arange(num_tags, device=device).view(1, num_tags, 1) * k
            + torch.arange(k, device=device).view(1, 1, k)
        ).expand(batch, num_tags, k)
        transitions_t = transitions.t().contiguous().view(1, num_tags, num_tags, 1)
        steps = emissions.reshape(batch, length, num_tags, 1, 1).unbind(1)  # T x (B, C, 1, 1)
        valid = mask.reshape(batch, length, 1, 1).unbind(1)  # T x (B, 1, 1)
        backpointers: list[torch.Tensor] = []
        for t in range(1, length):
            n = running[t]
            if n == 0:
                break
            if order is None:
                # (B, C_next, C_prev, k): candidates for each tag in its last two dims.
                candidates = score.unsqueeze(1) + transitions_t + steps[t]
                best, index = candidates.view(batch, num_tags, num_tags * k).topk(k, dim=2)
                score = torch.where(valid[t], best, score)
                pointer = torch.where(valid[t], index, identity)
                backpointers.append(pointer.view(batch, num_tags * k))
            else:
                # Sorted by length, the first n rows are exactly those still running.
                if n == batch:
                    candidates = score.unsqueeze(1) + transitions_t + steps[t]
                else:
                    candidates = score[:n].unsqueeze(1) + transitions_t + steps[t][:n]
                best, index = candidates.view(n, num_tags, num_tags * k).topk(k, dim=2)
                score = best if n == batch else torch.cat([best, score[n:]])
                backpointers.append(index.view(n, num_tags * k))

        final = (score + end.view(1, num_tags, 1)).view(batch, num_tags * k)
        best_scores, index = final.topk(k, dim=1)  # (B, k), index = tag * k + rank
        path: list[torch.Tensor] = [index]
        for t in range(length - 1, 0, -1):
            # Rows past their length, and steps no row reached, keep their index.
            if t <= len(backpointers):
                pointer = backpointers[t - 1]
                n = pointer.size(0)
                if n == batch:
                    index = pointer.gather(1, index)
                else:
                    index = torch.cat([pointer.gather(1, index[:n]), index[n:]])
            path.append(index)
        path.reverse()
        tags = torch.div(torch.stack(path, dim=2), k, rounding_mode="floor")  # (B, k, T)
        if order is not None:
            inverse = order.argsort()
            tags = tags.index_select(0, inverse)
            best_scores = best_scores.index_select(0, inverse)
        return tags, best_scores, lengths
