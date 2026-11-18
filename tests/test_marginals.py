"""Tests for forward-backward and posterior marginals."""

from __future__ import annotations

import itertools

import pytest
import torch

from torch_linear_crf import FORBIDDEN, LinearChainCRF, forward_backward
from torch_linear_crf import crf as crf_module


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


def _constraints(num_tags: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_ok = torch.ones(num_tags, dtype=torch.bool)
    start_ok[2] = False
    transition_ok = torch.ones(num_tags, num_tags, dtype=torch.bool)
    transition_ok[0, 2] = False
    transition_ok[1, 1] = False
    end_ok = torch.ones(num_tags, dtype=torch.bool)
    end_ok[1] = False
    return start_ok, transition_ok, end_ok


def _inputs(seed: int, dtype: torch.dtype = torch.float32):
    torch.manual_seed(seed)
    emissions = (torch.randn(3, 4, 3) * 2.0).to(dtype)
    mask = torch.tensor([[True] * 4, [True, True, True, False], [True, False, False, False]])
    return emissions, mask


def _enumerate_marginals(potentials, emissions, length: int) -> torch.Tensor:
    """P(y_t = c) over every tag sequence of the given length, scored directly."""
    start, transitions, end = potentials
    num_tags = emissions.size(1)
    paths = list(itertools.product(range(num_tags), repeat=length))
    scores = []
    for path in paths:
        score = start[path[0]] + emissions[0, path[0]]
        for t in range(1, length):
            score = score + transitions[path[t - 1], path[t]] + emissions[t, path[t]]
        scores.append(score + end[path[-1]])
    probs = torch.softmax(torch.stack(scores), 0)
    out = torch.zeros(length, num_tags, dtype=probs.dtype)
    for path, prob in zip(paths, probs, strict=True):
        for t, tag in enumerate(path):
            out[t, tag] += prob
    return out


@pytest.mark.parametrize("constrained", [True, False])
def test_marginals_match_enumeration(constrained: bool) -> None:
    emissions, mask = _inputs(30, torch.float64)
    crf = LinearChainCRF(3, constraints=_constraints(3)).double()
    got = crf.marginals(emissions, mask, constrained=constrained)
    potentials = crf.potentials(constrained)
    for row, length in enumerate([4, 3, 1]):
        expected = _enumerate_marginals(potentials, emissions[row], length)
        torch.testing.assert_close(got[row, :length], expected)
        assert torch.equal(got[row, length:], torch.zeros_like(got[row, length:]))
    if constrained:
        # A tag that may not start a sequence has no mass at position 0.
        assert got[:, 0, 2].max() < 1e-300


def test_alpha_beta_agree_with_log_z_at_every_valid_position() -> None:
    emissions, mask = _inputs(31)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    alpha, beta, log_z = crf.forward_backward(emissions, mask)
    per_position = torch.logsumexp(alpha + beta, dim=2)  # (B, T)
    expected = log_z.unsqueeze(1).expand_as(per_position)
    torch.testing.assert_close(per_position[mask], expected[mask])
    assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()


def test_log_z_matches_log_likelihood_partition() -> None:
    emissions, mask = _inputs(32)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    for constrained in (True, False):
        _, _, log_z = crf.forward_backward(emissions, mask, constrained=constrained)
        expected = crf._log_partition(emissions, mask, *crf._scores(constrained), False)
        torch.testing.assert_close(log_z, expected)


def test_marginals_are_the_gradient_of_log_z() -> None:
    """d log Z / d emissions[t, c] = P(y_t = c), zero at padding."""
    emissions, mask = _inputs(33, torch.float64)
    emissions.requires_grad_()
    crf = LinearChainCRF(3, constraints=_constraints(3)).double()
    _, _, log_z = crf.forward_backward(emissions, mask)
    (grad,) = torch.autograd.grad(log_z.sum(), emissions)
    torch.testing.assert_close(grad, crf.marginals(emissions, mask).detach())


def test_function_matches_method() -> None:
    emissions, mask = _inputs(34)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    for constrained in (True, False):
        for got, expected in zip(
            forward_backward(emissions, mask, *crf.potentials(constrained), temperature=2.0),
            crf.forward_backward(emissions, mask, constrained, 2.0),
            strict=True,
        ):
            assert torch.equal(got, expected)


def test_temperature_divides_every_score() -> None:
    emissions, mask = _inputs(35, torch.float64)
    crf = LinearChainCRF(3, constraints=_constraints(3)).double()
    start, transitions, end = crf.potentials()
    got = crf.log_marginals(emissions, mask, temperature=3.0)
    alpha, beta, log_z = forward_backward(emissions / 3, mask, start / 3, transitions / 3, end / 3)
    torch.testing.assert_close(got[mask], (alpha + beta - log_z.view(-1, 1, 1))[mask])
    assert torch.equal(
        crf.log_marginals(emissions, mask, temperature=1.0), crf.log_marginals(emissions, mask)
    )


def test_constraint_only_chain() -> None:
    """The linear-head use: per-token log-softmax with constraint-only potentials
    matches a CRF whose learned scores are all zero."""
    emissions, mask = _inputs(36)
    log_probs = torch.log_softmax(emissions, -1)
    start_ok, transition_ok, end_ok = _constraints(3)
    penalties = (
        torch.zeros(3).masked_fill(~start_ok, FORBIDDEN),
        torch.zeros(3, 3).masked_fill(~transition_ok, FORBIDDEN),
        torch.zeros(3).masked_fill(~end_ok, FORBIDDEN),
    )
    crf = LinearChainCRF(3, constraints=_constraints(3), include_start_end_transitions=False)
    with torch.no_grad():
        crf.transitions.zero_()
    alpha, beta, log_z = forward_backward(log_probs, mask, *penalties)
    torch.testing.assert_close(
        (alpha + beta - log_z.view(-1, 1, 1))[mask], crf.log_marginals(log_probs, mask)[mask]
    )


def test_gradcheck_forward_backward() -> None:
    emissions, mask = _inputs(37, torch.float64)
    torch.manual_seed(37)
    potentials = (
        torch.randn(3, dtype=torch.float64),
        torch.randn(3, 3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    inputs = tuple(x.clone().requires_grad_() for x in (emissions, *potentials))

    def f(e, start, transitions, end):
        alpha, beta, log_z = forward_backward(e, mask, start, transitions, end, temperature=1.5)
        return (alpha + beta - log_z.view(-1, 1, 1))[mask]

    assert torch.autograd.gradcheck(f, inputs)


def test_neg_inf_emissions_give_finite_gradients() -> None:
    emissions, mask = _inputs(38)
    emissions[:, :, 1] = float("-inf")  # every position, padding included
    emissions.requires_grad_()
    crf = LinearChainCRF(3)
    log_marginals = crf.log_marginals(emissions, mask)
    log_marginals[mask].logsumexp(-1).sum().backward()
    assert torch.isfinite(log_marginals[mask]).all()
    assert emissions.grad is not None and torch.isfinite(emissions.grad).all()
    assert torch.isfinite(crf.transitions.grad).all()
    assert crf.marginals(emissions, mask)[:, :, 1].max() == 0


def test_computes_in_fp32_under_bf16_autocast() -> None:
    emissions, mask = _inputs(39)
    crf = LinearChainCRF(3)
    reference = crf.marginals(emissions, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = crf.marginals(emissions, mask)
        seen = []
        inner = crf_module._forward_backward_impl

        def spy(*args):
            seen.append(torch.is_autocast_enabled("cpu"))
            return inner(*args)

        crf_module._forward_backward_impl = spy
        try:
            forward_backward(emissions, mask, *crf.potentials())
        finally:
            crf_module._forward_backward_impl = inner
    assert got.dtype is torch.float32
    assert torch.equal(got, reference)
    assert seen == [False]


def test_fp64_stays_fp64() -> None:
    emissions, mask = _inputs(40, torch.float64)
    crf = LinearChainCRF(3)
    alpha, beta, log_z = crf.forward_backward(emissions, mask)
    assert alpha.dtype is beta.dtype is log_z.dtype is torch.float64


def test_scripts_and_matches_eager() -> None:
    emissions, mask = _inputs(41)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    scripted = torch.jit.script(crf)
    for constrained in (True, False):
        assert torch.equal(
            crf.marginals(emissions, mask, constrained, 2.0),
            scripted.marginals(emissions, mask, constrained, 2.0),
        )
    scripted_fn = torch.jit.script(forward_backward)
    for got, expected in zip(
        scripted_fn(emissions, mask, *crf.potentials()),
        forward_backward(emissions, mask, *crf.potentials()),
        strict=True,
    ):
        assert torch.equal(got, expected)


def test_compiles_with_torch_compile() -> None:
    emissions, mask = _inputs(42)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    compiled = torch.compile(crf.marginals)
    torch.testing.assert_close(compiled(emissions, mask), crf.marginals(emissions, mask))
    # On CPU the mask-prefix check is the one graph break. Before torch 2.12,
    # torch.amp.is_autocast_available in the autocast guard adds two more.
    explained = torch._dynamo.explain(crf.marginals)(emissions, mask)
    assert explained.op_count > 0
    limit = 1 if _torch_version() >= (2, 12) else 3
    assert explained.graph_break_count <= limit, explained.break_reasons


def test_validation() -> None:
    emissions, mask = _inputs(43)
    crf = LinearChainCRF(3)
    start, transitions, end = crf.potentials()
    with pytest.raises(ValueError, match="shapes"):
        forward_backward(emissions, mask, start[:2], transitions, end)
    with pytest.raises(ValueError, match="shapes"):
        forward_backward(emissions, mask, start, transitions[:, :2], end)
    with pytest.raises(ValueError, match="mask must have shape"):
        forward_backward(emissions, mask[:, :3], start, transitions, end)
    with pytest.raises(ValueError, match="temperature"):
        forward_backward(emissions, mask, start, transitions, end, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        crf.marginals(emissions, mask, temperature=-1.0)
    with pytest.raises(ValueError, match="validate must be"):
        forward_backward(emissions, mask, start, transitions, end, validate="yes")
    bad_mask = mask.clone()
    bad_mask[1, 0] = False
    with pytest.raises(ValueError, match="at least one valid position"):
        forward_backward(emissions, bad_mask, start, transitions, end)
    with pytest.raises(ValueError, match="at least one labeled position"):
        crf.marginals(emissions, bad_mask)
    # validate=None skips the value check.
    assert forward_backward(emissions, bad_mask, start, transitions, end, validate=None)


@pytest.mark.cuda
def test_matches_cpu_without_host_sync_cuda() -> None:
    emissions, mask = _inputs(44)
    crf = LinearChainCRF(3, constraints=_constraints(3))
    reference = crf.marginals(emissions, mask)
    crf = crf.cuda()
    emissions = emissions.cuda().requires_grad_()
    mask = mask.cuda()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = crf.marginals(emissions, mask)
        _, _, log_z = forward_backward(emissions, mask, *crf.potentials())
        (got.sum() + log_z.sum()).backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(got.cpu(), reference)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        assert crf.marginals(emissions.bfloat16(), mask).dtype is torch.float32
