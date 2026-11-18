"""pad_to_length_bucket: padding the length is invisible to every method."""

from __future__ import annotations

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from torch_linear_crf import LinearChainCRF, pad_to_length_bucket


def _inputs(length: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    emissions = torch.randn(4, length, 5, generator=g) * 3
    tags = torch.randint(0, 5, (4, length), generator=g)
    mask = torch.ones(4, length, dtype=torch.bool)
    mask[1, max(1, length // 2) :] = False
    mask[2, 1:] = False
    allowed = torch.rand(4, length, 5, generator=g) > 0.4
    allowed[..., 2] = True
    log_weights = torch.log_softmax(torch.randn(4, length, 5, generator=g), -1)
    return emissions, tags, mask, allowed, log_weights


def _constraints():
    transition_ok = torch.ones(5, 5, dtype=torch.bool)
    transition_ok[0, 3] = transition_ok[4, 4] = False
    ok = torch.ones(5, dtype=torch.bool)
    return ok, transition_ok, ok


def test_shapes_values_and_no_op() -> None:
    emissions, tags, mask, allowed, log_weights = _inputs(13)
    e, m, t, a, w = pad_to_length_bucket(emissions, mask, tags, allowed, log_weights)
    assert [x.size(1) for x in (e, m, t, a, w)] == [16] * 5
    assert not m[:, 13:].any() and not a[:, 13:].any()
    assert (t[:, 13:] == 0).all() and (e[:, 13:] == 0).all()
    assert m.dtype is torch.bool and t.dtype is torch.long
    assert torch.equal(e[:, :13], emissions) and torch.equal(m[:, :13], mask)
    same = pad_to_length_bucket(emissions[:, :8], mask[:, :8], min_length=8)
    assert same[0] is emissions[:, :8] or same[0].size(1) == 8
    sizes = {
        n: pad_to_length_bucket(torch.zeros(1, n, 1), torch.ones(1, n, dtype=torch.bool))[0].size(1)
        for n in (1, 16, 17, 33, 64, 65)
    }
    assert sizes == {1: 16, 16: 16, 17: 32, 33: 64, 64: 64, 65: 128}
    e48 = pad_to_length_bucket(
        torch.zeros(1, 33, 1), torch.ones(1, 33, dtype=torch.bool), multiple=16
    )
    assert e48[0].size(1) == 48


@pytest.mark.parametrize("length", [1, 7, 13, 16, 30])
def test_every_method_is_unchanged(length: int) -> None:
    """Bitwise except log_likelihood's value, whose gold-path sum runs over the
    padded length and may round differently (measured at most 1.6 ulp)."""
    emissions, tags, mask, allowed, log_weights = _inputs(length, seed=length)
    e, m, t, a, w = pad_to_length_bucket(emissions, mask, tags, allowed, log_weights)
    torch.manual_seed(length)
    crf = LinearChainCRF(5, constraints=_constraints(), constrain_loss=True)
    for name, fn in (
        ("log_likelihood", lambda crf, e, m, t, a, w: crf.log_likelihood(e, t, m)),
        ("partial", lambda crf, e, m, t, a, w: crf.partial_log_likelihood(e, a, m, w)),
    ):
        x, xp = emissions.clone().requires_grad_(), e.clone().requires_grad_()
        v, vp = fn(crf, x, mask, tags, allowed, log_weights), fn(crf, xp, m, t, a, w)
        (g,) = torch.autograd.grad(v.sum(), x)
        (gp,) = torch.autograd.grad(vp.sum(), xp)
        if name == "log_likelihood":
            torch.testing.assert_close(vp, v, rtol=1e-6, atol=0.0)
        else:
            assert torch.equal(v, vp), name
        assert torch.equal(g, gp[:, :length]), name
    assert crf.decode(emissions, mask) == crf.decode(e, m)
    assert crf.decode_with_scores(emissions, mask, allowed) == crf.decode_with_scores(e, m, a)
    assert crf.decode_top_k(emissions, mask, 3) == crf.decode_top_k(e, m, 3)
    tags_out, scores = crf.decode_to_tensor(emissions, mask)
    tags_padded, scores_padded = crf.decode_to_tensor(e, m)
    assert torch.equal(tags_out, tags_padded[:, :length]) and torch.equal(scores, scores_padded)
    assert torch.equal(crf.marginals(emissions, mask), crf.marginals(e, m)[:, :length])


def test_parallel_scan_agrees_to_rounding() -> None:
    emissions, tags, mask, _, _ = _inputs(13, seed=3)
    e, m, t = pad_to_length_bucket(emissions, mask, tags)
    torch.manual_seed(3)
    crf = LinearChainCRF(5, parallel_scan=True)
    torch.testing.assert_close(
        crf.log_likelihood(e, t, m), crf.log_likelihood(emissions, tags, mask)
    )


@pytest.mark.parametrize(("bucketing", "graphs"), [({}, 2), ({"multiple": 16}, 3)])
def test_bucketing_bounds_the_number_of_compiled_graphs(bucketing: dict, graphs: int) -> None:
    """13 lengths from 26 to 62: one graph each unpadded (and eager past the
    recompile limit), 2 with power-of-two buckets, 3 with multiples of 16."""
    torch.manual_seed(4)
    crf = LinearChainCRF(5, validate=None)
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(crf.log_likelihood, backend=counter, dynamic=False)
    for length in range(26, 63, 3):
        emissions, tags, mask, _, _ = _inputs(length)
        e, m, t = pad_to_length_bucket(emissions, mask, tags, **bucketing)
        torch.testing.assert_close(
            compiled(e, t, m), crf.log_likelihood(emissions, tags, mask), rtol=1e-5, atol=1e-4
        )
    assert counter.frame_count == graphs


def test_errors() -> None:
    emissions, tags, mask, _, _ = _inputs(5)
    with pytest.raises(ValueError, match="multiple must be at least 1"):
        pad_to_length_bucket(emissions, mask, multiple=0)
    with pytest.raises(ValueError, match="min_length must be at least 1"):
        pad_to_length_bucket(emissions, mask, min_length=0)
    with pytest.raises(ValueError, match="dim 1 equal"):
        pad_to_length_bucket(emissions, mask[:, :4])
    with pytest.raises(ValueError, match="dim 1 equal"):
        pad_to_length_bucket(emissions, mask, tags[:, 0])


@pytest.mark.cuda
def test_no_host_sync_cuda() -> None:
    emissions, tags, mask, allowed, _ = (x.cuda() for x in _inputs(13))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        padded = pad_to_length_bucket(emissions, mask, tags, allowed)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert all(x.is_cuda and x.size(1) == 16 for x in padded)
