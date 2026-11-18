"""The opt-in parallel_scan log partition, against the loop it replaces."""

from __future__ import annotations

import pytest
import torch

from torch_linear_crf import FORBIDDEN, LinearChainCRF


def _pair(num_tags: int, constraints=None, constrain_loss: bool = False, dtype=torch.float64):
    """A loop CRF and a scan CRF with the same parameters."""
    loop = LinearChainCRF(num_tags, constraints=constraints, constrain_loss=constrain_loss)
    scan = LinearChainCRF(
        num_tags, constraints=constraints, constrain_loss=constrain_loss, parallel_scan=True
    )
    with torch.no_grad():
        for p in loop.parameters():
            p.normal_()
    scan.load_state_dict(loop.state_dict())
    return loop.to(dtype), scan.to(dtype)


def _ragged(batch: int, length: int, num_tags: int, seed: int, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, length + 1, (batch,), generator=g)
    lengths[0] = length
    mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
    emissions = torch.randn(batch, length, num_tags, generator=g, dtype=dtype) * 3
    tags = torch.randint(0, num_tags, (batch, length), generator=g)
    return emissions, tags, mask


def _constraints(num_tags: int):
    start_ok = torch.ones(num_tags, dtype=torch.bool)
    start_ok[-1] = False
    transition_ok = torch.rand(num_tags, num_tags, generator=torch.Generator().manual_seed(1)) > 0.3
    transition_ok[0, 0] = True
    end_ok = torch.ones(num_tags, dtype=torch.bool)
    end_ok[1] = False
    return start_ok, transition_ok, end_ok


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31])
@pytest.mark.parametrize("constrained", [False, True])
def test_matches_the_loop_in_fp64(length: int, constrained: bool) -> None:
    """Every length parity and power-of-two boundary, ragged rows, both likelihoods."""
    num_tags = 4
    constraints = _constraints(num_tags) if constrained else None
    loop, scan = _pair(num_tags, constraints, constrain_loss=constrained)
    emissions, tags, mask = _ragged(5, length, num_tags, seed=length)
    allowed = torch.rand(5, length, num_tags, generator=torch.Generator().manual_seed(2)) > 0.4
    allowed[..., 0] = True
    for fn in (
        lambda crf, e: crf.log_likelihood(e, tags, mask),
        lambda crf, e: crf.partial_log_likelihood(e, allowed, mask),
    ):
        x_loop = emissions.clone().requires_grad_()
        x_scan = emissions.clone().requires_grad_()
        v_loop, v_scan = fn(loop, x_loop), fn(scan, x_scan)
        v_loop.sum().backward()
        v_scan.sum().backward()
        torch.testing.assert_close(v_scan, v_loop, rtol=1e-12, atol=1e-10)
        torch.testing.assert_close(x_scan.grad, x_loop.grad, rtol=1e-12, atol=1e-10)
        for a, b in zip(scan.parameters(), loop.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-9)
            a.grad = b.grad = None


def test_gradcheck() -> None:
    _, scan = _pair(3, _constraints(3), constrain_loss=True)
    emissions, tags, mask = _ragged(3, 6, 3, seed=5)
    x = emissions.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda e: scan.log_likelihood(e, tags, mask), (x,))


def test_every_path_forbidden_and_neg_inf_inputs_stay_finite() -> None:
    """The cases the finite FORBIDDEN exists for: a row whose every path crosses a
    forbidden transition, a tag masked with -inf, and -inf padding."""
    num_tags = 3
    transition_ok = torch.eye(num_tags, dtype=torch.bool).logical_not()  # no repeats
    ok = torch.ones(num_tags, dtype=torch.bool)
    loop, scan = _pair(num_tags, (ok, transition_ok, ok), constrain_loss=True)
    emissions, _, mask = _ragged(3, 6, num_tags, seed=7)
    emissions[:, :, 1] = float("-inf")  # only tags 0 and 2, so some rows must repeat
    emissions[~mask] = float("-inf")
    tags = torch.zeros(3, 6, dtype=torch.long)  # 0 -> 0: forbidden at every step
    x_loop = emissions.clone().requires_grad_()
    x_scan = emissions.clone().requires_grad_()
    v_loop = loop.log_likelihood(x_loop, tags, mask)
    v_scan = scan.log_likelihood(x_scan, tags, mask)
    v_loop.sum().backward()
    v_scan.sum().backward()
    assert torch.isfinite(v_scan).all() and torch.isfinite(x_scan.grad).all()
    assert (v_scan[0] < FORBIDDEN / 2).item()
    torch.testing.assert_close(v_scan, v_loop, rtol=1e-12, atol=1e-8)
    torch.testing.assert_close(x_scan.grad, x_loop.grad, rtol=1e-12, atol=1e-8)


def test_fp32_accuracy_is_comparable_to_the_loop() -> None:
    """Against an fp64 loop as ground truth, the fp32 scan's error is the same
    order as the fp32 loop's (different summation order, not lower precision)."""
    num_tags, length = 9, 256
    loop64, _ = _pair(num_tags)
    emissions, tags, mask = _ragged(16, length, num_tags, seed=11)
    truth = loop64.log_likelihood(emissions, tags, mask)
    loop32, scan32 = _pair(num_tags, dtype=torch.float32)
    loop32.load_state_dict(loop64.state_dict())
    scan32.load_state_dict(loop64.state_dict())
    e32 = emissions.float()
    err_loop = (loop32.log_likelihood(e32, tags, mask).double() - truth).abs().max()
    err_scan = (scan32.log_likelihood(e32, tags, mask).double() - truth).abs().max()
    assert err_scan <= 10 * err_loop + 1e-4, (err_scan, err_loop)


def test_dtypes_autocast_scripting_compile() -> None:
    _, scan = _pair(5, dtype=torch.float32)
    emissions, tags, mask = _ragged(4, 9, 5, seed=13, dtype=torch.float32)
    expected = scan.log_likelihood(emissions, tags, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert scan.log_likelihood(emissions.bfloat16(), tags, mask).dtype is torch.float32
    assert scan.double().log_likelihood(emissions.double(), tags, mask).dtype is torch.float64
    scan.float()
    scripted = torch.jit.script(scan)
    torch.testing.assert_close(scripted.log_likelihood(emissions, tags, mask), expected)
    torch._dynamo.reset()
    compiled = torch.compile(scan.log_likelihood)
    torch.testing.assert_close(compiled(emissions, tags, mask), expected, rtol=1e-5, atol=1e-4)
    assert not LinearChainCRF(5).parallel_scan


@pytest.mark.cuda
def test_matches_the_loop_on_cuda() -> None:
    loop, scan = _pair(9, _constraints(9), constrain_loss=True, dtype=torch.float32)
    loop, scan = loop.cuda(), scan.cuda()
    emissions, tags, mask = _ragged(16, 64, 9, seed=17, dtype=torch.float32)
    emissions, tags, mask = emissions.cuda(), tags.cuda(), mask.cuda()
    x_loop = emissions.clone().requires_grad_()
    x_scan = emissions.clone().requires_grad_()
    v_loop = loop.log_likelihood(x_loop, tags, mask)
    v_scan = scan.log_likelihood(x_scan, tags, mask)
    v_loop.sum().backward()
    v_scan.sum().backward()
    torch.testing.assert_close(v_scan, v_loop, rtol=1e-5, atol=2e-3)
    torch.testing.assert_close(x_scan.grad, x_loop.grad, rtol=1e-4, atol=1e-4)
