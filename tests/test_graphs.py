"""GraphedCRF: CUDA graph replays of the CRF's methods, bitwise equal to eager."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from torch_linear_crf import GraphedCRF, LinearChainCRF, pad_to_length_bucket

NUM_TAGS = 5
FEATURES = 7
# (batch, length) per training step: lengths in the 16, 32 and 64 buckets, and
# smaller last batches (5 and 3) next to the full batch of 8.
STEPS = [(8, 13), (8, 30), (5, 7), (8, 16), (5, 29), (3, 40)]
# Steps whose gradients add to the previous step's instead of replacing them.
ACCUMULATE = {3}


def _constraints() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transition_ok = torch.ones(NUM_TAGS, NUM_TAGS, dtype=torch.bool)
    transition_ok[0, 3] = transition_ok[4, 4] = False
    start_ok = torch.ones(NUM_TAGS, dtype=torch.bool)
    start_ok[4] = False
    return start_ok, transition_ok, torch.ones(NUM_TAGS, dtype=torch.bool)


def _batch(batch: int, length: int, seed: int, device: str) -> list[torch.Tensor]:
    """(features, tags, mask, allowed); every row valid at 0, the first row full length."""
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, length, FEATURES, generator=g)
    tags = torch.randint(0, NUM_TAGS, (batch, length), generator=g)
    tags[:, 0] = tags[:, 0].clamp(max=3)
    lengths = torch.randint(1, length + 1, (batch,), generator=g)
    lengths[0] = length
    mask = torch.arange(length) < lengths.unsqueeze(1)
    allowed = torch.rand(batch, length, NUM_TAGS, generator=g) > 0.5
    # Tag 0 is valid everywhere under _constraints, so an allowed path always exists.
    allowed[..., 0] = True
    return [x.to(device) for x in (features, tags, mask, allowed)]


class _Tagger(nn.Module):
    def __init__(self, **crf_kwargs) -> None:
        super().__init__()
        self.emit = nn.Linear(FEATURES, NUM_TAGS)
        self.weigh = nn.Linear(FEATURES, NUM_TAGS)
        self.crf = LinearChainCRF(NUM_TAGS, **crf_kwargs)


CONFIGS = {
    "plain": ({}, None),
    "constrained": ({"constraints": _constraints(), "constrain_loss": True}, None),
    "decode_constraints_only": ({"constraints": _constraints()}, None),
    "parallel_scan": (
        {"constraints": _constraints(), "constrain_loss": True, "parallel_scan": True},
        None,
    ),
    "validate_none": ({"validate": None}, None),
    "no_start_end": ({"include_start_end_transitions": False}, None),
    "bf16_autocast": ({"constraints": _constraints(), "constrain_loss": True}, torch.bfloat16),
    "fp16_autocast": ({}, torch.float16),
}


def _step(model, method, features, tags, mask, allowed, autocast, pad):
    """One step's results: likelihood values, emission and parameter gradients, decodes.

    `method` is the CRF or a GraphedCRF over it; `pad` pads the inputs to
    `pad_to_length_bucket`'s buckets first, for the eager reference.
    """
    if pad:

        def prepare(*xs):
            return pad_to_length_bucket(*xs)
    else:

        def prepare(*xs):
            return xs

    out = {}
    with torch.autocast("cuda", dtype=autocast or torch.float32, enabled=autocast is not None):
        emissions = model.emit(features)
        emissions.retain_grad()
        e, m, t = prepare(emissions, mask, tags)
        out["ll"] = method.log_likelihood(e, t, m)
    (-out["ll"].sum() / mask.sum()).backward()
    out["ll_emission_grad"] = emissions.grad

    with torch.autocast("cuda", dtype=autocast or torch.float32, enabled=autocast is not None):
        emissions = model.emit(features)
        emissions.retain_grad()
        log_weights = model.weigh(features).log_softmax(-1)
        e, m, a, w = prepare(emissions, mask, allowed, log_weights)
        out["pll"] = method.partial_log_likelihood(e, a, m, w)
    (-out["pll"].sum()).backward()
    out["pll_emission_grad"] = emissions.grad
    out["param_grads"] = [p.grad.clone() for p in model.parameters()]

    with torch.no_grad():
        emissions = model.emit(features)
        e, m, a = prepare(emissions, mask, allowed)
        length = features.size(1)
        paths, scores = method.decode_to_tensor(e, m)
        out["decode_to_tensor"] = (paths[:, :length], scores)
        paths, scores = method.decode_to_tensor(e, m, a, pad_tag=7)
        out["decode_to_tensor_allowed"] = (paths[:, :length], scores)
        out["decode"] = method.decode(e, m)
    return out


def _assert_bitwise(expected, actual, where: str) -> None:
    if isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape, where
        assert torch.equal(expected, actual), where
    elif isinstance(expected, (list, tuple)) and expected and isinstance(expected[0], torch.Tensor):
        assert len(expected) == len(actual), where
        for i, (x, y) in enumerate(zip(expected, actual, strict=True)):
            _assert_bitwise(x, y, f"{where}[{i}]")
    else:
        assert expected == actual, where


def _train_both(crf_kwargs, autocast, graphed_kwargs, pad):
    """Train an eager and a graphed copy side by side, asserting bitwise equality each step."""
    torch.manual_seed(0)
    eager_model = _Tagger(**crf_kwargs).cuda()
    graphed_model = copy.deepcopy(eager_model)
    graphed = GraphedCRF(graphed_model.crf, **graphed_kwargs)
    eager_opt = torch.optim.Adam(eager_model.parameters(), lr=0.05)
    graphed_opt = torch.optim.Adam(graphed_model.parameters(), lr=0.05)
    for i, (batch, length) in enumerate(STEPS):
        inputs = _batch(batch, length, seed=i, device="cuda")
        if i not in ACCUMULATE:
            eager_opt.zero_grad()
            graphed_opt.zero_grad()
        expected = _step(eager_model, eager_model.crf, *inputs, autocast, pad=pad)
        actual = _step(graphed_model, graphed, *inputs, autocast, pad=False)
        for name in expected:
            _assert_bitwise(expected[name], actual[name], f"step {i}: {name}")
        eager_opt.step()
        graphed_opt.step()
    for p, q in zip(eager_model.parameters(), graphed_model.parameters(), strict=True):
        assert torch.equal(p, q)
    return graphed


@pytest.mark.cuda
@pytest.mark.parametrize("config", list(CONFIGS))
def test_training_loop_is_bitwise_eager_on_bucketed_inputs_cuda(config: str) -> None:
    """Fresh inputs and an Adam step each time, against eager on the same padded inputs."""
    crf_kwargs, autocast = CONFIGS[config]
    graphed = _train_both(crf_kwargs, autocast, {}, pad=True)
    # (batch, bucket) pairs: (8, 16), (8, 32), (5, 16), (5, 32), (3, 64); the
    # repeated (8, 16) reuses its graphs. Times ll, pll and two decode signatures.
    assert graphed.num_graphs == 5 * 4


@pytest.mark.cuda
@pytest.mark.parametrize("config", ["plain", "parallel_scan", "bf16_autocast"])
def test_exact_lengths_are_bitwise_eager_unpadded_cuda(config: str) -> None:
    crf_kwargs, autocast = CONFIGS[config]
    graphed = _train_both(crf_kwargs, autocast, {"min_length": 1, "multiple": 1}, pad=False)
    assert graphed.num_graphs == len(STEPS) * 4


@pytest.mark.cuda
def test_default_buckets_match_unpadded_eager_cuda() -> None:
    """Every gradient and decode bitwise; log_likelihood's value to rounding."""
    torch.manual_seed(1)
    crf = LinearChainCRF(NUM_TAGS, constraints=_constraints()).cuda()
    graphed = GraphedCRF(crf)
    _, tags, mask, _ = _batch(6, 21, seed=5, device="cuda")
    emissions = torch.randn(6, 21, NUM_TAGS, device="cuda") * 3
    x, y = emissions.clone().requires_grad_(), emissions.clone().requires_grad_()
    expected = crf.log_likelihood(x, tags, mask)
    expected_grads = torch.autograd.grad(expected.sum(), [x, *crf.parameters()])
    actual = graphed.log_likelihood(y, tags, mask)
    actual_grads = torch.autograd.grad(actual.sum(), [y, *crf.parameters()])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0.0)
    for g, h in zip(expected_grads, actual_grads, strict=True):
        assert torch.equal(g, h)
    assert crf.decode(emissions, mask) == graphed.decode(emissions, mask)


@pytest.mark.cuda
def test_no_host_sync_on_replay_cuda() -> None:
    torch.manual_seed(2)
    model = _Tagger(constraints=_constraints(), constrain_loss=True).cuda()
    graphed = GraphedCRF(model.crf)
    features, tags, mask, allowed = _batch(8, 13, seed=0, device="cuda")

    def step():
        emissions = model.emit(features)
        (-graphed.log_likelihood(emissions, tags, mask).sum()).backward()
        emissions = model.emit(features)
        (-graphed.partial_log_likelihood(emissions, allowed, mask).sum()).backward()
        with torch.no_grad():
            return graphed.decode_to_tensor(model.emit(features), mask)

    step()  # captures, which synchronizes
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.cuda
def test_another_replay_before_backward_raises_cuda() -> None:
    torch.manual_seed(3)
    crf = LinearChainCRF(NUM_TAGS).cuda()
    graphed = GraphedCRF(crf)
    _, tags, mask, _ = _batch(4, 9, seed=0, device="cuda")
    emissions = torch.randn(4, 9, NUM_TAGS, device="cuda", requires_grad=True)
    first = graphed.log_likelihood(emissions, tags, mask)
    second = graphed.log_likelihood(emissions, tags, mask)
    second.sum().backward()  # the latest call's backward is fine
    with pytest.raises(RuntimeError, match="another graphed call ran"):
        first.sum().backward()
    loss = graphed.log_likelihood(emissions, tags, mask)
    graphed.decode_to_tensor(emissions.detach(), mask)
    with pytest.raises(RuntimeError, match="another graphed call ran"):
        loss.sum().backward()


@pytest.mark.cuda
def test_no_grad_and_frozen_parameters_cuda() -> None:
    torch.manual_seed(4)
    crf = LinearChainCRF(NUM_TAGS).cuda()
    graphed = GraphedCRF(crf)
    _, tags, mask, _ = _batch(4, 9, seed=0, device="cuda")
    emissions = torch.randn(4, 9, NUM_TAGS, device="cuda")
    e, m, t = pad_to_length_bucket(emissions, mask, tags)
    with torch.no_grad():
        assert torch.equal(
            graphed.log_likelihood(emissions, tags, mask), crf.log_likelihood(e, t, m)
        )
    crf.requires_grad_(False)
    x, y = emissions.clone().requires_grad_(), e.clone().requires_grad_()
    (gx,) = torch.autograd.grad(graphed.log_likelihood(x, tags, mask).sum(), x)
    (gy,) = torch.autograd.grad(crf.log_likelihood(y, t, m).sum(), y)
    assert torch.equal(gx, gy[:, :9])
    assert graphed.num_graphs == 2


@pytest.mark.cuda
def test_changing_the_crf_drops_its_graphs_cuda() -> None:
    torch.manual_seed(5)
    crf = LinearChainCRF(NUM_TAGS, constraints=_constraints()).cuda()
    graphed = GraphedCRF(crf, min_length=1, multiple=1)
    _, tags, mask, _ = _batch(4, 9, seed=0, device="cuda")
    emissions = torch.randn(4, 9, NUM_TAGS, device="cuda")
    with torch.no_grad():
        for change in (
            lambda: None,
            lambda: setattr(crf, "constrain_loss", True),
            lambda: setattr(crf, "parallel_scan", True),
            crf.double,
        ):
            change()
            assert torch.equal(
                graphed.log_likelihood(emissions, tags, mask),
                crf.log_likelihood(emissions, tags, mask),
            )
            assert graphed.num_graphs == 1
    graphed.reset()
    assert graphed.num_graphs == 0


@pytest.mark.cuda
def test_sync_validation_is_rejected_cuda() -> None:
    crf = LinearChainCRF(NUM_TAGS, validate="sync").cuda()
    _, tags, mask, _ = _batch(4, 9, seed=0, device="cuda")
    with pytest.raises(ValueError, match="cannot capture"):
        GraphedCRF(crf).log_likelihood(torch.randn(4, 9, NUM_TAGS, device="cuda"), tags, mask)


def test_runs_the_crf_itself_off_cuda() -> None:
    """On CPU every method is the CRF's own call: same values, gradients, errors, no graphs."""
    torch.manual_seed(6)
    crf = LinearChainCRF(NUM_TAGS, constraints=_constraints(), constrain_loss=True)
    graphed = GraphedCRF(crf)
    _, tags, mask, allowed = _batch(4, 11, seed=0, device="cpu")
    emissions = torch.randn(4, 11, NUM_TAGS) * 3
    for method, args in (
        ("log_likelihood", (tags, mask)),
        ("partial_log_likelihood", (allowed, mask)),
    ):
        x, y = emissions.clone().requires_grad_(), emissions.clone().requires_grad_()
        expected = getattr(crf, method)(x, *args)
        expected_grads = torch.autograd.grad(expected.sum(), [x, *crf.parameters()])
        actual = getattr(graphed, method)(y, *args)
        actual_grads = torch.autograd.grad(actual.sum(), [y, *crf.parameters()])
        assert torch.equal(expected, actual), method
        for g, h in zip(expected_grads, actual_grads, strict=True):
            assert torch.equal(g, h), method
    _assert_bitwise(
        crf.decode_to_tensor(emissions, mask), graphed.decode_to_tensor(emissions, mask), "dtt"
    )
    assert crf.decode(emissions, mask, allowed) == graphed.decode(emissions, mask, allowed)
    with pytest.raises(ValueError, match="tags must have shape"):
        graphed.log_likelihood(emissions, tags[:, :5], mask)
    assert graphed.num_graphs == 0


def test_bucket_arguments_are_checked() -> None:
    crf = LinearChainCRF(NUM_TAGS)
    with pytest.raises(ValueError, match="min_length must be at least 1"):
        GraphedCRF(crf, min_length=0)
    with pytest.raises(ValueError, match="multiple must be at least 1"):
        GraphedCRF(crf, multiple=0)
