"""Tests for the core CRF.

The central checks are the brute-force ones: `_log_partition` and
`decode` are compared against explicit enumeration of every tag sequence on
instances small enough to enumerate, which is the only way to catch a subtly
wrong recursion.
"""

from __future__ import annotations

import io
import itertools
import warnings

import pytest
import torch
import torch.autograd.forward_ad as fwad

from torch_linear_crf import FORBIDDEN, LinearChainCRF


def _enumerate_log_partition(crf: LinearChainCRF, emissions, length: int) -> torch.Tensor:
    """logsumexp over every tag sequence of the given length, scored directly."""
    start, transitions, end = crf._scores(crf.constrain_loss)
    totals = []
    for path in itertools.product(range(crf.num_tags), repeat=length):
        score = start[path[0]] + emissions[0, path[0]]
        for t in range(1, length):
            score = score + transitions[path[t - 1], path[t]] + emissions[t, path[t]]
        totals.append(score + end[path[-1]])
    return torch.logsumexp(torch.stack(totals), 0)


def test_log_partition_matches_enumeration() -> None:
    torch.manual_seed(1)
    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 4, 3) * 2.0
    mask = torch.tensor([[True] * 4, [True, True, True, False]])
    got = crf._log_partition(emissions, mask, *crf._scores(False), False)
    for row, length in enumerate([4, 3]):
        expected = _enumerate_log_partition(crf, emissions[row], length)
        assert torch.allclose(got[row], expected, atol=1e-5), (row, got[row], expected)


def test_decode_matches_enumeration() -> None:
    """Viterbi must find the single highest-scoring path."""
    torch.manual_seed(2)
    crf = LinearChainCRF(3)
    with torch.no_grad():
        crf.transitions.uniform_(-2, 2)
    emissions = torch.randn(1, 5, 3) * 3.0
    mask = torch.ones(1, 5, dtype=torch.bool)
    start, transitions, end = crf._scores(True)

    best_path, best_score = None, None
    for path in itertools.product(range(3), repeat=5):
        score = start[path[0]] + emissions[0, 0, path[0]]
        for t in range(1, 5):
            score = score + transitions[path[t - 1], path[t]] + emissions[0, t, path[t]]
        score = score + end[path[-1]]
        if best_score is None or score > best_score:
            best_path, best_score = list(path), score
    assert crf.decode(emissions, mask)[0] == best_path


def test_log_likelihood_is_a_log_probability() -> None:
    """Summing p(tags) over every sequence must give 1."""
    torch.manual_seed(3)
    crf = LinearChainCRF(3)
    emissions = torch.randn(1, 4, 3) * 2.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    total = sum(
        crf.log_likelihood(emissions, torch.tensor([list(path)]), mask).exp().item()
        for path in itertools.product(range(3), repeat=4)
    )
    assert total == pytest.approx(1.0, abs=1e-5)


def test_constraints_restrict_the_decode() -> None:
    """With an explicit mask, decode never crosses a forbidden transition."""
    torch.manual_seed(5)
    # Tag 0 -> tag 1 is forbidden; everything else is allowed.
    start_ok = torch.ones(3, dtype=torch.bool)
    end_ok = torch.ones(3, dtype=torch.bool)
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 1] = False
    constrained = LinearChainCRF(3, constraints=(start_ok, transition_ok, end_ok))
    with torch.no_grad():
        # Push hard toward 0 -> 1 so an unconstrained model would take it.
        constrained.transitions[0, 1] = 10.0

    emissions = torch.randn(32, 6, 3) * 4.0
    mask = torch.ones(32, 6, dtype=torch.bool)
    for path in constrained.decode(emissions, mask):
        assert not any(a == 0 and b == 1 for a, b in itertools.pairwise(path))


def test_constrained_decode_with_a_mask_matches_enumeration() -> None:
    """Constrained decode with padding must find each row's best *valid* path
    over that row's own length -- constraints and masking interact only here,
    and every other decode test is either unmasked or unconstrained."""
    torch.manual_seed(11)
    # Tag 0 -> tag 1 and tag 2 -> tag 2 forbidden; tag 2 may not start a row.
    start_ok = torch.tensor([True, True, False])
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 1] = False
    transition_ok[2, 2] = False
    end_ok = torch.ones(3, dtype=torch.bool)
    crf = LinearChainCRF(3, constraints=(start_ok, transition_ok, end_ok))
    with torch.no_grad():
        crf.transitions.uniform_(-2, 2)
        # Push toward the forbidden pairs so an unconstrained decode would take them.
        crf.transitions[0, 1] = 8.0
        crf.transitions[2, 2] = 8.0

    lengths = [5, 3, 1]
    emissions = torch.randn(3, 5, 3) * 3.0
    mask = torch.arange(5).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    start, transitions, end = crf._scores(True)

    decoded = crf.decode(emissions, mask)
    for row, length in enumerate(lengths):
        best_path, best_score = None, None
        for path in itertools.product(range(3), repeat=length):
            score = start[path[0]] + emissions[row, 0, path[0]]
            for t in range(1, length):
                score = score + transitions[path[t - 1], path[t]] + emissions[row, t, path[t]]
            score = score + end[path[-1]]
            if best_score is None or score > best_score:
                best_path, best_score = list(path), score
        assert decoded[row] == best_path, (row, decoded[row], best_path)


def test_masked_positions_do_not_change_the_score() -> None:
    """Padding past a row's length must not affect its log-likelihood."""
    torch.manual_seed(6)
    crf = LinearChainCRF(4)
    emissions = torch.randn(1, 3, 4)
    tags = torch.tensor([[1, 2, 0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    short = crf.log_likelihood(emissions, tags, mask)

    padded_emissions = torch.cat([emissions, torch.randn(1, 2, 4) * 50], dim=1)
    padded_tags = torch.cat([tags, torch.zeros(1, 2, dtype=torch.long)], dim=1)
    padded_mask = torch.tensor([[True, True, True, False, False]])
    assert torch.allclose(short, crf.log_likelihood(padded_emissions, padded_tags, padded_mask))


def test_empty_row_raises() -> None:
    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 2, 3)
    tags = torch.zeros(2, 2, dtype=torch.long)
    mask = torch.tensor([[True, True], [False, False]])
    with pytest.raises(ValueError, match="at least one labeled position"):
        crf.log_likelihood(emissions, tags, mask)


def test_computes_in_fp32_under_bf16_autocast() -> None:
    """bf16's mantissa is too short for the recursion; the guard must hold."""
    torch.manual_seed(7)
    num_tags = 9
    crf = LinearChainCRF(num_tags)
    linear = torch.nn.Linear(32, num_tags)
    hidden = torch.randn(4, 16, 32)
    tags = torch.randint(0, num_tags, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        emissions = linear(hidden)
        assert emissions.dtype is torch.bfloat16, "expected autocast to hand the CRF bf16 logits"
        result = crf.log_likelihood(emissions, tags, mask)
        decoded = crf.decode(emissions[::2], mask[::2])
    assert result.dtype is torch.float32
    assert len(decoded) == 2


def test_autocast_is_disabled_inside_the_recursion() -> None:
    """The bf16 test above still passes with the guard deleted -- the explicit
    fp32 cast is what fixes the numbers -- so check the guard itself: autocast
    must be off by the time the recursion runs, for both entry points."""
    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 4, 3)
    tags = torch.randint(0, 3, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)

    seen: list[bool] = []
    inner_log_likelihood = crf._log_likelihood_impl
    inner_viterbi = crf._viterbi_impl

    def spy_log_likelihood(*args: torch.Tensor) -> torch.Tensor:
        seen.append(torch.is_autocast_enabled("cpu"))
        return inner_log_likelihood(*args)

    def spy_viterbi(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seen.append(torch.is_autocast_enabled("cpu"))
        return inner_viterbi(*args)

    crf._log_likelihood_impl = spy_log_likelihood
    crf._viterbi_impl = spy_viterbi
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu"), "expected the test's own autocast to be on"
        crf.log_likelihood(emissions, tags, mask)
        crf.decode(emissions, mask)
    assert seen == [False, False]


def test_autocast_guard_follows_the_tensor_device() -> None:
    """Outside TorchScript the guard must disable autocast for whatever device
    the emissions are on (XPU/MPS/HPU included), not send everything but CUDA
    down the "cpu" branch -- and must not ask for a device with no autocast."""
    from torch_linear_crf.crf import _autocast_device_type

    for device_type in ["cpu", "cuda", "xpu", "mps", "hpu"]:
        # mps autocast arrived after torch 2.4, the declared floor.
        expected = device_type if torch.amp.is_autocast_available(device_type) else "cpu"
        assert _autocast_device_type(torch.device(device_type)) == expected
    # A device with no autocast at all must not be passed to torch.autocast.
    assert _autocast_device_type(torch.device("meta")) == "cpu"
    assert not torch.amp.is_autocast_available("meta")


@pytest.mark.cuda
def test_computes_in_fp32_under_bf16_autocast_cuda() -> None:
    """CUDA replay of test_computes_in_fp32_under_bf16_autocast. This is the
    device the guard exists for: logsumexp is fp32-listed for CUDA autocast
    but not CPU, so the CPU-only version of this test can pass even with the
    guard deleted (see test_autocast_is_disabled_inside_the_recursion)."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    num_tags = 9
    crf = LinearChainCRF(num_tags).to(device)
    linear = torch.nn.Linear(32, num_tags).to(device)
    hidden = torch.randn(4, 16, 32, device=device)
    tags = torch.randint(0, num_tags, (4, 16), device=device)
    mask = torch.ones(4, 16, dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        emissions = linear(hidden)
        assert emissions.dtype is torch.bfloat16, "expected autocast to hand the CRF bf16 logits"
        result = crf.log_likelihood(emissions, tags, mask)
        decoded = crf.decode(emissions[::2], mask[::2])
    assert result.dtype is torch.float32
    assert result.device.type == "cuda"
    assert len(decoded) == 2


@pytest.mark.cuda
def test_autocast_guard_uses_cuda_branch_on_cuda() -> None:
    """The is_cuda branch in log_likelihood/_viterbi must genuinely select
    device_type="cuda" on a CUDA tensor rather than silently falling through
    to "cpu" -- a CPU-only test suite can't tell these branches apart since
    the cpu and cuda autocast flags are independent either way."""
    device = torch.device("cuda")
    crf = LinearChainCRF(3).to(device)
    emissions = torch.randn(2, 4, 3, device=device)
    tags = torch.randint(0, 3, (2, 4), device=device)
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)

    seen: list[tuple[bool, bool]] = []
    inner_log_likelihood = crf._log_likelihood_impl
    inner_viterbi = crf._viterbi_impl

    def spy_log_likelihood(*args: torch.Tensor) -> torch.Tensor:
        seen.append((torch.is_autocast_enabled("cpu"), torch.is_autocast_enabled("cuda")))
        return inner_log_likelihood(*args)

    def spy_viterbi(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seen.append((torch.is_autocast_enabled("cpu"), torch.is_autocast_enabled("cuda")))
        return inner_viterbi(*args)

    crf._log_likelihood_impl = spy_log_likelihood
    crf._viterbi_impl = spy_viterbi
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cuda"), "expected the test's own autocast to be on"
        crf.log_likelihood(emissions, tags, mask)
        crf.decode(emissions, mask)
    # Both flags False inside the recursion both times: cuda autocast (the
    # one actually on) was genuinely disabled, and cpu autocast was never on.
    assert seen == [(False, False), (False, False)]


@pytest.mark.cuda
def test_scripts_and_matches_eager_cuda() -> None:
    """CUDA replay of test_scripts_and_matches_eager -- exercises the scripted
    module's literal device_type="cuda" autocast branch, which TorchScript
    requires as a compile-time constant and the CPU test suite can't reach."""
    device = torch.device("cuda")
    torch.manual_seed(8)
    crf = LinearChainCRF(5).to(device)
    scripted = torch.jit.script(crf)

    emissions = torch.randn(3, 6, 5, device=device)
    tags = torch.randint(0, 5, (3, 6), device=device)
    mask = torch.ones(3, 6, dtype=torch.bool, device=device)
    mask[1, 4:] = False
    mask[2, 3:] = False

    assert torch.allclose(
        crf.log_likelihood(emissions, tags, mask),
        scripted.log_likelihood(emissions, tags, mask),
    )
    assert crf.decode(emissions, mask) == scripted.decode(emissions, mask)


def test_forbidden_penalty_is_finite() -> None:
    """-inf would make a fully-constrained row produce nan rather than a score."""
    assert FORBIDDEN < -1e3 and torch.isfinite(torch.tensor(FORBIDDEN))


def test_scripts_and_matches_eager() -> None:
    """torch.jit.script must compile log_likelihood/decode/decode_with_scores
    and produce output identical to eager mode."""
    torch.manual_seed(8)
    crf = LinearChainCRF(5)
    scripted = torch.jit.script(crf)

    emissions = torch.randn(3, 6, 5)
    tags = torch.randint(0, 5, (3, 6))
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[1, 4:] = False
    mask[2, 3:] = False

    assert torch.allclose(
        crf.log_likelihood(emissions, tags, mask),
        scripted.log_likelihood(emissions, tags, mask),
    )
    assert crf.decode(emissions, mask) == scripted.decode(emissions, mask)
    assert crf.decode_with_scores(emissions, mask) == scripted.decode_with_scores(emissions, mask)


def test_compiles_with_torch_compile() -> None:
    """log_likelihood/decode must both compile and produce output identical to eager mode."""
    torch.manual_seed(10)
    crf = LinearChainCRF(5)
    emissions = torch.randn(3, 6, 5)
    tags = torch.randint(0, 5, (3, 6))
    mask = torch.ones(3, 6, dtype=torch.bool)

    compiled_ll = torch.compile(crf.log_likelihood)
    assert torch.allclose(
        compiled_ll(emissions, tags, mask), crf.log_likelihood(emissions, tags, mask)
    )
    # The gradient too: on torch 2.4-2.12, inductor dropped the gold-path
    # term of the emission gradient when _sequence_score used gather.
    eager_input = emissions.clone().requires_grad_()
    crf.log_likelihood(eager_input, tags, mask).sum().backward()
    compiled_input = emissions.clone().requires_grad_()
    compiled_ll(compiled_input, tags, mask).sum().backward()
    torch.testing.assert_close(compiled_input.grad, eager_input.grad)

    compiled_decode = torch.compile(crf.decode)
    assert compiled_decode(emissions, mask) == crf.decode(emissions, mask)

    # Matching eager output alone cannot tell compilation from a silent no-op,
    # so check Dynamo actually captured the recursion, and that the only graph
    # breaks are the two host reads in input validation on CPU (the
    # mask-prefix check and the tag-range check) -- more is a regression.
    explained = torch._dynamo.explain(crf.log_likelihood)(emissions, tags, mask)
    assert explained.op_count > 0
    assert explained.graph_break_count <= 2, explained.break_reasons


def test_decode_with_scores_matches_decode_and_log_likelihood() -> None:
    """The path must match plain decode(); the score must match the
    log-likelihood's own numerator (the unnormalized score of that path)."""
    torch.manual_seed(9)
    crf = LinearChainCRF(4)
    emissions = torch.randn(5, 7, 4) * 2.0
    mask = torch.ones(5, 7, dtype=torch.bool)

    plain = crf.decode(emissions, mask)
    with_scores = crf.decode_with_scores(emissions, mask)
    assert [path for path, _ in with_scores] == plain

    start, transitions, end = crf._scores(True)
    for (path, score), row_emissions in zip(with_scores, emissions, strict=True):
        tags = torch.tensor([path])
        expected = crf._sequence_score(
            row_emissions.unsqueeze(0),
            tags,
            torch.ones_like(tags, dtype=torch.bool),
            start,
            transitions,
            end,
        )
        assert torch.allclose(torch.tensor(score), expected[0], atol=1e-4)


def test_include_start_end_transitions_false_is_a_fixed_zero() -> None:
    """Disabled start/end transitions must not be learned parameters, and
    must contribute nothing to the score -- equivalent to not having them."""
    crf = LinearChainCRF(4, include_start_end_transitions=False)
    assert not isinstance(crf.start_transitions, torch.nn.Parameter)
    assert not isinstance(crf.end_transitions, torch.nn.Parameter)
    assert torch.all(crf.start_transitions == 0)
    assert torch.all(crf.end_transitions == 0)
    assert {name for name, _ in crf.named_parameters()} == {"transitions"}

    start, _, end = crf._scores(False)
    assert torch.all(start == 0) and torch.all(end == 0)

    # Disabling the parameters must give exactly what keeping them and zeroing
    # them gives: same transitions, same emissions, same log-likelihood.
    with_start_end = LinearChainCRF(4, include_start_end_transitions=True)
    with torch.no_grad():
        crf.transitions.copy_(with_start_end.transitions)
        with_start_end.start_transitions.zero_()
        with_start_end.end_transitions.zero_()
    emissions = torch.randn(2, 5, 4)
    tags = torch.randint(0, 4, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)

    ll = crf.log_likelihood(emissions, tags, mask)
    assert ll.shape == (2,)
    assert torch.allclose(ll, with_start_end.log_likelihood(emissions, tags, mask))
    assert crf.decode(emissions, mask) == with_start_end.decode(emissions, mask)


def test_validation_rejects_shape_mismatches() -> None:
    crf = LinearChainCRF(4)
    with pytest.raises(ValueError, match="3 dimensions"):
        crf.decode(torch.randn(2, 5), torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="num_tags=4"):
        crf.log_likelihood(
            torch.randn(2, 5, 3),
            torch.zeros(2, 5, dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="agree on"):
        crf.decode(torch.randn(2, 5, 4), torch.ones(3, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="tags must have shape"):
        crf.log_likelihood(
            torch.randn(2, 5, 4),
            torch.zeros(2, 6, dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        )


def test_validation_rejects_out_of_range_tags() -> None:
    """An ignore-index convention like -100 must raise clearly, not crash
    inside .gather() with an opaque index-out-of-range error."""
    crf = LinearChainCRF(3)
    emissions = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid tag ids"):
        crf.log_likelihood(emissions, torch.tensor([[0, -100]]), mask)
    with pytest.raises(ValueError, match="valid tag ids"):
        crf.log_likelihood(emissions, torch.tensor([[0, 3]]), mask)


def test_validation_rejects_mask_not_starting_valid() -> None:
    """decode/decode_with_scores must reject the same malformed masks log_likelihood does."""
    crf = LinearChainCRF(3)
    emissions = torch.randn(1, 3, 3)
    mask = torch.tensor([[False, True, True]])
    with pytest.raises(ValueError, match="at least one labeled position"):
        crf.decode(emissions, mask)
    with pytest.raises(ValueError, match="at least one labeled position"):
        crf.decode_with_scores(emissions, mask)


def test_end_constraint_shape_is_validated() -> None:
    """A malformed end_ok must raise, not broadcast in masked_fill and forbid every end tag."""
    start_ok = torch.ones(3, dtype=torch.bool)
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    bad_end_ok = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="constraints do not match"):
        LinearChainCRF(3, constraints=(start_ok, transition_ok, bad_end_ok))


def test_disabled_start_end_transitions_are_not_persistent() -> None:
    """With include_start_end_transitions=False, start/end transitions must
    not be saved to (or restorable from) a state_dict -- loading a checkpoint
    saved with them enabled must error instead of silently reintroducing a
    nonzero start/end bias into what's documented as a fixed zero."""
    crf = LinearChainCRF(4, include_start_end_transitions=False)
    assert "start_transitions" not in crf.state_dict()
    assert "end_transitions" not in crf.state_dict()

    enabled = LinearChainCRF(4, include_start_end_transitions=True)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        crf.load_state_dict(enabled.state_dict())


def test_masked_nonfinite_emissions_do_not_poison_the_score() -> None:
    """A masked-out position with a non-finite emission (e.g. -inf padding
    from an upstream head) must not turn the whole row's score and gradient
    into nan -- regression for `step * mask` instead of `torch.where`."""
    torch.manual_seed(13)
    crf = LinearChainCRF(3)
    emissions = torch.randn(1, 3, 3)
    with torch.no_grad():
        emissions[0, 2] = float("-inf")
    emissions.requires_grad_()
    tags = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[True, True, False]])

    ll = crf.log_likelihood(emissions, tags, mask)
    assert torch.isfinite(ll).all()
    ll.sum().backward()
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()


def test_neg_inf_emissions_at_valid_positions_give_finite_gradients() -> None:
    """A tag masked out with -inf at every valid position: the likelihoods
    clamp it to FORBIDDEN, so the gradients stay finite and the values match
    the same input written with FORBIDDEN."""
    torch.manual_seed(14)
    crf = LinearChainCRF(4)
    base = torch.randn(2, 5, 4)
    tags = torch.randint(0, 3, (2, 5))
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    allowed = torch.rand(2, 5, 4) > 0.3
    allowed[:, :, 0] = True

    for fn in (
        lambda e: crf.log_likelihood(e, tags, mask),
        lambda e: crf.partial_log_likelihood(e, allowed, mask),
    ):
        results = []
        for fill in (float("-inf"), FORBIDDEN):
            emissions = base.clone()
            emissions[:, :, 3] = fill
            emissions.requires_grad_()
            crf.zero_grad()
            value = fn(emissions)
            value.sum().backward()
            grads = [emissions.grad, crf.transitions.grad, crf.start_transitions.grad]
            assert torch.isfinite(value).all()
            assert all(torch.isfinite(g).all() for g in grads)
            results.append((value.detach(), [g.clone() for g in grads]))
        (inf_value, inf_grads), (forbidden_value, forbidden_grads) = results
        assert torch.equal(inf_value, forbidden_value)
        assert all(torch.equal(a, b) for a, b in zip(inf_grads, forbidden_grads, strict=True))


def test_neg_inf_log_weights_give_finite_gradients() -> None:
    emissions, allowed, mask = _partial_inputs(23)
    emissions.requires_grad_()
    log_weights = torch.log_softmax(torch.randn(2, 4, 3), -1)
    log_weights[:, :, 2] = float("-inf")
    crf = LinearChainCRF(3)
    value = crf.partial_log_likelihood(emissions, allowed, mask, log_weights=log_weights)
    value.sum().backward()
    assert torch.isfinite(value).all()
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()
    assert torch.isfinite(crf.transitions.grad).all()


def test_emissions_at_or_above_forbidden_are_not_clamped() -> None:
    """The clamp leaves value and gradient untouched at or above FORBIDDEN;
    below it, the emission is scored as FORBIDDEN with zero gradient."""
    torch.manual_seed(15)
    crf = LinearChainCRF(3)
    emissions = torch.randn(1, 3, 3)
    emissions[0, 1, 2] = FORBIDDEN
    emissions[0, 2, 1] = 2 * FORBIDDEN
    emissions.requires_grad_()
    tags = torch.tensor([[0, 2, 1]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    ll = crf.log_likelihood(emissions, tags, mask)
    ll.sum().backward()
    assert emissions.grad is not None
    assert emissions.grad[0, 1, 2] != 0
    assert emissions.grad[0, 2, 1] == 0
    expected = crf.log_likelihood(emissions.detach().clamp(min=FORBIDDEN), tags, mask)
    assert torch.equal(ll.detach(), expected)


def test_decode_does_not_clamp_neg_inf() -> None:
    """decode keeps -inf as an absolute exclusion: a -inf emission loses even
    to a forbidden transition, which FORBIDDEN alone would tie with."""
    transition_ok = torch.ones(2, 2, dtype=torch.bool)
    transition_ok[0, 1] = False
    constraints = (torch.ones(2, dtype=torch.bool), transition_ok, torch.ones(2, dtype=torch.bool))
    crf = LinearChainCRF(2, constraints=constraints, include_start_end_transitions=False)
    with torch.no_grad():
        crf.transitions.zero_()
    # Clamped, path [1, 1] would score FORBIDDEN and beat [0, 1] at
    # -1 + FORBIDDEN; unclamped, [1, 1] is -inf and [0, 1] wins.
    emissions = torch.tensor([[[-1.0, float("-inf")], [-1e5, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    [(path, score)] = crf.decode_with_scores(emissions, mask)
    assert path == [0, 1]
    assert score == pytest.approx(-1.0 + FORBIDDEN)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_runs_directly_in_every_float_dtype(dtype: torch.dtype) -> None:
    """log_likelihood/decode must work when emissions already have any float
    dtype (not just fp32 fed through autocast), and must promote up to at
    least fp32 rather than silently downcasting a higher-precision input."""
    torch.manual_seed(14)
    crf = LinearChainCRF(4)
    emissions = (torch.randn(3, 5, 4) * 2.0).to(dtype)
    tags = torch.randint(0, 4, (3, 5))
    mask = torch.ones(3, 5, dtype=torch.bool)

    ll = crf.log_likelihood(emissions, tags, mask)
    assert ll.dtype is torch.promote_types(dtype, torch.float32)
    assert torch.isfinite(ll).all()
    assert len(crf.decode(emissions, mask)) == 3


def test_fp64_model_stays_fp64() -> None:
    """An fp64 module and its output stay fp64, not downcast to fp32."""
    torch.manual_seed(15)
    crf = LinearChainCRF(3).double()
    emissions = torch.randn(2, 4, 3, dtype=torch.float64)
    tags = torch.randint(0, 3, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)

    ll = crf.log_likelihood(emissions, tags, mask)
    assert ll.dtype is torch.float64
    start, transitions, end = crf._scores(False, torch.float64)
    assert (
        start.dtype is torch.float64
        and transitions.dtype is torch.float64
        and end.dtype is torch.float64
    )


def test_gradcheck_log_likelihood() -> None:
    """Finite-difference gradient check, in fp64 so it is numerically meaningful."""
    torch.manual_seed(16)
    crf = LinearChainCRF(3).double()
    emissions = (torch.randn(2, 4, 3, dtype=torch.float64) * 2.0).requires_grad_()
    tags = torch.randint(0, 3, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)

    def f(x: torch.Tensor) -> torch.Tensor:
        return crf.log_likelihood(x, tags, mask)

    assert torch.autograd.gradcheck(f, (emissions,))


def _enumerate_allowed(crf, emissions, length, allowed, log_weights=None) -> torch.Tensor:
    """logsumexp over the tag sequences inside `allowed`, each scored directly (plus weights)."""
    start, transitions, end = crf._scores(crf.constrain_loss)
    totals = []
    for path in itertools.product(range(crf.num_tags), repeat=length):
        if not all(bool(allowed[t, path[t]]) for t in range(length)):
            continue
        score = start[path[0]] + emissions[0, path[0]]
        for t in range(1, length):
            score = score + transitions[path[t - 1], path[t]] + emissions[t, path[t]]
        if log_weights is not None:
            score = score + sum(log_weights[t, path[t]] for t in range(length))
        totals.append(score + end[path[-1]])
    return torch.logsumexp(torch.stack(totals), 0)


def test_validate_modes() -> None:
    with pytest.raises(ValueError, match="validate must be"):
        LinearChainCRF(3, validate="yes")
    emissions = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    bad = torch.full((2, 4), 3)
    with pytest.raises(ValueError, match="valid tag ids"):
        LinearChainCRF(3, validate="sync").log_likelihood(emissions, bad, mask)
    # None skips value checks; the bad tag then fails inside the computation.
    off = LinearChainCRF(3, validate=None)
    with pytest.raises((IndexError, RuntimeError)):
        off.log_likelihood(emissions, bad, mask)
    allowed = torch.ones(2, 4, 3, dtype=torch.bool)
    allowed[0, 1] = False
    assert torch.isfinite(off.partial_log_likelihood(emissions, allowed, mask)).all()
    for validate in ("async", "sync", None):
        crf = LinearChainCRF(3, validate=validate)
        scripted = torch.jit.script(crf)
        tags = torch.zeros(2, 4, dtype=torch.long)
        assert torch.equal(
            crf.log_likelihood(emissions, tags, mask),
            scripted.log_likelihood(emissions, tags, mask),
        )


@pytest.mark.cuda
def test_sync_validation_raises_on_cuda() -> None:
    crf = LinearChainCRF(3, validate="sync").cuda()
    emissions = torch.randn(2, 4, 3, device="cuda")
    mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="valid tag ids"):
        crf.log_likelihood(emissions, torch.full((2, 4), 3, device="cuda"), mask)


def _partial_inputs(seed: int):
    torch.manual_seed(seed)
    emissions = torch.randn(2, 4, 3) * 2.0
    allowed = torch.rand(2, 4, 3) > 0.4
    # Tag 1 everywhere: every position allows something, and some allowed path is
    # valid under the constraints used below (they forbid only 0 -> 2).
    allowed[:, :, 1] = True
    mask = torch.tensor([[True] * 4, [True, True, True, False]])
    return emissions, allowed, mask


@pytest.mark.parametrize("constrain_loss", [False, True])
def test_partial_log_likelihood_matches_enumeration(constrain_loss: bool) -> None:
    emissions, allowed, mask = _partial_inputs(20)
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 2] = False
    constraints = (torch.ones(3, dtype=torch.bool), transition_ok, torch.ones(3, dtype=torch.bool))
    crf = LinearChainCRF(3, constraints=constraints, constrain_loss=constrain_loss)
    got = crf.partial_log_likelihood(emissions, allowed, mask)
    for row, length in enumerate([4, 3]):
        expected = _enumerate_allowed(crf, emissions[row], length, allowed[row]) - (
            _enumerate_log_partition(crf, emissions[row], length)
        )
        assert torch.allclose(got[row], expected, atol=1e-4), (row, got[row], expected)


def test_partial_log_likelihood_with_log_weights_matches_enumeration() -> None:
    emissions, allowed, mask = _partial_inputs(21)
    log_weights = torch.log_softmax(torch.randn(2, 4, 3), -1)
    crf = LinearChainCRF(3)
    got = crf.partial_log_likelihood(emissions, allowed, mask, log_weights=log_weights)
    for row, length in enumerate([4, 3]):
        numerator = _enumerate_allowed(crf, emissions[row], length, allowed[row], log_weights[row])
        expected = numerator - _enumerate_log_partition(crf, emissions[row], length)
        assert torch.allclose(got[row], expected, atol=1e-4), (row, got[row], expected)


def test_partial_log_likelihood_reduces_to_log_likelihood() -> None:
    """One allowed tag per position is the ordinary likelihood; every tag allowed is log 1."""
    emissions, _, mask = _partial_inputs(22)
    tags = torch.randint(0, 3, (2, 4))
    crf = LinearChainCRF(3)
    one_hot = torch.nn.functional.one_hot(tags, 3).bool()
    assert torch.allclose(
        crf.partial_log_likelihood(emissions, one_hot, mask),
        crf.log_likelihood(emissions, tags, mask),
        atol=1e-5,
    )
    everything = torch.ones(2, 4, 3, dtype=torch.bool)
    assert torch.allclose(
        crf.partial_log_likelihood(emissions, everything, mask), torch.zeros(2), atol=1e-5
    )


def test_partial_log_likelihood_computes_in_fp32_under_bf16_autocast() -> None:
    emissions, allowed, mask = _partial_inputs(23)
    crf = LinearChainCRF(3)
    reference = crf.partial_log_likelihood(emissions, allowed, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = crf.partial_log_likelihood(emissions, allowed, mask)
    assert got.dtype == torch.float32
    assert torch.equal(got, reference)


def test_partial_log_likelihood_validation() -> None:
    emissions, allowed, mask = _partial_inputs(24)
    crf = LinearChainCRF(3)
    with pytest.raises(ValueError, match="allowed must have shape"):
        crf.partial_log_likelihood(emissions, allowed[:, :3], mask)
    with pytest.raises(ValueError, match="log_weights must have shape"):
        crf.partial_log_likelihood(emissions, allowed, mask, log_weights=torch.zeros(2, 4, 2))
    none_allowed = allowed.clone()
    none_allowed[0, 2] = False
    with pytest.raises(ValueError, match="at least one tag"):
        crf.partial_log_likelihood(emissions, none_allowed, mask)
    # A padding position may allow nothing.
    none_allowed = allowed.clone()
    none_allowed[1, 3] = False
    assert torch.isfinite(crf.partial_log_likelihood(emissions, none_allowed, mask)).all()


def test_gradcheck_partial_log_likelihood() -> None:
    torch.manual_seed(25)
    crf = LinearChainCRF(3).double()
    emissions = (torch.randn(2, 4, 3, dtype=torch.float64) * 2.0).requires_grad_()
    _, allowed, mask = _partial_inputs(25)
    log_weights = torch.log_softmax(torch.randn(2, 4, 3, dtype=torch.float64), -1)

    def f(x: torch.Tensor) -> torch.Tensor:
        return crf.partial_log_likelihood(x, allowed, mask, log_weights=log_weights)

    assert torch.autograd.gradcheck(f, (emissions,))


def test_partial_log_likelihood_scripts_and_matches_eager() -> None:
    emissions, allowed, mask = _partial_inputs(26)
    crf = LinearChainCRF(3)
    scripted = torch.jit.script(crf)
    log_weights = torch.randn(2, 4, 3)
    for weights in (None, log_weights):
        assert torch.allclose(
            crf.partial_log_likelihood(emissions, allowed, mask, weights),
            scripted.partial_log_likelihood(emissions, allowed, mask, weights),
        )


def test_potentials() -> None:
    torch.manual_seed(27)
    start_ok = torch.tensor([True, False, True])
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[2, 1] = False
    end_ok = torch.tensor([True, True, False])
    crf = LinearChainCRF(3, constraints=(start_ok, transition_ok, end_ok))
    start, transitions, end = crf.potentials(constrained=False)
    assert torch.equal(start, crf.start_transitions.float())
    assert torch.equal(transitions, crf.transitions.float())
    assert torch.equal(end, crf.end_transitions.float())
    c_start, c_transitions, c_end = crf.potentials()
    assert torch.equal(c_start, start + crf.start_penalty)
    assert c_start[1] < -1e3 and c_transitions[2, 1] < -1e3 and c_end[2] < -1e3
    # They keep their gradient.
    c_transitions.sum().backward()
    assert crf.transitions.grad is not None
    scripted = torch.jit.script(crf)
    for x, y in zip(crf.potentials(), scripted.potentials(), strict=True):
        assert torch.equal(x, y)


def test_async_checks_still_raise_on_cpu() -> None:
    """validate="async" (the default) changes only CUDA inputs; on CPU the checks raise."""
    crf = LinearChainCRF(3)
    assert crf.validate == "async"
    emissions = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid tag ids"):
        crf.log_likelihood(emissions, torch.full((2, 4), 3), mask)
    bad_mask = mask.clone()
    bad_mask[1, 0] = False
    with pytest.raises(ValueError, match="at least one labeled position"):
        crf.log_likelihood(emissions, torch.zeros(2, 4, dtype=torch.long), bad_mask)
    allowed = torch.ones(2, 4, 3, dtype=torch.bool)
    allowed[0, 1] = False
    with pytest.raises(ValueError, match="at least one tag"):
        crf.partial_log_likelihood(emissions, allowed, mask)


@pytest.mark.cuda
def test_async_checks_make_no_host_sync_cuda() -> None:
    """By default the likelihoods run on CUDA without a device-to-host sync."""
    torch.manual_seed(28)
    crf = LinearChainCRF(3).cuda()
    emissions = torch.randn(2, 4, 3, device="cuda", requires_grad=True)
    tags = torch.randint(0, 3, (2, 4), device="cuda")
    mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    allowed = torch.ones(2, 4, 3, dtype=torch.bool, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        total = crf.log_likelihood(emissions, tags, mask).sum()
        total = total + crf.partial_log_likelihood(emissions, allowed, mask).sum()
        total.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    reference = LinearChainCRF(3, validate="sync").cuda()
    reference.load_state_dict(crf.state_dict())
    torch.testing.assert_close(
        crf.log_likelihood(emissions, tags, mask), reference.log_likelihood(emissions, tags, mask)
    )


def _best_allowed_path(crf, emissions, length, allowed):
    """Highest-scoring path inside `allowed`, scored directly with the constraint penalties."""
    start, transitions, end = crf._scores(True)
    best_path, best_score = None, None
    for path in itertools.product(range(crf.num_tags), repeat=length):
        if not all(bool(allowed[t, path[t]]) for t in range(length)):
            continue
        score = start[path[0]] + emissions[0, path[0]]
        for t in range(1, length):
            score = score + transitions[path[t - 1], path[t]] + emissions[t, path[t]]
        score = score + end[path[-1]]
        if best_score is None or score > best_score:
            best_path, best_score = list(path), float(score)
    return best_path, best_score


def test_decode_with_allowed_matches_enumeration() -> None:
    emissions, allowed, mask = _partial_inputs(29)
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 2] = False
    constraints = (torch.ones(3, dtype=torch.bool), transition_ok, torch.ones(3, dtype=torch.bool))
    crf = LinearChainCRF(3, constraints=constraints)
    with torch.no_grad():
        crf.transitions.uniform_(-2, 2)
    got = crf.decode_with_scores(emissions, mask, allowed)
    assert [path for path, _ in got] == crf.decode(emissions, mask, allowed)
    for row, length in enumerate([4, 3]):
        path, score = _best_allowed_path(crf, emissions[row], length, allowed[row])
        assert got[row][0] == path
        assert got[row][1] == pytest.approx(score, abs=1e-4)


def test_decode_with_everything_allowed_is_unrestricted() -> None:
    emissions, _, mask = _partial_inputs(30)
    crf = LinearChainCRF(3)
    everything = torch.ones(2, 4, 3, dtype=torch.bool)
    assert crf.decode_with_scores(emissions, mask, everything) == crf.decode_with_scores(
        emissions, mask
    )
    tags = torch.randint(0, 3, (2, 4))
    one_hot = torch.nn.functional.one_hot(tags, 3).bool()
    assert crf.decode(emissions, mask, one_hot) == [tags[0].tolist(), tags[1, :3].tolist()]


def test_decode_never_picks_a_disallowed_tag_over_a_forbidden_transition() -> None:
    """Allowed tags are absolute; constraint penalties are finite. Only 0 and 1
    are allowed and 0 -> 1 / 1 -> 0 are forbidden, so the best allowed path
    stays on one tag, and tag 2 (unrestricted best) never appears."""
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 1] = False
    transition_ok[1, 0] = False
    constraints = (torch.ones(3, dtype=torch.bool), transition_ok, torch.ones(3, dtype=torch.bool))
    crf = LinearChainCRF(3, constraints=constraints)
    emissions = torch.tensor([[[0.0, -50.0, 100.0], [-50.0, 0.0, 100.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    allowed = torch.tensor([[[True, True, False], [True, True, False]]])
    [path] = crf.decode(emissions, mask, allowed)
    assert 2 not in path
    # With only 0 then only 1 allowed, the path must cross the forbidden 0 -> 1.
    allowed = torch.tensor([[[True, False, False], [False, True, False]]])
    [(path, score)] = crf.decode_with_scores(emissions, mask, allowed)
    assert path == [0, 1]
    assert score < FORBIDDEN / 2


def test_decode_allowed_validation() -> None:
    emissions, allowed, mask = _partial_inputs(31)
    crf = LinearChainCRF(3)
    with pytest.raises(ValueError, match="allowed must have shape"):
        crf.decode(emissions, mask, allowed[:, :3])
    none_allowed = allowed.clone()
    none_allowed[0, 2] = False
    with pytest.raises(ValueError, match="at least one tag"):
        crf.decode_with_scores(emissions, mask, none_allowed)
    # A padding position may allow nothing.
    none_allowed = allowed.clone()
    none_allowed[1, 3] = False
    assert len(crf.decode(emissions, mask, none_allowed)[1]) == 3


def test_decode_with_allowed_scripts_and_matches_eager() -> None:
    emissions, allowed, mask = _partial_inputs(32)
    crf = LinearChainCRF(3)
    scripted = torch.jit.script(crf)
    for restriction in (None, allowed):
        assert crf.decode(emissions, mask, restriction) == scripted.decode(
            emissions, mask, restriction
        )
        assert crf.decode_with_scores(emissions, mask, restriction) == (
            scripted.decode_with_scores(emissions, mask, restriction)
        )


def _ranked_paths(crf, emissions, length, allowed=None):
    """Every path of the given length (inside `allowed`, if given), best first."""
    start, transitions, end = (x.detach() for x in crf._scores(True))
    scored = []
    for path in itertools.product(range(crf.num_tags), repeat=length):
        if allowed is not None and not all(bool(allowed[t, path[t]]) for t in range(length)):
            continue
        score = start[path[0]] + emissions[0, path[0]]
        for t in range(1, length):
            score = score + transitions[path[t - 1], path[t]] + emissions[t, path[t]]
        scored.append((float(score + end[path[-1]]), list(path)))
    scored.sort(key=lambda item: -item[0])
    return scored


@pytest.mark.parametrize("use_allowed", [False, True])
def test_decode_top_k_matches_enumeration(use_allowed: bool) -> None:
    emissions, allowed, mask = _partial_inputs(33)
    transition_ok = torch.ones(3, 3, dtype=torch.bool)
    transition_ok[0, 2] = False
    constraints = (torch.ones(3, dtype=torch.bool), transition_ok, torch.ones(3, dtype=torch.bool))
    crf = LinearChainCRF(3, constraints=constraints)
    with torch.no_grad():
        crf.transitions.uniform_(-2, 2)
    restriction = allowed if use_allowed else None
    got = crf.decode_top_k(emissions, mask, 10, restriction)
    for row, length in enumerate([4, 3]):
        expected = _ranked_paths(crf, emissions[row], length, allowed[row] if use_allowed else None)
        expected = expected[:10]
        assert [path for path, _ in got[row]] == [path for _, path in expected]
        for (_, score), (expected_score, _) in zip(got[row], expected, strict=True):
            assert score == pytest.approx(expected_score, abs=1e-4)


def test_decode_top_k_first_is_decode_with_scores() -> None:
    emissions, allowed, mask = _partial_inputs(34)
    crf = LinearChainCRF(3)
    for restriction in (None, allowed):
        top = crf.decode_top_k(emissions, mask, 3, restriction)
        best = crf.decode_with_scores(emissions, mask, restriction)
        for row in range(2):
            assert top[row][0][0] == best[row][0]
            assert top[row][0][1] == pytest.approx(best[row][1])


def test_decode_top_k_returns_only_existing_paths() -> None:
    """A length-1 row has num_tags paths; an allowed mask can leave fewer."""
    crf = LinearChainCRF(3)
    emissions = torch.randn(2, 2, 3)
    mask = torch.tensor([[True, False], [True, True]])
    got = crf.decode_top_k(emissions, mask, 20)
    assert len(got[0]) == 3 and len(got[1]) == 9
    assert sorted(path for path, _ in got[0]) == [[0], [1], [2]]
    allowed = torch.zeros(2, 2, 3, dtype=torch.bool)
    allowed[:, :, 1] = True
    allowed[1, 1, 2] = True
    got = crf.decode_top_k(emissions, mask, 20, allowed)
    assert [path for path, _ in got[0]] == [[1]]
    assert sorted(path for path, _ in got[1]) == [[1, 1], [1, 2]]


def test_decode_top_k_validation_and_scripting() -> None:
    emissions, allowed, mask = _partial_inputs(35)
    crf = LinearChainCRF(3)
    with pytest.raises(ValueError, match="k must be at least 1"):
        crf.decode_top_k(emissions, mask, 0)
    with pytest.raises(ValueError, match="allowed must have shape"):
        crf.decode_top_k(emissions, mask, 2, allowed[:, :3])
    scripted = torch.jit.script(crf)
    for restriction in (None, allowed):
        assert crf.decode_top_k(emissions, mask, 4, restriction) == scripted.decode_top_k(
            emissions, mask, 4, restriction
        )
    compiled = torch.compile(crf.decode_top_k)
    assert compiled(emissions, mask, 4) == crf.decode_top_k(emissions, mask, 4)


def test_length_sorted_cpu_decode_matches_rows_alone() -> None:
    """At 64 x 12 tags, CPU decode sorts rows by length and steps only the
    running ones; every result must equal that row decoded alone, including
    with no row reaching the padded length."""
    g = torch.Generator().manual_seed(37)
    emissions = torch.randn(64, 20, 12, generator=g) * 3
    lengths = torch.randint(1, 18, (64,), generator=g)
    mask = torch.arange(20).unsqueeze(0) < lengths.unsqueeze(1)
    allowed = torch.rand(64, 20, 12, generator=g) > 0.5
    allowed[..., 0] = True
    torch.manual_seed(37)
    transition_ok = torch.rand(12, 12, generator=g) > 0.3
    ok = torch.ones(12, dtype=torch.bool)
    crf = LinearChainCRF(12, constraints=(ok, transition_ok, ok))
    scripted = torch.jit.script(crf)
    for restriction in (None, allowed):
        rows = [
            (
                emissions[i : i + 1],
                mask[i : i + 1],
                None if restriction is None else restriction[i : i + 1],
            )
            for i in range(64)
        ]
        alone = [crf.decode_with_scores(*row)[0] for row in rows]
        assert crf.decode_with_scores(emissions, mask, restriction) == alone
        assert scripted.decode_with_scores(emissions, mask, restriction) == alone
        top = [crf.decode_top_k(e, m, 3, a)[0] for e, m, a in rows]
        assert crf.decode_top_k(emissions, mask, 3, restriction) == top
        assert scripted.decode_top_k(emissions, mask, 3, restriction) == top
    tags, scores = torch.compile(crf.decode_to_tensor, dynamic=False)(emissions, mask)
    expected_tags, expected_scores = crf.decode_to_tensor(emissions, mask)
    assert torch.equal(tags, expected_tags) and torch.equal(scores, expected_scores)
    assert (tags[~mask] == -1).all()


def test_length_sorted_cpu_decode_is_eager_only(monkeypatch) -> None:
    """The sort reads running row counts to the host: eager uses it, while
    under torch.compile each decode has as many graphs as below the cutoff,
    and torch.jit.trace does not fix one batch's counts into the trace."""
    g = torch.Generator().manual_seed(38)

    def batch():
        lengths = torch.randint(1, 21, (64,), generator=g)
        return torch.randn(64, 20, 12, generator=g), torch.arange(20) < lengths.unsqueeze(1)

    emissions, mask = batch()
    crf = LinearChainCRF(12, validate=None)
    sorts: list[bool] = []
    real = LinearChainCRF._sorts_rows

    def spy(self, emissions, entries):
        sorts.append(real(self, emissions, entries))
        return sorts[-1]

    monkeypatch.setattr(LinearChainCRF, "_sorts_rows", spy)
    crf.decode(emissions, mask)
    crf.decode_top_k(emissions, mask, 3)
    assert sorts == [True, True]
    monkeypatch.undo()  # torch.jit.trace scripts the CRF's exported methods

    # Below the cutoff (4 x 12 x 12 entries per step) nothing is sorted;
    # torch before 2.12 also breaks at decode's list conversion there.
    small = emissions[:4], mask[:4]
    for name, extra in (("decode", ()), ("decode_to_tensor", ()), ("decode_top_k", (3,))):
        counts = []
        for args in ((emissions, mask), small):
            torch._dynamo.reset()
            explained = torch._dynamo.explain(getattr(crf, name))(*args, *extra)
            counts.append(explained.graph_count)
        assert counts[0] == counts[1], (name, counts, explained.break_reasons)

    class Decode(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.crf = crf

        def forward(self, emissions, mask):
            return self.crf.decode_to_tensor(emissions, mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(Decode(), (emissions, mask))
    for emissions, mask in [batch() for _ in range(3)]:
        tags, scores = traced(emissions, mask)
        expected_tags, expected_scores = crf.decode_to_tensor(emissions, mask)
        assert torch.equal(tags, expected_tags) and torch.equal(scores, expected_scores)


@pytest.mark.cuda
def test_decode_top_k_matches_cpu_cuda() -> None:
    emissions, allowed, mask = _partial_inputs(36)
    crf = LinearChainCRF(3)
    expected = crf.decode_top_k(emissions, mask, 5, allowed)
    crf = crf.cuda()
    got = crf.decode_top_k(emissions.cuda(), mask.cuda(), 5, allowed.cuda())
    assert [[path for path, _ in row] for row in got] == [
        [path for path, _ in row] for row in expected
    ]
    for row_got, row_expected in zip(got, expected, strict=True):
        for (_, a), (_, b) in zip(row_got, row_expected, strict=True):
            assert a == pytest.approx(b, abs=1e-4)


def test_potentials_follow_a_wider_model_dtype() -> None:
    """fp32 or wider: an fp64 model's potentials stay fp64, as in every other method."""
    crf = LinearChainCRF(3)
    assert all(x.dtype is torch.float32 for x in crf.potentials())
    assert all(x.dtype is torch.float32 for x in crf.to(torch.bfloat16).potentials())
    crf64 = LinearChainCRF(3).double()
    start, transitions, end = crf64.potentials(constrained=False)
    assert torch.equal(transitions, crf64.transitions)
    assert start.dtype is transitions.dtype is end.dtype is torch.float64


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_warns_when_parameters_are_16_bit(dtype: torch.dtype) -> None:
    crf = LinearChainCRF(3).to(dtype)
    emissions = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.warns(UserWarning, match="keep its parameters in fp32"):
        result = crf.log_likelihood(emissions, torch.zeros(2, 4, dtype=torch.long), mask)
    assert result.dtype is torch.float32
    with pytest.warns(UserWarning, match="keep its parameters in fp32"):
        torch.jit.script(crf).decode(emissions, mask)


def test_no_precision_warning_for_fp32_or_fp64_parameters(recwarn) -> None:
    emissions = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    LinearChainCRF(3).decode(emissions, mask)
    LinearChainCRF(3).double().decode(emissions.double(), mask)
    assert not [w for w in recwarn if "keep its parameters" in str(w.message)]


@pytest.mark.parametrize(("batch", "num_tags"), [(2, 3), (8, 37), (40, 37)])
def test_partial_log_likelihood_batched_path_is_bitwise_the_two_pass_result(
    batch: int, num_tags: int
) -> None:
    """Small inputs run numerator and denominator as one 2B-row recursion
    (B * C^2 <= 16384 on CPU); the result must be bitwise the two-pass one."""
    torch.manual_seed(37)
    crf = LinearChainCRF(num_tags)
    emissions = (torch.randn(batch, 6, num_tags) * 3).requires_grad_()
    allowed = torch.rand(batch, 6, num_tags) > 0.5
    allowed[..., 0] = True
    mask = torch.ones(batch, 6, dtype=torch.bool)
    mask[0, 4:] = False
    log_weights = torch.log_softmax(torch.randn(batch, 6, num_tags), -1)
    got = crf.partial_log_likelihood(emissions, allowed, mask, log_weights)
    (got_grad,) = torch.autograd.grad(got.sum(), emissions)

    start, transitions, end = crf._scores(crf.constrain_loss)
    floored = emissions.clamp(min=FORBIDDEN)
    weighted = (floored + log_weights).clamp(min=FORBIDDEN)
    restricted = weighted.masked_fill(~allowed, FORBIDDEN)
    expected = crf._log_partition(
        restricted, mask, start, transitions, end, False
    ) - crf._log_partition(floored, mask, start, transitions, end, False)
    (expected_grad,) = torch.autograd.grad(expected.sum(), emissions)
    assert torch.equal(got, expected)
    assert torch.equal(got_grad, expected_grad)


def test_attribute_and_state_dict_names_are_stable() -> None:
    """Saved checkpoints and downstream code use these names; renaming any of
    them breaks loading a checkpoint saved by an earlier version."""
    crf = LinearChainCRF(3)
    assert list(crf.state_dict()) == [
        "transitions",
        "start_transitions",
        "end_transitions",
        "start_penalty",
        "transition_penalty",
        "end_penalty",
    ]
    assert crf.num_tags == 3
    no_ends = LinearChainCRF(3, include_start_end_transitions=False)
    assert list(no_ends.state_dict()) == [
        "transitions",
        "start_penalty",
        "transition_penalty",
        "end_penalty",
    ]


def test_decode_to_tensor_matches_decode_with_scores() -> None:
    emissions, allowed, mask = _partial_inputs(38)
    crf = LinearChainCRF(3)
    for restriction in (None, allowed):
        tags, scores = crf.decode_to_tensor(emissions, mask, restriction)
        assert tags.dtype is torch.long and tags.shape == mask.shape
        assert scores.shape == (2,)
        expected = crf.decode_with_scores(emissions, mask, restriction)
        for row, (path, score) in enumerate(expected):
            assert tags[row, : len(path)].tolist() == path
            assert torch.all(tags[row, len(path) :] == -1)
            assert scores[row].item() == score
    tags, _ = crf.decode_to_tensor(emissions, mask, pad_tag=0)
    assert torch.all(tags[1, 3:] == 0)
    scripted = torch.jit.script(crf)
    for got, want in zip(
        scripted.decode_to_tensor(emissions, mask),
        crf.decode_to_tensor(emissions, mask),
        strict=True,
    ):
        assert torch.equal(got, want)
    torch._dynamo.reset()
    compiled = torch.compile(crf.decode_to_tensor)
    assert torch.equal(compiled(emissions, mask)[0], crf.decode_to_tensor(emissions, mask)[0])


@pytest.mark.cuda
def test_decode_to_tensor_makes_no_host_sync_cuda() -> None:
    emissions, allowed, mask = _partial_inputs(39)
    crf = LinearChainCRF(3).cuda()
    emissions, allowed, mask = emissions.cuda(), allowed.cuda(), mask.cuda()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tags, scores = crf.decode_to_tensor(emissions, mask, allowed)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert tags.is_cuda and scores.is_cuda
    expected = crf.decode_with_scores(emissions, mask, allowed)
    for row, (path, _) in enumerate(expected):
        assert tags[row, : len(path)].tolist() == path


@pytest.fixture
def keep_denormals():
    """This thread with FTZ/DAZ off, so torch.equal and .item() read denormals.

    A CPU torch.compile on torch 2.4 or 2.5 with gcc < 13 turns both on in the
    thread that loads the kernel and in threads it starts later. If either was
    on, both are turned back on afterwards. Skips where they cannot be cleared.
    """
    daz = torch.equal(torch.tensor([1e-40]), torch.zeros(1))
    if not torch.set_flush_denormal(False) and daz:
        pytest.skip("cannot turn off denormal flushing here")
    yield
    if daz:
        torch.set_flush_denormal(True)


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _clamp_inputs(seed: int, scale: float, band: bool, inf: bool = False):
    g = torch.Generator().manual_seed(seed)
    emissions = torch.randn(64, 10, 17, generator=g) * scale
    if band:  # tag 1 about 95 below the rest: exp's denormal range
        emissions[..., 1] -= 95.0
    emissions[torch.rand(64, 10, 17, generator=g) < 0.1] = float("-inf")
    if inf:  # row 1's first position, always valid
        emissions[1, 0, 3] = float("inf")
    lengths = torch.randint(1, 11, (64,), generator=g)
    lengths[0] = 10
    mask = torch.arange(10).unsqueeze(0) < lengths.unsqueeze(1)
    tags = torch.randint(0, 17, (64, 10), generator=g)
    allowed = torch.rand(64, 10, 17, generator=g) > 0.5
    allowed[..., 0] = True
    transition_ok = torch.rand(17, 17, generator=g) > 0.4
    transition_ok[0, :] = transition_ok[:, 0] = True
    start_ok = torch.rand(17, generator=g) > 0.3
    start_ok[0] = True
    return emissions, mask, tags, allowed, (start_ok, transition_ok, start_ok.clone())


@pytest.fixture
def clamped_calls(monkeypatch):
    """A list that grows by one entry per `_ClampedLogSumExp` call."""
    import torch_linear_crf.crf as crf_module

    calls: list[int] = []
    real = crf_module._clamped_logsumexp
    monkeypatch.setattr(
        crf_module, "_clamped_logsumexp", lambda x, d: calls.append(d) or real(x, d)
    )
    return calls


def _clamp_crfs(constraints, constrain_loss: bool = True, validate: str | None = "async"):
    """The same CRF with the fast path off, then on."""
    crfs = []
    for fast in (False, True):
        torch.manual_seed(40)
        crfs.append(
            LinearChainCRF(
                17,
                constraints=constraints,
                constrain_loss=constrain_loss,
                validate=validate,
                fast_cpu_logsumexp=fast,
            )
        )
    return crfs


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equal, with nan matching nan."""
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a.masked_fill(nan, 0), b.masked_fill(nan, 0))


# Each case: (name, constrain_loss, clamped calls, fn(crf, x, tags, mask, allowed, log_weights)).
# Batch 64 takes partial_log_likelihood's two separate recursions; the
# marginals and forward_backward run constrained under constrain_loss=False.
# The calls: T - 1 steps and the final sum per forward recursion (T = 10),
# T - 1 steps per backward one.
_CLAMP_SITES = [
    ("log_likelihood", True, 10, lambda c, x, t, m, a, w: c.log_likelihood(x, t, m)),
    ("pll", False, 10, lambda c, x, t, m, a, w: c.partial_log_likelihood(x, a, m)),
    ("pll weights", True, 20, lambda c, x, t, m, a, w: c.partial_log_likelihood(x, a, m, w)),
    ("marginals", False, 19, lambda c, x, t, m, a, w: c.marginals(x, m)),
    ("log_marginals", False, 19, lambda c, x, t, m, a, w: c.log_marginals(x, m)),
    (
        "forward_backward",
        False,
        19,
        lambda c, x, t, m, a, w: torch.cat([v.flatten() for v in c.forward_backward(x, m)]),
    ),
]


@pytest.mark.parametrize(
    ("scale", "band", "inf"),
    [(3.0, False, False), (30.0, False, False), (3.0, True, False), (3.0, False, True)],
)
@pytest.mark.parametrize(("name", "constrain_loss", "calls", "fn"), _CLAMP_SITES)
def test_fast_cpu_logsumexp_is_bitwise_and_engaged(
    one_thread, keep_denormals, clamped_calls, scale, band, inf, name, constrain_loss, calls, fn
) -> None:
    """On one thread every recursion at each site runs the clamped logsumexp;
    values and every gradient must equal torch.logsumexp's bitwise, denormal
    terms and +inf included."""
    emissions, mask, tags, allowed, constraints = _clamp_inputs(40, scale, band, inf)
    log_weights = torch.log_softmax(emissions.nan_to_num(neginf=0.0, posinf=0.0) / 3.0, -1)
    results = []
    for crf in _clamp_crfs(constraints, constrain_loss):
        params = [crf.transitions, crf.start_transitions, crf.end_transitions]
        x = emissions.clone().requires_grad_()
        value = fn(crf, x, tags, mask, allowed, log_weights)
        results.append((value, *torch.autograd.grad(value.sum(), [x, *params])))
        if not crf.fast_cpu_logsumexp:
            assert not clamped_calls
    assert len(clamped_calls) == calls, name
    for a, b in zip(*results, strict=True):
        assert _same(a, b), name


def _transform_case(validate: str | None = "async"):
    """Finite inputs (for second derivatives), the CRF with the fast path off
    and on, and the loss both run."""
    emissions, mask, tags, _, constraints = _clamp_inputs(42, 3.0, False)

    def loss(crf, x):
        return crf.log_likelihood(x, tags, mask).sum()

    return emissions.clamp(min=-50.0), *_clamp_crfs(constraints, validate=validate), loss


def test_fast_cpu_logsumexp_double_backward(one_thread, clamped_calls) -> None:
    emissions, stock, fast, loss = _transform_case()

    def double_backward(crf):
        x = emissions.clone().requires_grad_()
        (gx,) = torch.autograd.grad(loss(crf, x), x, create_graph=True)
        return torch.autograd.grad((gx * gx).sum(), x)[0]

    assert torch.equal(double_backward(fast), double_backward(stock))
    assert clamped_calls


def test_fast_cpu_logsumexp_steps_aside_under_torch_func(one_thread, clamped_calls) -> None:
    emissions, stock, fast, loss = _transform_case()
    expected = torch.func.grad(lambda x: loss(stock, x))(emissions)
    assert torch.equal(torch.func.grad(lambda x: loss(fast, x))(emissions), expected)
    stacked = torch.stack([emissions, emissions / 2])
    expected = torch.func.vmap(lambda x: loss(stock, x))(stacked)
    assert torch.equal(torch.func.vmap(lambda x: loss(fast, x))(stacked), expected)
    assert not clamped_calls


def test_fast_cpu_logsumexp_forward_mode_ad(one_thread, clamped_calls) -> None:
    emissions, stock, fast, loss = _transform_case()
    tangent = torch.randn(emissions.shape, generator=torch.Generator().manual_seed(43))
    jvps = []
    for crf in (stock, fast):
        with fwad.dual_level():
            jvps.append(fwad.unpack_dual(loss(crf, fwad.make_dual(emissions, tangent))).tangent)
    assert torch.equal(jvps[1], jvps[0])
    assert clamped_calls


def test_clamped_logsumexp_jvp_keeps_the_denormal_band(keep_denormals) -> None:
    from torch_linear_crf.crf import _ClampedLogSumExp

    x = torch.tensor([[0.0, -95.0, -1e4]])
    tangent = torch.tensor([[0.0, 1.0, 1.0]])
    tangents = []
    for fn in (lambda v: torch.logsumexp(v, 1), lambda v: _ClampedLogSumExp.apply(v, 1)):
        with fwad.dual_level():
            tangents.append(fwad.unpack_dual(fn(fwad.make_dual(x, tangent))).tangent)
    assert tangents[0].item() > 0.0  # exp(-95)
    assert torch.equal(tangents[1], tangents[0])


def test_fast_cpu_logsumexp_steps_aside_under_jit_trace(one_thread) -> None:
    emissions, stock, fast, loss = _transform_case()

    class Loss(torch.nn.Module):
        def __init__(self, crf):
            super().__init__()
            self.crf = crf

        def forward(self, x):
            return loss(self.crf, x)

    traced = torch.jit.trace(Loss(fast), emissions)
    assert "ClampedLogSumExp" not in str(traced.graph)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    buffer.seek(0)
    assert torch.equal(torch.jit.load(buffer)(emissions), loss(stock, emissions))


def test_fast_cpu_logsumexp_gate(one_thread) -> None:
    emissions, mask, tags, _, constraints = _clamp_inputs(41, 3.0, False)
    constrained = LinearChainCRF(17, constraints=constraints, constrain_loss=True)
    assert constrained._clamps_logsumexp(emissions, True, False)
    assert not constrained._clamps_logsumexp(emissions[:8], True, False)  # below 16384
    assert not constrained._clamps_logsumexp(emissions.double(), True, False)
    assert not LinearChainCRF(17)._clamps_logsumexp(emissions, True, False)  # nothing forbidden
    assert LinearChainCRF(17)._clamps_logsumexp(emissions, False, True)  # masked emissions
    off = LinearChainCRF(17, constraints=constraints, constrain_loss=True, fast_cpu_logsumexp=False)
    for forbidden in range(3):  # only start, only transitions, or only end
        parts = [torch.ones(17, dtype=torch.bool), torch.ones(17, 17, dtype=torch.bool)]
        parts.append(torch.ones(17, dtype=torch.bool))
        parts[forbidden].view(-1)[1] = False
        one = LinearChainCRF(17, constraints=tuple(parts), constrain_loss=True)
        assert one._clamps_logsumexp(emissions, True, False)
    assert not off._clamps_logsumexp(emissions, True, True)
    # 16384 << (threads - 1) entries per step: the first batch of 17 tags at or past it.
    for threads, first in ((1, 57), (2, 114), (3, 227), (4, 454)):
        torch.set_num_threads(threads)
        assert not constrained._clamps_logsumexp(torch.empty(first - 1, 1, 17), True, False)
        assert constrained._clamps_logsumexp(torch.empty(first, 1, 17), True, False)
    torch.set_num_threads(5)
    assert not constrained._clamps_logsumexp(torch.empty(100000, 1, 17), True, False)
    torch.set_num_threads(1)
    # 16 tags: 64 x 16^2 is exactly 16384, which engages.
    assert LinearChainCRF(16)._clamps_logsumexp(torch.empty(64, 1, 16), False, True)
    assert not LinearChainCRF(16)._clamps_logsumexp(torch.empty(63, 1, 16), False, True)
    scripted = torch.jit.script(constrained)
    assert torch.equal(
        scripted.log_likelihood(emissions, tags, mask),
        constrained.log_likelihood(emissions, tags, mask),
    )


def test_fast_cpu_logsumexp_skips_stacked_pll(one_thread, clamped_calls) -> None:
    """The stacked recursion at 64 x 16^2 (2 * 64 rows) keeps torch.logsumexp,
    though the gate admits the batch."""
    emissions, mask, _, allowed, _ = _clamp_inputs(45, 3.0, False)
    crf = LinearChainCRF(16)
    assert crf._clamps_logsumexp(emissions[..., :16], False, True)
    crf.partial_log_likelihood(emissions[..., :16], allowed[..., :16], mask)
    assert not clamped_calls


def test_module_forward_backward_keeps_torch_logsumexp(one_thread, clamped_calls) -> None:
    from torch_linear_crf import forward_backward

    emissions, mask, _, _, constraints = _clamp_inputs(46, 3.0, True)
    crf = LinearChainCRF(17, constraints=constraints)
    assert crf._clamps_logsumexp(emissions, True, True)
    forward_backward(emissions, mask, *crf.potentials())
    assert not clamped_calls


def test_fast_cpu_logsumexp_steps_aside_under_make_fx(one_thread, clamped_calls) -> None:
    from torch.fx.experimental.proxy_tensor import make_fx

    # validate=None: the value checks cannot run on traced tensors.
    emissions, stock, fast, loss = _transform_case(validate=None)

    def grad(crf):
        def fn(x):
            x = x.detach().requires_grad_()
            return torch.autograd.grad(loss(crf, x), x)[0]

        return fn

    for pre_dispatch in (False, True):
        traced = make_fx(grad(fast), pre_dispatch=pre_dispatch)(emissions)
        assert not clamped_calls
        assert torch.equal(traced(emissions), grad(stock)(emissions))


def test_fast_cpu_logsumexp_steps_aside_under_torch_export(one_thread, clamped_calls) -> None:
    # validate=None: the value checks cannot run on traced tensors.
    emissions, stock, fast, loss = _transform_case(validate=None)

    class Loss(torch.nn.Module):
        def __init__(self, crf):
            super().__init__()
            self.crf = crf

        def forward(self, x):
            return loss(self.crf, x)

    exported = torch.export.export(Loss(fast), (emissions,), strict=False)
    assert not clamped_calls
    assert torch.equal(exported.module()(emissions), loss(stock, emissions))


def test_fast_cpu_logsumexp_skips_an_fp64_model(one_thread, clamped_calls) -> None:
    """fp32 emissions into an fp64 CRF recurse in fp64, where the clamp at -87 is wrong."""
    emissions, mask, tags, _, constraints = _clamp_inputs(44, 3.0, False)
    double = LinearChainCRF(17, constraints=constraints, constrain_loss=True).double()
    double.log_likelihood(emissions, tags, mask)
    assert not clamped_calls


def test_clamped_exp_is_exp_bitwise_when_exact(keep_denormals) -> None:
    from torch_linear_crf.crf import _clamped_exp

    y = torch.cat([torch.linspace(-120.0, -80.0, 4001), torch.tensor([0.0, -1e4, float("-inf")])])
    assert torch.equal(_clamped_exp(y, True), y.exp())
    loose = _clamped_exp(y, False)
    assert torch.equal(loose[y >= -87.0], y.exp()[y >= -87.0]) and not loose[y < -87.0].any()
