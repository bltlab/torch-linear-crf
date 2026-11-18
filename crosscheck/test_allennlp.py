"""torch-linear-crf against AllenNLP's ConditionalRandomField (pinned commit).

AllenNLP applies constraints only in `viterbi_tags`, as a replacement score
of -10000 for a disallowed transition, where we add FORBIDDEN to the learned
score; the two agree on every path that uses no disallowed transition, which
the cases guarantee exists. `viterbi_tags` also builds its transition
matrix in fp32 whatever the model's dtype, so its scores carry fp32 rounding
and are compared at VITERBI_RTOL and VITERBI_ATOL; the paths are compared
exactly.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from allennlp_stub import load
from cases import ATOL, RTOL, case_params, make_case, ours

# fp32 rounding of sums of terms ~10 in magnitude, relative and absolute.
VITERBI_RTOL = 1e-6
VITERBI_ATOL = 1e-5


@pytest.fixture(scope="module")
def allennlp(tmp_path_factory):
    return load(tmp_path_factory.mktemp("allennlp"))


def allowed_pairs(constraints):
    """AllenNLP's constraint format: (from, to) ids, start = C, end = C + 1."""
    start_ok, transition_ok, end_ok = constraints
    num_tags = start_ok.numel()
    pairs = [(i, j) for i in range(num_tags) for j in range(num_tags) if transition_ok[i, j]]
    pairs += [(num_tags, j) for j in range(num_tags) if start_ok[j]]
    pairs += [(i, num_tags + 1) for i in range(num_tags) if end_ok[i]]
    return pairs


@pytest.mark.parametrize(("seed", "shape", "constrained"), case_params())
def test_log_likelihood_viterbi_top_k(allennlp, seed, shape, constrained) -> None:
    num_tags, length, batch = shape
    case = make_case(seed, num_tags, length, batch, constrained)
    crf = ours(case)
    constraints = allowed_pairs(case["constraints"]) if constrained else None
    theirs = allennlp.ConditionalRandomField(num_tags, constraints=constraints).double()
    with torch.no_grad():
        theirs.start_transitions.copy_(case["start"])
        theirs.transitions.copy_(case["transitions"])
        theirs.end_transitions.copy_(case["end"])
    emissions, tags, mask = case["emissions"], case["tags"], case["mask"]

    if not constrained:
        # AllenNLP's likelihood never applies constraints; ours does only with
        # constrain_loss, so compare the unconstrained objective.
        theirs_ll = theirs._joint_likelihood(
            emissions, theirs.transitions, tags, mask
        ) - theirs._input_likelihood(emissions, theirs.transitions, mask)
        torch.testing.assert_close(
            crf.log_likelihood(emissions, tags, mask), theirs_ll, rtol=RTOL, atol=ATOL
        )
        torch.testing.assert_close(
            ours(case, parallel_scan=True).log_likelihood(emissions, tags, mask),
            theirs_ll,
            rtol=RTOL,
            atol=ATOL,
        )

    best = theirs.viterbi_tags(emissions, mask)
    decoded = crf.decode_with_scores(emissions, mask)
    for (path, score), (theirs_path, theirs_score) in zip(decoded, best, strict=True):
        assert path == theirs_path
        assert score == pytest.approx(float(theirs_score), rel=VITERBI_RTOL, abs=VITERBI_ATOL)

    k = 3
    theirs_top = theirs.viterbi_tags(emissions, mask, top_k=k)
    ours_top = crf.decode_top_k(emissions, mask, k)
    for b in range(batch):
        theirs_row = [(p, float(s)) for p, s in theirs_top[b]]
        # AllenNLP fills a short row's missing paths from its -10000 sentinel
        # states; compare only the paths that exist.
        theirs_row = [(p, s) for p, s in theirs_row if s > -5000][: len(ours_top[b])]
        ours_row = ours_top[b][: len(theirs_row)]
        if constrained:
            ours_row = [(p, s) for p, s in ours_row if s > -5000]
            theirs_row = theirs_row[: len(ours_row)]
        assert [s for _, s in ours_row] == pytest.approx(
            [s for _, s in theirs_row], rel=VITERBI_RTOL, abs=VITERBI_ATOL
        )
        scores = [s for _, s in theirs_row]
        if all(a - c > 1e-9 for a, c in itertools.pairwise(scores)):
            assert [p for p, _ in ours_row] == [p for p, _ in theirs_row]
