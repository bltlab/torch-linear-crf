"""torch-linear-crf against pytorch-crf 0.7.2 (no constraints there)."""

from __future__ import annotations

import pytest
import torch
import torchcrf
from cases import ATOL, RTOL, case_params, make_case, ours


@pytest.mark.parametrize(("seed", "shape", "constrained"), case_params((False,)))
def test_log_likelihood_and_decode(seed, shape, constrained) -> None:
    num_tags, length, batch = shape
    case = make_case(seed, num_tags, length, batch, constrained)
    crf = ours(case)
    theirs = torchcrf.CRF(num_tags, batch_first=True).double()
    with torch.no_grad():
        theirs.start_transitions.copy_(case["start"])
        theirs.transitions.copy_(case["transitions"])
        theirs.end_transitions.copy_(case["end"])
    emissions, tags, mask = case["emissions"], case["tags"], case["mask"]

    torch.testing.assert_close(
        crf.log_likelihood(emissions, tags, mask),
        theirs(emissions, tags, mask, reduction="none"),
        rtol=RTOL,
        atol=ATOL,
    )
    assert crf.decode(emissions, mask) == theirs.decode(emissions, mask)
    torch.testing.assert_close(
        ours(case, parallel_scan=True).log_likelihood(emissions, tags, mask),
        theirs(emissions, tags, mask, reduction="none"),
        rtol=RTOL,
        atol=ATOL,
    )
