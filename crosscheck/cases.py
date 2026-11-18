"""Cross-checks against external CRF implementations, at pinned versions.

    uvx nox -s crosscheck

Each test builds the same chain in torch-linear-crf and in one external
implementation (pytorch-crf 0.7.2, torch-struct 0.5, AllenNLP's
ConditionalRandomField at a pinned commit) and requires them to agree in
fp64 on random batches: a change here that moves any value past the
tolerances below is drift, whichever side is wrong.
"""

from __future__ import annotations

import pytest
import torch

from torch_linear_crf import LinearChainCRF

# fp64 agreement between independent implementations: summation order
# differs, so not bitwise, but far tighter than any real disagreement.
RTOL = 1e-12
ATOL = 1e-9


def make_case(seed: int, num_tags: int, length: int, batch: int, constrained: bool, min_length=1):
    """One random batch: parameters, emissions, tags, lengths and constraints."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(min_length, length + 1, (batch,), generator=g)
    lengths[0] = length
    case = {
        "start": torch.randn(num_tags, generator=g, dtype=torch.float64),
        "transitions": torch.randn(num_tags, num_tags, generator=g, dtype=torch.float64),
        "end": torch.randn(num_tags, generator=g, dtype=torch.float64),
        "emissions": torch.randn(batch, length, num_tags, generator=g, dtype=torch.float64) * 3,
        "tags": torch.randint(0, num_tags, (batch, length), generator=g),
        "lengths": lengths,
        "mask": torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1),
        "constraints": None,
    }
    if constrained:
        start_ok = torch.rand(num_tags, generator=g) < 0.7
        transition_ok = torch.rand(num_tags, num_tags, generator=g) < 0.6
        end_ok = torch.rand(num_tags, generator=g) < 0.7
        # Tag 0 may start, repeat and end, so every length has a valid path.
        start_ok[0] = transition_ok[0, 0] = end_ok[0] = True
        case["constraints"] = (start_ok, transition_ok, end_ok)
    return case


def ours(case: dict, parallel_scan: bool = False) -> LinearChainCRF:
    """torch-linear-crf in fp64 with the case's parameters and constraints."""
    crf = LinearChainCRF(
        case["start"].numel(), constraints=case["constraints"], parallel_scan=parallel_scan
    ).double()
    with torch.no_grad():
        crf.start_transitions.copy_(case["start"])
        crf.transitions.copy_(case["transitions"])
        crf.end_transitions.copy_(case["end"])
    return crf


SHAPES = [(3, 2, 4), (5, 7, 8), (9, 12, 16), (4, 20, 6)]
SEEDS = range(5)


def case_params(constrained_options=(False, True)):
    """pytest params over seeds x shapes x constrained, with readable ids."""
    return [
        pytest.param(
            seed,
            shape,
            constrained,
            id=f"s{seed}-C{shape[0]}T{shape[1]}B{shape[2]}"
            + ("-constrained" if constrained else ""),
        )
        for seed in SEEDS
        for shape in SHAPES
        for constrained in constrained_options
    ]
