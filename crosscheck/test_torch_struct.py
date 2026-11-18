"""torch-linear-crf against torch-struct 0.5's LinearChainCRF.

torch-struct takes per-edge log potentials, `[b, n, next, previous]` for
n = 0..T-2, so a row needs at least two positions; start, end and the
constraint penalties are folded into them. Our constrained potentials go in
as they are, so this checks the constrained distribution too.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch_struct
from cases import ATOL, RTOL, case_params, make_case, ours


def edge_potentials(emissions, start, transitions, end, lengths):
    batch, _, num_tags = emissions.shape
    potentials = transitions.t().view(1, 1, num_tags, num_tags) + emissions[:, 1:].unsqueeze(3)
    potentials = potentials.clone()
    potentials[:, 0] += (start + emissions[:, 0]).view(batch, 1, num_tags)
    potentials[torch.arange(batch), lengths - 2] += end.view(num_tags, 1)
    return potentials


def paths_from_edges(edges, lengths):
    """Tag paths from one-hot edge structures (..., T-1, next, previous)."""
    first = edges[..., 0, :, :].sum(-2).argmax(-1)
    rest = edges.sum(-1).argmax(-1)
    paths = torch.cat([first.unsqueeze(-1), rest], dim=-1)
    return [path[: int(n)].tolist() for path, n in zip(paths, lengths, strict=True)]


@pytest.mark.parametrize(("seed", "shape", "constrained"), case_params())
def test_partition_marginals_viterbi_top_k(seed, shape, constrained) -> None:
    num_tags, length, batch = shape
    case = make_case(seed, num_tags, length, batch, constrained, min_length=2)
    crf = ours(case)
    emissions, mask, lengths = case["emissions"], case["mask"], case["lengths"]
    start, transitions, end = (x.detach() for x in crf.potentials(constrained=True))
    dist = torch_struct.LinearChainCRF(
        edge_potentials(emissions, start, transitions, end, lengths), lengths=lengths
    )

    _, _, log_z = crf.forward_backward(emissions, mask)
    torch.testing.assert_close(log_z, dist.partition, rtol=RTOL, atol=ATOL)
    scan = ours(case, parallel_scan=True)
    scan_log_z = scan._log_partition(emissions, mask, start, transitions, end, False)
    torch.testing.assert_close(scan_log_z, dist.partition, rtol=RTOL, atol=ATOL)

    edges = dist.marginals
    node = torch.zeros_like(emissions)
    node[:, 0] = edges[:, 0].sum(1)
    node[:, 1:] = edges.sum(3)
    marginals = crf.marginals(emissions, mask)
    torch.testing.assert_close(marginals[mask], node[mask], rtol=RTOL, atol=ATOL)

    assert crf.decode(emissions, mask) == paths_from_edges(dist.argmax, lengths)

    k = 3
    top_edges = dist.topk(k)  # (k, B, T-1, C, C)
    potentials = edge_potentials(emissions, start, transitions, end, lengths)
    top_scores = (top_edges * potentials).sum((-1, -2, -3))  # (k, B)
    ours_top = crf.decode_top_k(emissions, mask, k)
    for b in range(batch):
        theirs_paths = paths_from_edges(top_edges[:, b], lengths[b].expand(k))
        theirs_scores = top_scores[:, b].tolist()
        assert [s for _, s in ours_top[b]] == pytest.approx(theirs_scores, rel=RTOL, abs=ATOL)
        # Paths only where scores are distinct: ties may be ordered either way.
        gaps = [a - c for a, c in itertools.pairwise(theirs_scores)]
        if all(gap > 1e-9 for gap in gaps):
            assert [p for p, _ in ours_top[b]] == theirs_paths
