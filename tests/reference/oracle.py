"""Brute-force linear-chain CRF in pure Python floats, for the reference file.

Enumerates every tag path, so it only suits a handful of tags and positions.
It uses no torch, so the expected values it produces do not depend on the
torch version under test. It implements the semantics `LinearChainCRF`
documents: penalties of FORBIDDEN on disallowed start/transition/end tags,
emissions below FORBIDDEN raised to FORBIDDEN in the likelihoods and the
marginals (after any temperature), disallowed emissions replaced by
FORBIDDEN in `partial_log_likelihood`'s numerator, and -inf (no floor) for
tags `decode` may not choose.
"""

from __future__ import annotations

import itertools
import math

FORBIDDEN = -1e4
NEG_INF = float("-inf")


def logsumexp(values: list[float]) -> float:
    top = max(values)
    if top == NEG_INF:
        return NEG_INF
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def potentials(params: dict, constrained: bool) -> tuple[list, list, list]:
    """(start, transitions, end), with the constraint penalties if `constrained`."""
    start = list(params["start"])
    transitions = [list(row) for row in params["transitions"]]
    end = list(params["end"])
    if constrained:
        num_tags = len(start)
        for i in range(num_tags):
            if not params["start_ok"][i]:
                start[i] += FORBIDDEN
            if not params["end_ok"][i]:
                end[i] += FORBIDDEN
            for j in range(num_tags):
                if not params["transition_ok"][i][j]:
                    transitions[i][j] += FORBIDDEN
    return start, transitions, end


def path_score(path, emissions, start, transitions, end) -> float:
    score = start[path[0]] + emissions[0][path[0]]
    for t in range(1, len(path)):
        score += transitions[path[t - 1]][path[t]] + emissions[t][path[t]]
    return score + end[path[-1]]


def all_paths(num_tags: int, length: int):
    return itertools.product(range(num_tags), repeat=length)


def floor(emissions: list[list[float]]) -> list[list[float]]:
    return [[max(x, FORBIDDEN) for x in row] for row in emissions]


def log_partition(emissions, start, transitions, end) -> float:
    num_tags = len(start)
    return logsumexp(
        [
            path_score(p, emissions, start, transitions, end)
            for p in all_paths(num_tags, len(emissions))
        ]
    )


def log_likelihood(params, emissions, tags, constrain_loss: bool) -> float:
    """emissions: one row's valid positions; tags: same length."""
    scores = potentials(params, constrain_loss)
    floored = floor(emissions)
    return path_score(tags, floored, *scores) - log_partition(floored, *scores)


def partial_log_likelihood(params, emissions, allowed, log_weights, constrain_loss: bool):
    scores = potentials(params, constrain_loss)
    floored = floor(emissions)
    weighted = floored
    if log_weights is not None:
        weighted = floor(
            [
                [e + w for e, w in zip(er, wr, strict=True)]
                for er, wr in zip(floored, log_weights, strict=True)
            ]
        )
    restricted = [
        [x if ok else FORBIDDEN for x, ok in zip(row, oks, strict=True)]
        for row, oks in zip(weighted, allowed, strict=True)
    ]
    return log_partition(restricted, *scores) - log_partition(floored, *scores)


def marginals(params, emissions, constrained: bool, temperature: float):
    """P(y_t = c) for each valid position t and tag c."""
    start, transitions, end = potentials(params, constrained)
    start = [x / temperature for x in start]
    transitions = [[x / temperature for x in row] for row in transitions]
    end = [x / temperature for x in end]
    scaled = floor([[x / temperature for x in row] for row in emissions])
    num_tags = len(start)
    paths = list(all_paths(num_tags, len(scaled)))
    scores = [path_score(p, scaled, start, transitions, end) for p in paths]
    log_z = logsumexp(scores)
    out = [[0.0] * num_tags for _ in scaled]
    for t in range(len(scaled)):
        for c in range(num_tags):
            terms = [s for p, s in zip(paths, scores, strict=True) if p[t] == c]
            out[t][c] = math.exp(logsumexp(terms) - log_z) if terms else 0.0
    return out, log_z


def ranked_paths(params, emissions, allowed=None) -> list[tuple[list[int], float]]:
    """Every path with a finite score under decode's semantics, best first."""
    scores = potentials(params, True)
    num_tags = len(scores[0])
    if allowed is not None:
        emissions = [
            [x if ok else NEG_INF for x, ok in zip(row, oks, strict=True)]
            for row, oks in zip(emissions, allowed, strict=True)
        ]
    ranked = []
    for p in all_paths(num_tags, len(emissions)):
        s = path_score(p, emissions, *scores)
        if s > NEG_INF:
            ranked.append((list(p), s))
    ranked.sort(key=lambda item: -item[1])
    return ranked
