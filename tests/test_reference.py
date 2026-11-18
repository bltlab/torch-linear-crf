"""Check every public method against tests/reference/reference.json.

The expected values come from a pure-Python brute-force oracle
(tests/reference/oracle.py), so they hold for every torch version; the
compatibility suite (`nox -s compat`) runs this file on each torch minor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_linear_crf import LinearChainCRF

REFERENCE = json.loads((Path(__file__).parent / "reference" / "reference.json").read_text())
CASES = REFERENCE["cases"]
# (rtol, atol) per dtype. fp32 values near FORBIDDEN (-1e4) carry ~1e-3 of
# rounding, hence the absolute term.
TOLERANCE = {torch.float64: (1e-10, 1e-8), torch.float32: (1e-5, 2e-3)}
MARGINAL_ATOL = {torch.float64: 1e-10, torch.float32: 1e-5}
DEVICES = [pytest.param("cpu"), pytest.param("cuda", marks=pytest.mark.cuda)]


def _build(
    case: dict, constrain_loss: bool, dtype: torch.dtype, device: str, parallel_scan: bool = False
):
    params = case["params"]
    constraints = tuple(
        torch.tensor(params[key]) for key in ("start_ok", "transition_ok", "end_ok")
    )
    crf = LinearChainCRF(
        len(params["start"]),
        constraints=constraints,
        constrain_loss=constrain_loss,
        parallel_scan=parallel_scan,
    )
    crf = crf.to(device=device, dtype=dtype)
    with torch.no_grad():
        for name in ("start", "transitions", "end"):
            target = (
                crf.transitions if name == "transitions" else getattr(crf, name + "_transitions")
            )
            target.copy_(torch.tensor(params[name], dtype=dtype))
    lengths = torch.tensor(case["lengths"])
    length = len(case["emissions"][0])
    inputs = {
        "emissions": torch.tensor(case["emissions"], dtype=dtype, device=device),
        "mask": (torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)).to(device),
        "tags": torch.tensor(case["tags"], device=device),
        "allowed": torch.tensor(case["allowed"], device=device),
        "log_weights": torch.tensor(case["log_weights"], dtype=dtype, device=device),
    }
    return crf, inputs


def _close(got: float, expected: float, dtype: torch.dtype) -> bool:
    rtol, atol = TOLERANCE[dtype]
    return abs(got - expected) <= atol + rtol * abs(expected)


@pytest.mark.parametrize("parallel_scan", [False, True], ids=["loop", "scan"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_matches_reference(
    case: dict, dtype: torch.dtype, device: str, parallel_scan: bool
) -> None:
    rows = case["expected"]["rows"]
    for constrain_loss in (False, True):
        key = "constrained" if constrain_loss else "unconstrained"
        crf, x = _build(case, constrain_loss, dtype, device, parallel_scan)
        ll = crf.log_likelihood(x["emissions"], x["tags"], x["mask"]).tolist()
        pll = crf.partial_log_likelihood(
            x["emissions"], x["allowed"], x["mask"], x["log_weights"]
        ).tolist()
        _, _, log_z = crf.forward_backward(
            x["emissions"], x["mask"], constrain_loss, case["temperature"]
        )
        marginals = crf.marginals(x["emissions"], x["mask"], constrain_loss, case["temperature"])
        for b, row in enumerate(rows):
            n = case["lengths"][b]
            assert _close(ll[b], row[f"log_likelihood_{key}"], dtype), (key, b, ll[b])
            assert _close(pll[b], row[f"partial_log_likelihood_{key}"], dtype), (key, b, pll[b])
            assert _close(float(log_z[b]), row[f"log_z_{key}"], dtype), (key, b)
            expected = torch.tensor(row[f"marginals_{key}"], dtype=torch.float64)
            atol = MARGINAL_ATOL[dtype]
            if dtype == torch.float32 and row[f"log_z_{key}"] < -1e3:
                # Every path crosses a FORBIDDEN penalty, so alpha, beta and
                # log Z sit near -1e4, where fp32 resolves only ~1e-3.
                atol = 2e-3
            torch.testing.assert_close(marginals[b, :n].double().cpu(), expected, rtol=0, atol=atol)
            assert torch.all(marginals[b, n:] == 0)

    crf, x = _build(case, False, dtype, device)
    for allowed, suffix in ((None, ""), (x["allowed"], "_allowed")):
        decoded = crf.decode_with_scores(x["emissions"], x["mask"], allowed)
        top = crf.decode_top_k(x["emissions"], x["mask"], case["k"], allowed)
        assert crf.decode(x["emissions"], x["mask"], allowed) == [p for p, _ in decoded]
        for b, row in enumerate(rows):
            path, score = row[f"decode{suffix}"]
            assert decoded[b][0] == path, (suffix, b)
            assert _close(decoded[b][1], score, dtype), (suffix, b)
            expected_top = row[f"top_k{suffix}"]
            assert [p for p, _ in top[b]] == [p for p, _ in expected_top], (suffix, b)
            for (_, got), (_, want) in zip(top[b], expected_top, strict=True):
                assert _close(got, want, dtype), (suffix, b)


@pytest.mark.parametrize("parallel_scan", [False, True], ids=["loop", "scan"])
@pytest.mark.parametrize("device", DEVICES)
def test_compiled_matches_reference_and_eager(device: str, parallel_scan: bool) -> None:
    """torch.compile against the oracle, gradients against eager: on torch
    2.4-2.12 inductor once got this gradient wrong while values matched."""
    case = next(c for c in CASES if c["name"].startswith("neg_inf-C4"))
    crf, x = _build(case, True, torch.float32, device, parallel_scan)
    rows = case["expected"]["rows"]

    def losses(emissions):
        return (
            crf.log_likelihood(emissions, x["tags"], x["mask"]),
            crf.partial_log_likelihood(emissions, x["allowed"], x["mask"], x["log_weights"]),
            crf.log_marginals(emissions, x["mask"], True, case["temperature"])[x["mask"]],
        )

    eager_in = x["emissions"].clone().requires_grad_()
    eager = losses(eager_in)
    sum(v.sum() for v in eager).backward()
    torch._dynamo.reset()
    compiled_in = x["emissions"].clone().requires_grad_()
    compiled = torch.compile(losses)(compiled_in)
    sum(v.sum() for v in compiled).backward()

    for b, row in enumerate(rows):
        assert _close(float(compiled[0][b]), row["log_likelihood_constrained"], torch.float32)
        assert _close(
            float(compiled[1][b]), row["partial_log_likelihood_constrained"], torch.float32
        )
    for got, want in zip(compiled, eager, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(compiled_in.grad, eager_in.grad, rtol=1e-5, atol=1e-4)
