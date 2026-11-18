"""Write tests/reference/reference.json from the pure-Python oracle.

    python tests/reference/make_reference.py

Inputs come from Python's `random` with fixed seeds and the expected values
from `oracle`, so neither depends on torch. Rerun only when the documented
semantics change, and review the diff of reference.json when you do.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import oracle

OUT = Path(__file__).with_name("reference.json")
NEG_INF = float("-inf")


def make_case(seed: int, num_tags: int, length: int, batch: int, kind: str) -> dict:
    rng = random.Random(seed)
    params = {
        "start": [rng.gauss(0, 1) for _ in range(num_tags)],
        "transitions": [[rng.gauss(0, 1) for _ in range(num_tags)] for _ in range(num_tags)],
        "end": [rng.gauss(0, 1) for _ in range(num_tags)],
        "start_ok": [True] * num_tags,
        "transition_ok": [[True] * num_tags for _ in range(num_tags)],
        "end_ok": [True] * num_tags,
    }
    if kind != "plain":
        # Forbid some, never all, of each kind, so every row has a valid path.
        params["start_ok"][num_tags - 1] = False
        params["end_ok"][1] = False
        for i in range(num_tags):
            params["transition_ok"][i][(i + 1) % num_tags] = False
    lengths = [length] + [rng.randint(1, length) for _ in range(batch - 1)]
    emissions = [
        [[rng.gauss(0, 3) for _ in range(num_tags)] for _ in range(length)] for _ in range(batch)
    ]
    for b in range(batch):
        for t in range(lengths[b], length):
            emissions[b][t] = [NEG_INF] * num_tags  # -inf padding
    if kind == "neg_inf":
        # Tag 0 masked out with -inf at every valid position of row 0.
        for t in range(lengths[0]):
            emissions[0][t][0] = NEG_INF
    tags = [
        [rng.randrange(1 if kind == "neg_inf" else 0, num_tags) for _ in range(length)]
        for _ in range(batch)
    ]
    allowed = [
        [[rng.random() < 0.6 for _ in range(num_tags)] for _ in range(length)] for _ in range(batch)
    ]
    for b in range(batch):
        for t in range(length):
            allowed[b][t][rng.randrange(1, num_tags)] = True  # at least one, never only tag 0
    log_weights = [
        [[rng.gauss(-1, 1) for _ in range(num_tags)] for _ in range(length)] for _ in range(batch)
    ]
    if kind == "neg_inf":
        for t in range(length):
            log_weights[0][t][num_tags - 1] = NEG_INF  # a zero teacher weight
    temperature = 1.0 if seed % 2 else 1.7
    k = 4

    expected = {"rows": []}
    for b in range(batch):
        n = lengths[b]
        e, tg, al, lw = emissions[b][:n], tags[b][:n], allowed[b][:n], log_weights[b][:n]
        row = {}
        for constrain_loss in (False, True):
            key = "constrained" if constrain_loss else "unconstrained"
            row[f"log_likelihood_{key}"] = oracle.log_likelihood(params, e, tg, constrain_loss)
            row[f"partial_log_likelihood_{key}"] = oracle.partial_log_likelihood(
                params, e, al, lw, constrain_loss
            )
            marg, log_z = oracle.marginals(params, e, constrain_loss, temperature)
            row[f"marginals_{key}"] = marg
            row[f"log_z_{key}"] = log_z
        ranked = oracle.ranked_paths(params, e)
        row["decode"] = ranked[0]
        row["top_k"] = ranked[:k]
        ranked_allowed = oracle.ranked_paths(params, e, al)
        row["decode_allowed"] = ranked_allowed[0]
        row["top_k_allowed"] = ranked_allowed[:k]
        expected["rows"].append(row)

    return {
        "name": f"{kind}-C{num_tags}-T{length}-B{batch}-s{seed}",
        "params": params,
        "lengths": lengths,
        "emissions": emissions,
        "tags": tags,
        "allowed": allowed,
        "log_weights": log_weights,
        "temperature": temperature,
        "k": k,
        "expected": expected,
    }


def main() -> None:
    cases = []
    seed = 0
    for kind in ("plain", "constrained", "neg_inf"):
        for num_tags, length, batch in ((3, 1, 2), (3, 5, 3), (4, 4, 3), (5, 3, 2)):
            cases.append(make_case(seed, num_tags, length, batch, kind))
            seed += 1
    OUT.write_text(json.dumps({"forbidden": oracle.FORBIDDEN, "cases": cases}, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
