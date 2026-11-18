"""Derive hard CRF transition constraints from a tag encoding.

Supported encodings: BIO, BIOES, BILOU, IO, IOB (IOB1) and IOB-lenient. The
transition rules match seqscore's (https://github.com/bltlab/seqscore);
tests/test_constraints.py checks every pair against it when seqscore is
installed.

The idea of applying hard transition constraints to a CRF, and of deriving
them from the tag encoding, follows AllenNLP's
`allennlp.modules.conditional_random_field.allowed_transitions`
(https://github.com/allenai/allennlp, Apache License 2.0; no code copied
verbatim).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

OUTSIDE = "O"
DELIM = "-"


class _Rules(NamedTuple):
    """Valid (from_state, to_state) pairs, split by whether the types match.

    The outside label has no type, so outside -> outside is a same-type pair.
    """

    states: frozenset[str]
    same_type: frozenset[tuple[str, str]]
    different_type: frozenset[tuple[str, str]]


def _bioes(end: str, single: str) -> _Rules:
    b, i, o, e, s = "B", "I", OUTSIDE, end, single
    return _Rules(
        frozenset({b, i, o, e, s}),
        frozenset({(b, i), (b, e), (i, i), (i, e), (e, b), (e, s), (s, b), (s, s), (o, o)}),
        frozenset({(e, b), (e, o), (e, s), (s, b), (s, o), (s, s), (o, b), (o, s)}),
    )


def _all_valid(states: set[str]) -> _Rules:
    pairs = frozenset((a, b) for a in states for b in states)
    return _Rules(frozenset(states), pairs, pairs)


_B, _I, _O = "B", "I", OUTSIDE
_IOB = _Rules(
    frozenset({_B, _I, _O}),
    frozenset({(_B, _B), (_B, _I), (_I, _I), (_I, _B), (_O, _O)}),
    # B-X only follows I-X or B-X: it separates two adjacent same-type mentions.
    frozenset({(_B, _I), (_B, _O), (_I, _I), (_I, _O), (_O, _I)}),
)
ENCODINGS: dict[str, _Rules] = {
    "BIO": _Rules(
        frozenset({_B, _I, _O}),
        frozenset({(_B, _I), (_B, _B), (_I, _I), (_I, _B), (_O, _O)}),
        frozenset({(_B, _B), (_B, _O), (_I, _B), (_I, _O), (_O, _B)}),
    ),
    "BIOES": _bioes("E", "S"),
    "BILOU": _bioes("L", "U"),
    "IO": _all_valid({_I, _O}),
    "IOB": _IOB,
    # IOB that also allows B anywhere, which leaves no transition invalid.
    "IOB-lenient": _all_valid({_B, _I, _O}),
}


def _split(label: str, encoding: str, rules: _Rules) -> tuple[str, str | None]:
    if label == OUTSIDE:
        return OUTSIDE, None
    state, delim, entity_type = label.partition(DELIM)
    if not delim or not entity_type or state == OUTSIDE or state not in rules.states:
        raise ValueError(f"label {label!r} is not valid under encoding {encoding!r}")
    return state, entity_type


def encoding_constraints(
    id_to_label: list[str], encoding: str = "BIO"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (start_ok, transition_ok, end_ok) boolean masks for a label set.

    Pass the result straight to `LinearChainCRF(num_tags, constraints=...)`.
    Labels are `O` or `STATE-TYPE`, e.g. `B-PER`. A sequence is treated as
    preceded and followed by `O`, so a label may start a sequence when
    `O -> label` is valid and end one when `label -> O` is: under BIOES a
    sequence may not end on `B-X` or `I-X`, while under BIO any label may.

    Raises ValueError for an unknown encoding, or a label that is not `O`
    and not `STATE-TYPE` with a state the encoding uses (`S-PER` under BIO).
    """
    rules = ENCODINGS.get(encoding)
    if rules is None:
        raise ValueError(f"unknown encoding {encoding!r}; supported: {', '.join(ENCODINGS)}")
    parsed = [_split(label, encoding, rules) for label in id_to_label]

    def ok(first: tuple[str, str | None], second: tuple[str, str | None]) -> bool:
        pairs = rules.same_type if first[1] == second[1] else rules.different_type
        return (first[0], second[0]) in pairs

    outside = (OUTSIDE, None)
    start_ok = torch.tensor([ok(outside, label) for label in parsed], dtype=torch.bool)
    end_ok = torch.tensor([ok(label, outside) for label in parsed], dtype=torch.bool)
    transition_ok = torch.tensor(
        [[ok(first, second) for second in parsed] for first in parsed], dtype=torch.bool
    )
    return start_ok, transition_ok, end_ok
