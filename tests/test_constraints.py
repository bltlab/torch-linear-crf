"""Tests for transition constraints derived from a tag encoding."""

from __future__ import annotations

import itertools

import pytest
import torch

from torch_linear_crf import LinearChainCRF
from torch_linear_crf.constraints import ENCODINGS, encoding_constraints

LABELS = ["O", "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER"]


def _invalid_bio(labels: list[str]) -> bool:
    previous = "O"
    for label in labels:
        if label.startswith("I-") and not (previous != "O" and previous[2:] == label[2:]):
            return True
        previous = label
    return False


def test_bio_constraints_rules() -> None:
    start_ok, transition_ok, end_ok = encoding_constraints(LABELS, "BIO")
    index = {label: i for i, label in enumerate(LABELS)}
    assert not start_ok[index["I-LOC"]] and start_ok[index["B-LOC"]] and start_ok[index["O"]]
    assert bool(end_ok.all())  # a mention may run to the end of the sentence
    assert transition_ok[index["B-LOC"], index["I-LOC"]]
    assert transition_ok[index["I-LOC"], index["I-LOC"]]
    assert not transition_ok[index["O"], index["I-LOC"]]
    assert not transition_ok[index["B-ORG"], index["I-LOC"]]
    assert transition_ok[index["I-LOC"], index["B-ORG"]]
    for bad in ["LOC", "S-LOC", "O-LOC", "B-", "-LOC", "B"]:
        with pytest.raises(ValueError, match="not valid under encoding"):
            encoding_constraints(["O", bad], "BIO")
    with pytest.raises(ValueError, match="unknown encoding"):
        encoding_constraints(["O"], "BMES")


def test_constrained_decode_never_emits_invalid_bio() -> None:
    """With adversarial transitions, unconstrained decoding produces invalid BIO."""
    torch.manual_seed(5)
    constrained = LinearChainCRF(len(LABELS), constraints=encoding_constraints(LABELS, "BIO"))
    unconstrained = LinearChainCRF(len(LABELS))
    with torch.no_grad():
        constrained.transitions.uniform_(-3, 3)
        unconstrained.transitions.copy_(constrained.transitions)
        unconstrained.start_transitions.copy_(constrained.start_transitions)
        unconstrained.end_transitions.copy_(constrained.end_transitions)

    emissions = torch.randn(64, 20, len(LABELS)) * 8.0
    mask = torch.ones(64, 20, dtype=torch.bool)
    assert not any(
        _invalid_bio([LABELS[i] for i in p]) for p in constrained.decode(emissions, mask)
    )
    # Otherwise the test would pass on a model that never wanted to break BIO.
    assert any(_invalid_bio([LABELS[i] for i in p]) for p in unconstrained.decode(emissions, mask))


def test_decode_matches_enumeration_with_constraints() -> None:
    """Viterbi must find the single highest-scoring *valid* path."""
    torch.manual_seed(2)
    labels = ["O", "B-X", "I-X"]
    crf = LinearChainCRF(3, constraints=encoding_constraints(labels, "BIO"))
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


BIOES_LABELS = ["O", "B-LOC", "E-LOC", "I-LOC", "S-LOC", "B-PER", "E-PER", "I-PER", "S-PER"]


def test_bioes_constraints_rules() -> None:
    """BIOES restricts sequence ends, which BIO does not."""
    start_ok, transition_ok, end_ok = encoding_constraints(BIOES_LABELS, "BIOES")
    index = {label: i for i, label in enumerate(BIOES_LABELS)}
    assert [label for label in BIOES_LABELS if start_ok[index[label]]] == [
        "O",
        "B-LOC",
        "S-LOC",
        "B-PER",
        "S-PER",
    ]
    # Under BIO every label may end a sequence, under BIOES only E-X, S-X and O may.
    assert [label for label in BIOES_LABELS if end_ok[index[label]]] == [
        "O",
        "E-LOC",
        "S-LOC",
        "E-PER",
        "S-PER",
    ]
    assert transition_ok[index["B-LOC"], index["I-LOC"]]
    assert transition_ok[index["B-LOC"], index["E-LOC"]]
    assert not transition_ok[index["B-LOC"], index["O"]]
    assert not transition_ok[index["B-LOC"], index["E-PER"]]
    assert transition_ok[index["E-LOC"], index["B-PER"]]
    assert transition_ok[index["S-LOC"], index["S-PER"]]


def test_bioes_constrained_decode_is_always_valid() -> None:
    """Decoded BIOES must be scoreable, which for BIOES means valid: it has no repair."""
    seqscore_encoding = pytest.importorskip("seqscore.encoding")
    torch.manual_seed(9)
    encoding = seqscore_encoding.get_encoding("BIOES")
    crf = LinearChainCRF(len(BIOES_LABELS), constraints=encoding_constraints(BIOES_LABELS, "BIOES"))
    with torch.no_grad():
        crf.transitions.uniform_(-3, 3)
    emissions = torch.randn(64, 16, len(BIOES_LABELS)) * 8.0
    mask = torch.ones(64, 16, dtype=torch.bool)
    for path in crf.decode(emissions, mask):
        labels = [BIOES_LABELS[i] for i in path]
        # decode_labels raises on an invalid sequence, so this is the real check.
        encoding.decode_labels(labels)


LABELS_BY_ENCODING = {
    "BIO": ["O", "B-LOC", "I-LOC", "B-PER", "I-PER"],
    "BIOES": BIOES_LABELS,
    "BILOU": ["O", "B-LOC", "I-LOC", "L-LOC", "U-LOC", "B-PER", "I-PER", "L-PER", "U-PER"],
    "IO": ["O", "I-LOC", "I-PER"],
    "IOB": ["O", "B-LOC", "I-LOC", "B-PER", "I-PER"],
    "IOB-lenient": ["O", "B-LOC", "I-LOC", "B-PER", "I-PER"],
}


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_constraints_match_seqscore(encoding: str) -> None:
    """Every start, end and transition agrees with seqscore's validator."""
    seqscore_encoding = pytest.importorskip("seqscore.encoding")
    try:
        enc = seqscore_encoding.get_encoding(encoding)
    except ValueError:
        pytest.skip(f"installed seqscore has no {encoding}")
    labels = LABELS_BY_ENCODING[encoding]
    parsed = [enc.split_label(label) for label in labels]
    outside = enc.dialect.outside
    start_ok, transition_ok, end_ok = encoding_constraints(labels, encoding)
    assert start_ok.tolist() == [enc.is_valid_transition(outside, None, *p) for p in parsed]
    assert end_ok.tolist() == [enc.is_valid_transition(*p, outside, None) for p in parsed]
    assert transition_ok.tolist() == [
        [enc.is_valid_transition(*first, *second) for second in parsed] for first in parsed
    ]


@pytest.mark.parametrize("encoding", ["IO", "IOB-lenient"])
def test_unconstrained_encodings(encoding: str) -> None:
    """IO and IOB-lenient allow every transition."""
    for mask in encoding_constraints(LABELS_BY_ENCODING[encoding], encoding):
        assert bool(mask.all())


def test_iob_rules() -> None:
    """IOB (IOB1) uses B-X only directly after a mention of the same type."""
    labels = LABELS_BY_ENCODING["IOB"]
    start_ok, transition_ok, end_ok = encoding_constraints(labels, "IOB")
    index = {label: i for i, label in enumerate(labels)}
    assert [label for label in labels if start_ok[index[label]]] == ["O", "I-LOC", "I-PER"]
    assert bool(end_ok.all())
    assert transition_ok[index["I-LOC"], index["B-LOC"]]
    assert transition_ok[index["B-LOC"], index["B-LOC"]]
    assert not transition_ok[index["I-LOC"], index["B-PER"]]
    assert not transition_ok[index["O"], index["B-LOC"]]
    assert transition_ok[index["I-LOC"], index["I-PER"]]
