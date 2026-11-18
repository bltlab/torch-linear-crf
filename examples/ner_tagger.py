"""A complete BiLSTM-CRF named entity tagger with torch-linear-crf.

    python examples/ner_tagger.py              # CPU, about 10 seconds
    python examples/ner_tagger.py --device cuda

The data is a small synthetic corpus generated below, so the example runs
without downloads; replace `make_corpus` with your own reader of
(tokens, labels) pairs. It shows the parts a real tagger needs: a label set
and its constraints, padding and the mask, the training loss, constrained
decoding back to label strings, and per-token confidence from the marginals.
"""

from __future__ import annotations

import argparse
import random

import torch
from torch import nn

from torch_linear_crf import LinearChainCRF
from torch_linear_crf.constraints import encoding_constraints

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def make_corpus(
    n: int, rng: random.Random, unseen: float = 0.0
) -> list[tuple[list[str], list[str]]]:
    """Synthetic sentences: people work for organizations and travel to places.

    With `unseen`, that fraction of names and places comes from pools the
    training data never uses, as real test data would.
    """
    first = ["Ada", "Alan", "Grace", "Edsger", "Barbara", "Donald", "Frances", "John"]
    last = ["Lovelace", "Turing", "Hopper", "Dijkstra", "Liskov", "Knuth", "Allen", "McCarthy"]
    orgs = [["Acme", "Corp"], ["Initech"], ["Globex", "Inc"], ["Hooli"], ["Stark", "Industries"]]
    places = [["Paris"], ["New", "York"], ["Boston"], ["Rio", "de", "Janeiro"], ["Oslo"]]
    new_first, new_last = ["Katherine", "Radia", "Ken"], ["Johnson", "Perlman", "Thompson"]
    new_orgs, new_places = [["Umbrella", "Corp"], ["Wayne", "Enterprises"]], [["Lagos"], ["Kyoto"]]
    filler = ["yesterday", "quietly", "again", "finally", "today"]

    def pick(seen: list, new: list):
        return rng.choice(new if rng.random() < unseen else seen)

    def mention(tokens: list[str], kind: str) -> tuple[list[str], list[str]]:
        return tokens, [f"B-{kind}"] + [f"I-{kind}"] * (len(tokens) - 1)

    corpus = []
    for _ in range(n):
        person = mention([pick(first, new_first), pick(last, new_last)], "PER")
        parts = [person]
        roll = rng.random()
        if roll < 0.35:
            parts += [(["works", "for"], ["O", "O"]), mention(pick(orgs, new_orgs), "ORG")]
        elif roll < 0.7:
            parts += [(["flew", "to"], ["O", "O"]), mention(pick(places, new_places), "LOC")]
        else:
            # "visited" takes either type, so an unseen name after it is ambiguous.
            kind, seen, new = rng.choice([("ORG", orgs, new_orgs), ("LOC", places, new_places)])
            parts += [(["visited"], ["O"]), mention(pick(seen, new), kind)]
        parts.append(([rng.choice(filler)], ["O"]))
        tokens = [token for words, _ in parts for token in words]
        labels = [label for _, tags in parts for label in tags]
        corpus.append((tokens, labels))
    return corpus


class Tagger(nn.Module):
    """Word embeddings -> BiLSTM -> per-token label scores (emissions) -> CRF."""

    def __init__(self, vocab_size: int, labels: list[str], dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.emit = nn.Linear(2 * dim, len(labels))
        # Constraints come from the encoding's own validity rules, so decode can
        # never produce an invalid BIO sequence (an I-X without a B-X before it).
        self.crf = LinearChainCRF(len(labels), constraints=encoding_constraints(labels, "BIO"))

    def emissions(self, words: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(self.embed(words))
        return self.emit(hidden)  # (batch, length, num_labels)

    def loss(self, words: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # log_likelihood is per sentence; average over tokens for a stable scale.
        return -self.crf.log_likelihood(self.emissions(words), tags, mask).sum() / mask.sum()

    @torch.no_grad()
    def predict(self, words: torch.Tensor, mask: torch.Tensor):
        emissions = self.emissions(words)
        paths = self.crf.decode(emissions, mask)  # one list of label ids per sentence
        marginals = self.crf.marginals(emissions, mask)  # (batch, length, num_labels)
        return paths, marginals


def batch(
    examples,
    word_ids: dict[str, int],
    label_ids: dict[str, int],
    device: str,
    word_dropout: float = 0.0,
):
    """Pad to the longest sentence; the mask marks each row's real tokens (a prefix).

    `word_dropout` replaces that fraction of words with the unknown id, so the
    model learns to tag from context words it has never seen.
    """
    length = max(len(tokens) for tokens, _ in examples)
    words = torch.zeros(len(examples), length, dtype=torch.long)
    tags = torch.zeros(len(examples), length, dtype=torch.long)  # padding holds any valid id
    mask = torch.zeros(len(examples), length, dtype=torch.bool)
    for row, (tokens, labels) in enumerate(examples):
        ids = [word_ids.get(t, 1) if random.random() >= word_dropout else 1 for t in tokens]
        words[row, : len(tokens)] = torch.tensor(ids)
        tags[row, : len(labels)] = torch.tensor([label_ids[label] for label in labels])
        mask[row, : len(tokens)] = True
    return words.to(device), tags.to(device), mask.to(device)


def spans(labels: list[str]) -> set[tuple[int, int, str]]:
    """(start, end, type) for each BIO mention; constrained decoding makes this exact."""
    found, start = set(), None
    for i, label in enumerate([*labels, "O"]):
        if start is not None and not label.startswith("I-"):
            found.add((start, i, labels[start][2:]))
            start = None
        if label.startswith("B-"):
            start = i
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)  # word dropout
    train, test = make_corpus(2000, rng), make_corpus(300, rng, unseen=0.3)
    vocab = sorted({token for tokens, _ in train for token in tokens})
    word_ids = {token: i + 2 for i, token in enumerate(vocab)}  # 0 = padding, 1 = unknown
    label_ids = {label: i for i, label in enumerate(LABELS)}

    model = Tagger(len(vocab) + 2, LABELS).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        total = 0.0
        for start in range(0, len(train), 32):
            examples = train[start : start + 32]
            words, tags, mask = batch(examples, word_ids, label_ids, args.device, word_dropout=0.1)
            loss = model.loss(words, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: mean loss {total / (len(train) / 32):.4f}")

    model.eval()
    gold_spans = predicted_spans = correct = 0
    for start in range(0, len(test), 64):
        examples = test[start : start + 64]
        words, _, mask = batch(examples, word_ids, label_ids, args.device)
        paths, _ = model.predict(words, mask)
        for (_, gold), path in zip(examples, paths, strict=True):
            predicted = spans([LABELS[i] for i in path])
            expected = spans(gold)
            gold_spans += len(expected)
            predicted_spans += len(predicted)
            correct += len(predicted & expected)
    precision, recall = correct / max(predicted_spans, 1), correct / max(gold_spans, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"test mention F1 {f1:.3f} (precision {precision:.3f}, recall {recall:.3f})")

    # Every name here is unseen in training. "flew to" says Kyoto is a place;
    # after "visited", Lagos could be an organization or a place, and the
    # marginals show the model's uncertainty.
    for tokens in (
        ["Radia", "Perlman", "flew", "to", "Kyoto", "again"],
        ["Ken", "Thompson", "visited", "Lagos", "today"],
    ):
        words, _, mask = batch([(tokens, ["O"] * len(tokens))], word_ids, label_ids, args.device)
        paths, marginals = model.predict(words, mask)
        for position, (token, label_id) in enumerate(zip(tokens, paths[0], strict=True)):
            confidence = marginals[0, position, label_id].item()
            print(f"{token:9s} {LABELS[label_id]:6s} p={confidence:.3f}")
        print()


if __name__ == "__main__":
    main()
