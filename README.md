# torch-linear-crf

A new actively maintained linear CRF package for PyTorch.

```bash
pip install torch-linear-crf
```

## Comparison with other CRF packages

| | **torch-linear-crf** | pytorch-crf | AllenNLP | Notes |
|---|---|---|---|---|
| Forward algorithm + Viterbi, batch-vectorized | ✅ | ✅ | ✅ | |
| Separate start/end/transition params | ✅ | ✅ | ✅ | |
| Option to disable start/end transitions | ✅ | ❌ | ✅ | |
| Decode returns score with path | ✅ | ❌ | ✅ | |
| Top-*k* Viterbi | ✅ | ❌ | ✅ | |
| Partial evidence-constrained decode | ✅ | ❌ | ⚠️ | Restrict any position to a set of allowed tags when decoding. AllenNLP has this only in its `viterbi_decode` utility, with one fixed tag per position, and its CRF's `viterbi_tags` doesn't expose it. |
| Full shape validation with clear errors | ✅ | ✅ | ❌ | AllenNLP checks no input shapes; a mismatch fails inside the recursion or broadcasts silently. |
| bf16-autocast correctness guard | ✅ | ❌ | ❌ | Under bf16 autocast the recursion still runs in fp32. Run in bf16, it measured 6.9 nats off fp32, against 0.36 for bf16 emissions into an fp32 recursion. |
| Hard transition constraints | ✅ | ❌ | ✅ | Forbidden transitions, such as O to I-X in BIO, are never decoded. |
| Constraints optionally affect training loss | ✅ | N/A | ❌ | Optionally normalize over valid sequences only. AllenNLP applies constraints only when decoding. |
| `torch.jit.script` works (current PyTorch) | ✅ | ✅ | ❌ | |
| `torch.compile` works (current PyTorch) | ✅ | ✅ | ⚠️ | AllenNLP's likelihood compiles; `viterbi_tags` runs eagerly. |
| Posterior/marginal tag probabilities | ✅ | ❌ | ❌ | Per-token tag probabilities, for example as confidence scores. |
| Installable on current Python without extra work | ✅ | ✅ | ❌ | AllenNLP's last release, 2.10.1, requires torch < 1.13. |
| No framework coupling | ✅ | ✅ | ❌ | AllenNLP's CRF ships inside the full AllenNLP library. |
| Actively maintained | ✅ | ❌ | ❌ | pytorch-crf's last release was February 2019. AllenNLP's repository is archived; its last release was October 2022. |

## Usage

### A BiLSTM-CRF tagger

```python
import torch
from torch import nn
from torch_linear_crf import LinearChainCRF
from torch_linear_crf.constraints import encoding_constraints

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]


class Tagger(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.emit = nn.Linear(2 * dim, len(LABELS))
        # Decoding can never produce an invalid BIO sequence.
        self.crf = LinearChainCRF(len(LABELS), constraints=encoding_constraints(LABELS, "BIO"))

    def emissions(self, words):
        return self.emit(self.lstm(self.embed(words))[0])  # (batch, length, labels)


model = Tagger(vocab_size=1000)
words = torch.randint(1, 1000, (2, 6))  # two sentences padded to length 6
tags = torch.tensor([[1, 2, 0, 3, 0, 0], [0, 3, 4, 0, 0, 0]])  # any valid id at padding
mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]).bool()  # real tokens first

loss = -model.crf.log_likelihood(model.emissions(words), tags, mask).sum() / mask.sum()
loss.backward()

with torch.no_grad():
    emissions = model.emissions(words)
    paths = model.crf.decode(emissions, mask)  # label ids per sentence, cut to length
    labels = [[LABELS[i] for i in path] for path in paths]
    confidence = model.crf.marginals(emissions, mask)  # p(label | sentence) per token
```

### The API at a glance

```python
import torch
from torch_linear_crf import LinearChainCRF

num_tags = 5
crf = LinearChainCRF(num_tags)

emissions = torch.randn(8, 20, num_tags)  # (batch, length, tags)
tags = torch.randint(0, num_tags, (8, 20))
mask = torch.ones(8, 20, dtype=torch.bool)  # contiguous valid prefix per row

loss = -crf.log_likelihood(emissions, tags, mask).sum()
loss.backward()

decoded = crf.decode(emissions, mask)  # list[list[int]], one per row

# Or with each path's Viterbi score:
decoded_with_scores = crf.decode_with_scores(emissions, mask)  # list[tuple[list[int], float]]

# The 5 best paths per row, best first:
best_five = crf.decode_top_k(emissions, mask, 5)  # list[list[tuple[list[int], float]]]

# The best paths as device tensors, with no host sync (-1 at padding):
tags, scores = crf.decode_to_tensor(emissions, mask)  # (batch, length) long, (batch,)
```

### Constraints from a tag encoding

```python
from torch_linear_crf import LinearChainCRF
from torch_linear_crf.constraints import encoding_constraints

labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
crf = LinearChainCRF(len(labels), constraints=encoding_constraints(labels, "BIO"))
# crf.decode(...) now never emits an invalid BIO sequence.
```

### Without start/end transitions

```python
crf = LinearChainCRF(num_tags, include_start_end_transitions=False)
```

### Parallel scan

```python
crf = LinearChainCRF(num_tags, parallel_scan=True)  # log_likelihood, partial_log_likelihood
```

### Posterior marginals

```python
probs = crf.marginals(emissions, mask)  # (batch, length, tags), zero at padding
log_probs = crf.log_marginals(emissions, mask, temperature=2.0)
alpha, beta, log_z = crf.forward_backward(emissions, mask)

# forward_backward also accepts scores from outside a CRF, e.g. the
# log-probabilities of a per-token softmax tagger, with invalid BIO
# transitions forbidden:
from torch_linear_crf import FORBIDDEN, forward_backward

penalty = [
    torch.zeros(ok.shape).masked_fill(~ok, FORBIDDEN) for ok in encoding_constraints(labels, "BIO")
]
alpha, beta, log_z = forward_backward(token_logits.log_softmax(-1), mask, *penalty)
```

### TorchScript and torch.compile

```python
scripted = torch.jit.script(crf)
compiled = torch.compile(crf.log_likelihood)
```
