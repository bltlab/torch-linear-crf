"""AllenNLP's real ConditionalRandomField, fetched at a pinned commit.

`pip install allennlp` fails on a current Python (v2.8.0 depends on
`checklist==0.0.11`, whose build shells out to `pip install jupyter`), so
`load` fetches conditional_random_field.py verbatim from github.com/allenai/allennlp
(Apache License 2.0) at COMMIT and runs it against a stub of the only two
`allennlp.nn.util` functions it imports (`logsumexp`, `viterbi_decode`,
extracted verbatim) and a one-line `ConfigurationError`. None of AllenNLP's
CRF logic is reimplemented.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import urllib.request
from pathlib import Path

# The last commit on allenai/allennlp's main branch; the repository is archived.
COMMIT = "80fb6061e568cb9d6ab5d45b661e86eb61b92c82"
RAW = f"https://raw.githubusercontent.com/allenai/allennlp/{COMMIT}/allennlp"

UTIL_STUB = '''
"""Two functions extracted verbatim from allennlp/nn/util.py, Apache License
2.0 -- everything conditional_random_field.py needs from allennlp.nn.util."""
import logging
import math

import torch

from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


def logsumexp(tensor, dim=-1, keepdim=False):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(
    tag_sequence,
    transition_matrix,
    tag_observations=None,
    allowed_start_transitions=None,
    allowed_end_transitions=None,
    top_k=None,
):
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f"top_k must be either None or an integer >=1. Instead received {top_k}")

    sequence_length, num_tags = list(tag_sequence.size())
    has_start_end_restrictions = (
        allowed_end_transitions is not None or allowed_start_transitions is not None
    )
    if has_start_end_restrictions:
        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)
        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix
        allowed_start_transitions = torch.cat(
            [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        allowed_end_transitions = torch.cat(
            [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        new_transition_matrix[-2, :] = allowed_start_transitions
        new_transition_matrix[-1, :] = -math.inf
        new_transition_matrix[:, -1] = allowed_end_transitions
        new_transition_matrix[:, -2] = -math.inf
        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError(
                "Observations were provided, but they were not the same length "
                "as the sequence. Found sequence of length: {} and evidence: {}".format(
                    sequence_length, tag_observations
                )
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []
    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        observation = tag_observations[timestep]
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning(
                    "The pairwise potential between tags you have passed as "
                    "observations is extremely unlikely. Double check your evidence "
                    "or transition potentials!"
                )
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        viterbi_path.reverse()
        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]
    return viterbi_paths, viterbi_scores
'''

CHECKS_STUB = "class ConfigurationError(Exception):\n    pass\n"


def load(root: Path):
    """Build the stub package under `root` and return AllenNLP's CRF module."""
    pkg = root / "allennlp"
    crf_dir = pkg / "modules" / "conditional_random_field"
    for directory in (pkg / "common", pkg / "nn", crf_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for init in (pkg, pkg / "common", pkg / "nn", pkg / "modules"):
        (init / "__init__.py").write_text("")
    (pkg / "common" / "checks.py").write_text(CHECKS_STUB)
    (pkg / "nn" / "util.py").write_text(textwrap.dedent(UTIL_STUB))
    source = urllib.request.urlopen(
        f"{RAW}/modules/conditional_random_field/conditional_random_field.py"
    ).read()
    (crf_dir / "conditional_random_field.py").write_bytes(source)
    (crf_dir / "__init__.py").write_text("")
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(
            "allennlp.modules.conditional_random_field.conditional_random_field"
        )
    finally:
        sys.path.remove(str(root))
