"""A linear-chain CRF layer for PyTorch.

    from torch_linear_crf import LinearChainCRF

For hard transition constraints from a tag encoding (BIO, BIOES, BILOU,
IO, IOB, IOB-lenient), use `torch_linear_crf.constraints.encoding_constraints`.
"""

from __future__ import annotations

from torch_linear_crf.crf import (
    FORBIDDEN,
    LinearChainCRF,
    forward_backward,
    pad_to_length_bucket,
)
from torch_linear_crf.graphs import GraphedCRF

__all__ = ["FORBIDDEN", "GraphedCRF", "LinearChainCRF", "forward_backward", "pad_to_length_bucket"]
__version__ = "0.1.0"
