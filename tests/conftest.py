"""CUDA tests are opt-in, so a test run never lands on a GPU another job is using.

Tests marked `cuda` run only with TLC_TEST_CUDA=1, and then
CUDA_VISIBLE_DEVICES must name the GPU(s) to use: unset or empty is an error,
not a silent fallback to whatever device 0 is.
"""

from __future__ import annotations

import os

import pytest
import torch


def cuda_requested() -> bool:
    return os.environ.get("TLC_TEST_CUDA") == "1"


def pytest_configure(config: pytest.Config) -> None:
    if not cuda_requested():
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise pytest.UsageError(
            "TLC_TEST_CUDA=1 needs CUDA_VISIBLE_DEVICES set to the GPU(s) to use"
        )
    if not torch.cuda.is_available():
        raise pytest.UsageError(
            "TLC_TEST_CUDA=1 but torch sees no CUDA device (CUDA_VISIBLE_DEVICES="
            + os.environ["CUDA_VISIBLE_DEVICES"]
            + ", torch "
            + torch.__version__
            + ")"
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if cuda_requested():
        return
    skip = pytest.mark.skip(reason="CUDA tests are opt-in: set TLC_TEST_CUDA=1")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
