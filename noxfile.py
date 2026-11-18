"""Test sessions, run the same way locally and in GitHub Actions.

    uvx nox -s compat                  # every torch minor in ci/torch-versions.json, CPU
    uvx nox -s crosscheck              # agreement with external CRF implementations
    uvx nox -s "compat(2.4.1)"         # one version
    CUDA_VISIBLE_DEVICES=1 uvx nox -s "compat(2.14.1)" -- --cuda

CUDA is opt-in: without --cuda every session hides the GPUs and installs
CPU-only torch wheels. With --cuda, CUDA_VISIBLE_DEVICES must name the GPU
to use (unset or empty is an error), torch comes from PyPI's CUDA wheels,
and the CUDA-marked tests run (see tests/conftest.py).

ci/torch-versions.json pairs each torch release with the newest Python on
which its torch.compile and torch.jit.script work (torch 2.5 has no Dynamo
on 3.13; torch 2.9 has neither on 3.14), so every test runs everywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "compat", "crosscheck"]

TORCH_VERSIONS = json.loads(Path("ci/torch-versions.json").read_text())
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# Pinned so a tool release cannot change what passes.
LINT_TOOLS = ["ruff==0.16.8", "pyright==1.1.414", "typos==1.50.3", "actionlint-py==1.7.12.25"]
# Pinned so the cross-checks catch drift on our side, not theirs.
CROSSCHECK_DEPS = ["pytorch-crf==0.7.2", "torch-struct==0.5"]


def _cuda_env(session: nox.Session) -> tuple[bool, dict[str, str]]:
    """Whether --cuda was given, and the environment pytest runs under."""
    if "--cuda" not in session.posargs:
        return False, {"CUDA_VISIBLE_DEVICES": "", "TLC_TEST_CUDA": "0"}
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        session.error("--cuda needs CUDA_VISIBLE_DEVICES set to the GPU to use")
    return True, {"TLC_TEST_CUDA": "1"}


def _pytest_args(session: nox.Session) -> list[str]:
    return [arg for arg in session.posargs if arg != "--cuda"]


@nox.session(python=TORCH_VERSIONS[-1]["python"])
def lint(session: nox.Session) -> None:
    """ruff, typos, actionlint on the workflows, and pyright on src/."""
    session.install(*LINT_TOOLS)
    session.run("ruff", "check")
    session.run("ruff", "format", "--check")
    session.run("typos")
    session.run("actionlint")
    # pyright resolves torch from this venv.
    session.install(f"torch=={TORCH_VERSIONS[-1]['torch']}", "--index-url", CPU_INDEX)
    session.install("-e", ".")
    session.run("pyright")


@nox.session
@nox.parametrize(
    ["python", "torch"],
    [(v["python"], v["torch"]) for v in TORCH_VERSIONS],
    ids=[v["torch"] for v in TORCH_VERSIONS],
)
def compat(session: nox.Session, torch: str) -> None:
    """The test suite, reference checks included, on one torch release."""
    cuda, env = _cuda_env(session)
    if cuda:
        session.install(f"torch=={torch}")
    else:
        session.install(f"torch=={torch}", "--index-url", CPU_INDEX)
    # Older torch's inductor imports setuptools, which a fresh venv lacks.
    session.install("-e", ".", "seqscore>=0.8.0", "pytest>=8", "setuptools")
    session.run(
        "python",
        "-c",
        "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
        env=env,
    )
    session.run("pytest", "-q", *_pytest_args(session), env=env)


@nox.session(python=TORCH_VERSIONS[-1]["python"])
def crosscheck(session: nox.Session) -> None:
    """Agreement with pytorch-crf, torch-struct and AllenNLP (see crosscheck/cases.py)."""
    session.install(f"torch=={TORCH_VERSIONS[-1]['torch']}", "--index-url", CPU_INDEX)
    session.install("-e", ".", "pytest>=8", *CROSSCHECK_DEPS)
    session.run("pytest", "-q", "crosscheck", *session.posargs, env={"CUDA_VISIBLE_DEVICES": ""})
