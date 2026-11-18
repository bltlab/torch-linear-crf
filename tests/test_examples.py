"""The examples/ scripts run end to end, so they cannot silently break."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_ner_tagger_runs_and_learns() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "ner_tagger.py"), "--epochs", "2"],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    f1_line = next(
        line for line in result.stdout.splitlines() if line.startswith("test mention F1")
    )
    assert float(f1_line.split()[3]) > 0.9, result.stdout


def test_readme_tagger_snippet_runs() -> None:
    """The trimmed tagger at the top of README's Usage section executes as written."""
    readme = (EXAMPLES.parent / "README.md").read_text()
    start = readme.index("A BiLSTM-CRF tagger")
    code = readme[start:].split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict = {}
    exec(code, namespace)
    assert len(namespace["labels"]) == 2
    assert tuple(namespace["confidence"].shape) == (2, 6, 5)
