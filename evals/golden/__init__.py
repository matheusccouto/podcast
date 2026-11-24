"""Golden reference fixtures and the shared dataset loader for evaluations."""

from collections.abc import Callable
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden

from evals.golden.curator import CuratorGoldenCase
from evals.golden.writer import GoldenCase, GoldenStory

__all__ = ["CuratorGoldenCase", "GoldenCase", "GoldenStory", "load_dataset"]


def load_dataset(
    directory: Path,
    build: Callable[[Path], Golden | None],
) -> EvaluationDataset:
    """Load every `*.json` fixture in `directory` through `build`.

    `build` parses one fixture path into its Golden, or returns None when the
    fixture carries nothing to evaluate.
    """
    goldens = [
        golden
        for path in sorted(directory.glob("*.json"))
        if (golden := build(path)) is not None
    ]
    return EvaluationDataset(goldens=goldens)
