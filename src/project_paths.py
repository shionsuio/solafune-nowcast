"""Shared workspace and dataset path helpers.

Keep filesystem layout decisions in one place so new scripts can reuse the
same assumptions without duplicating path logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SOLAFUNE_DATA_FOLDERS = (
    "train_dataset",
    "evaluation_dataset",
    "sample_submission",
)

SOLAFUNE_REQUIRED_FILES = (
    ("train_dataset", "train_dataset.csv"),
    ("evaluation_dataset", "evaluation_target.csv"),
    ("sample_submission", "evaluation_target.csv"),
)


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Canonical Solafune workspace layout rooted at one directory."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))

    @property
    def train_dir(self) -> Path:
        return self.root / "train_dataset"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation_dataset"

    @property
    def sample_submission_dir(self) -> Path:
        return self.root / "sample_submission"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def gpm_dir(self) -> Path:
        return self.train_dir / "gpm_imerg"


def find_solafune_input_root(search_root: Path = Path("/kaggle/input")) -> Path:
    """Find the mounted Kaggle dataset that contains the Solafune layout."""
    if not search_root.exists():
        raise FileNotFoundError(f"Input mount not found: {search_root}")

    candidate_csvs = sorted(search_root.rglob("train_dataset.csv"))
    for csv_path in candidate_csvs:
        candidate = csv_path.parent.parent
        if candidate == search_root.parent:
            continue
        if all(
            (candidate / folder / filename).exists()
            for folder, filename in SOLAFUNE_REQUIRED_FILES
        ):
            if all((candidate / folder).is_dir() for folder in SOLAFUNE_DATA_FOLDERS):
                if (candidate / "sample_submission" / "test_files").is_dir():
                    return candidate

    raise FileNotFoundError(
        f"Could not find a Kaggle dataset under {search_root} with the expected Solafune layout"
    )
