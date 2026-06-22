"""Helpers for running the Solafune pipelines in Kaggle Notebooks.

The Kaggle dataset mount is read-only, while model checkpoints and outputs
need a writable location. This module creates symlinks from a writable working
directory to the mounted input dataset so the existing training code can keep
using the same `Config(root=...)` layout.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_PATHS = (
    ("train_dataset", "train_dataset.csv"),
    ("evaluation_dataset", "evaluation_target.csv"),
    ("sample_submission", "evaluation_target.csv"),
)
REQUIRED_DIRS = ("train_dataset", "evaluation_dataset", "sample_submission")
REQUIRED_SAMPLE_SUBDIRS = ("test_files",)


def detect_kaggle_input_root(search_root: Path = Path("/kaggle/input")) -> Path:
    """Return the mounted dataset directory that contains Solafune assets."""
    if not search_root.exists():
        raise FileNotFoundError(f"Input mount not found: {search_root}")

    for candidate in sorted(search_root.iterdir()):
        if not candidate.is_dir():
            continue
        if all((candidate / folder / filename).exists() for folder, filename in REQUIRED_PATHS):
            if all((candidate / folder).is_dir() for folder in REQUIRED_DIRS) and all(
                (candidate / "sample_submission" / subdir).is_dir()
                for subdir in REQUIRED_SAMPLE_SUBDIRS
            ):
                return candidate

    raise FileNotFoundError(
        f"Could not find a Kaggle dataset under {search_root} with the expected Solafune layout"
    )


def ensure_kaggle_workspace(
    workspace_root: Path = Path("/kaggle/working"),
    input_root: Path | None = None,
) -> Path:
    """Create symlinks for the dataset folders inside a writable workspace."""
    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    input_root = input_root or detect_kaggle_input_root()

    for folder, _ in REQUIRED_PATHS:
        source = input_root / folder
        destination = workspace_root / folder
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.is_dir() and destination.resolve() == source.resolve():
                continue
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.symlink_to(source, target_is_directory=True)

    return workspace_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default="/kaggle/working")
    parser.add_argument("--input-root", default=None)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root)
    input_root = Path(args.input_root) if args.input_root else None
    resolved = ensure_kaggle_workspace(workspace_root, input_root)
    print(f"Workspace ready: {resolved}")
    for folder, filename in REQUIRED_PATHS:
        print(f"{folder}: {(resolved / folder / filename).exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
