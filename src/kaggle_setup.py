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

from project_paths import (
    SOLAFUNE_DATA_FOLDERS,
    SOLAFUNE_REQUIRED_FILES,
    find_solafune_input_root,
)


def detect_kaggle_input_root(search_root: Path = Path("/kaggle/input")) -> Path:
    return find_solafune_input_root(search_root)


def ensure_kaggle_workspace(
    workspace_root: Path = Path("/kaggle/working"),
    input_root: Path | None = None,
) -> Path:
    """Create symlinks for the dataset folders inside a writable workspace."""
    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    input_root = input_root or detect_kaggle_input_root()

    for folder in SOLAFUNE_DATA_FOLDERS:
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
    for folder, filename in SOLAFUNE_REQUIRED_FILES:
        print(f"{folder}: {(resolved / folder / filename).exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
