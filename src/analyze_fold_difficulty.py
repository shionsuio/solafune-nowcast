"""CLI wrapper for target difficulty analysis."""

from __future__ import annotations

import argparse

from experiments.difficulty import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/fold_difficulty")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
