"""CLI wrapper for the feature ablation experiment."""

from __future__ import annotations

import argparse

from experiments.feature_ablation import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=1800)
    parser.add_argument("--validation-rows", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--band-stats-root", default=None)
    parser.add_argument("--output", default="outputs/feature_ablation/results.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
