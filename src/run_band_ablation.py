"""CLI wrapper for the band ablation experiment."""

from __future__ import annotations

import argparse

from experiments.band_ablation import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-rows", type=int, default=300)
    parser.add_argument("--stats-rows-per-satellite", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--band-stats-root", default=None)
    parser.add_argument("--output", default="outputs/band_ablation/results.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
