"""CLI wrapper for the ConvNeXt probe experiment."""

from __future__ import annotations

import argparse

from experiments.probes import run_convnext_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=300)
    parser.add_argument("--validation-rows", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-size", type=int, default=96)
    args = parser.parse_args()
    run_convnext_probe(args)


if __name__ == "__main__":
    main()
