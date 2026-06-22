"""CLI wrapper for the U-Net probe experiment."""

from __future__ import annotations

import argparse

from experiments.probes import run_unet_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-rows", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-size", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--band-stats-root", default=None)
    args = parser.parse_args()
    run_unet_probe(args)


if __name__ == "__main__":
    main()
