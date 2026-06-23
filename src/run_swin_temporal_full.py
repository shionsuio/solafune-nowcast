"""Run the full Swin-T baseline with explicit temporal image features.

This is the follow-up to the lightweight Swin feature ablation:

- same location-disjoint folds as Swin v2
- same matched six physical bands
- same satellite-specific normalization / stem / embedding
- adds temporal differences plus temporal mean/std image channels
- saves checkpoints under models/swin_v2_temporal so they do not collide with
  the existing baseline checkpoints
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from kaggle_setup import ensure_kaggle_workspace
from swin_nowcast_v2 import Config, make_folds, prepare_metadata, train_fold


def parse_folds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    write_stats_dir = root / "outputs" / "band_stats"
    write_stats_dir.mkdir(parents=True, exist_ok=True)
    if args.band_stats_root:
        read_stats_dir = Path(args.band_stats_root)
        for fold_index in parse_folds(args.folds):
            destination = write_stats_dir / f"band_stats_fold{fold_index}.json"
            source = read_stats_dir / destination.name
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)

    config = Config(
        root=str(root),
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
        stats_samples_per_satellite=args.stats_samples_per_satellite,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
        use_temporal_differences=True,
        use_temporal_summary=True,
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(write_stats_dir),
    )

    train_csv = config.paths.train_dir / "train_dataset.csv"
    dataframe = prepare_metadata(train_csv)
    folds = make_folds(dataframe, config.n_folds)

    results = []
    for fold_index in parse_folds(args.folds):
        result = train_fold(config, dataframe, folds[fold_index])
        history = result["history"].copy()
        history_path = config.model_dir / f"history_fold{fold_index}.csv"
        best_epoch = int(history.loc[history["validation_rmse"].idxmin(), "epoch"])
        results.append(
            {
                "fold": fold_index,
                "best_epoch": best_epoch,
                "validation_rmse": result["validation_rmse"],
                "checkpoint": result["checkpoint"],
                "history": str(history_path),
            }
        )

    output_dir = root / "outputs" / args.model_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved: {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stats-samples-per-satellite", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-subdir", default="swin_v2_temporal")
    parser.add_argument("--band-stats-root", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
