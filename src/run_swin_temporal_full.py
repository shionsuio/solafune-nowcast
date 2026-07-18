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

import numpy as np
import pandas as pd

from kaggle_setup import ensure_kaggle_workspace
from swin_nowcast_v2 import (
    Config,
    align_observation_frames,
    extend_temporal_context,
    make_folds,
    prepare_metadata,
    train_fold,
)

# 2023-01-01 00:00 GPM tiles for these locations are corrupted duplicates
BAD_LABEL_LOCATIONS = ("aceh", "bihar", "dhaka", "guangdong", "jakarta")


def parse_folds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_folds_json(path: Path, row_count: int) -> list[dict]:
    raw_folds = json.loads(path.read_text())
    folds = []
    for raw in raw_folds:
        train = np.asarray(raw["train_indices"], dtype=np.int64)
        validation = np.asarray(raw["validation_indices"], dtype=np.int64)
        if train.size + validation.size != row_count:
            raise ValueError(f"Fold {raw['fold']} does not cover the dataframe")
        if np.intersect1d(train, validation).size:
            raise ValueError(f"Fold {raw['fold']} overlaps train and validation rows")
        folds.append(
            {
                "fold": int(raw["fold"]),
                "train_indices": train,
                "validation_indices": validation,
                "validation_locations": list(raw["validation_locations"]),
            }
        )
    return sorted(folds, key=lambda fold: fold["fold"])


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    write_stats_dir = root / "outputs" / args.model_subdir / "band_stats"
    write_stats_dir.mkdir(parents=True, exist_ok=True)
    if args.band_stats_root:
        read_stats_dir = Path(args.band_stats_root)
        for fold_index in parse_folds(args.folds):
            destination = write_stats_dir / f"band_stats_fold{fold_index}.json"
            source = read_stats_dir / destination.name
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)

    context_steps = int(getattr(args, "temporal_context_steps", 0))
    config = Config(
        root=str(root),
        max_observations=3 * (1 + context_steps),
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        loss_type=args.loss_type,
        heavy_rain_weight_alpha=args.heavy_rain_weight_alpha,
        heavy_rain_weight_scale=args.heavy_rain_weight_scale,
        heavy_rain_weight_max=args.heavy_rain_weight_max,
        raw_huber_loss_weight=args.raw_huber_loss_weight,
        raw_huber_beta=args.raw_huber_beta,
        raw_huber_max=args.raw_huber_max,
        use_two_head=args.use_two_head,
        rain_bce_weight_0_1=args.rain_bce_weight_0_1,
        rain_bce_weight_1=args.rain_bce_weight_1,
        rain_bce_weight_5=args.rain_bce_weight_5,
        workers=args.workers,
        stats_samples_per_satellite=args.stats_samples_per_satellite,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp,
        use_temporal_differences=not args.disable_temporal_features,
        use_temporal_summary=not args.disable_temporal_features,
        use_temporal_frame_attention=getattr(args, "temporal_frame_attention", False),
        use_location_features=args.use_location_features,
        location_metadata_path=args.location_metadata_path,
        location_feature_mode=args.location_feature_mode,
        sample_weight_path=getattr(args, "sample_weight_path", None),
        sample_weight_column=getattr(args, "sample_weight_column", "weight_sqrt_clipped"),
        band_mode=getattr(args, "band_mode", "matched6"),
        use_flow_divergence=getattr(args, "flow_divergence", False),
        use_flow_extrapolation=getattr(args, "flow_extrapolation", False),
        encoder_name=getattr(args, "encoder_name", "swin_tiny_patch4_window7_224"),
        pseudo_label_npz=getattr(args, "pseudo_label_npz", None),
        pseudo_label_csv=getattr(args, "pseudo_label_csv", None),
        pseudo_sample_weight=getattr(args, "pseudo_sample_weight", 1.0),
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(write_stats_dir),
    )

    train_csv = config.paths.train_dir / "train_dataset.csv"
    dataframe = prepare_metadata(train_csv)
    if getattr(args, "align_frames", False):
        before = dataframe["observation_files"].map(len)
        dataframe = align_observation_frames(dataframe)
        repaired = int(((before > 0) & (before < 3)).sum())
        print(f"align-frames: repaired {repaired} frame-deficient rows")
    if context_steps:
        dataframe = extend_temporal_context(dataframe, context_steps)
    folds = (
        load_folds_json(args.folds_json, len(dataframe))
        if args.folds_json
        else make_folds(dataframe, config.n_folds)
    )
    if getattr(args, "exclude_bad_labels", False):
        bad_mask = (
            (dataframe["datetime"] == pd.Timestamp("2023-01-01 00:00:00"))
            & dataframe["name_location"].isin(BAD_LABEL_LOCATIONS)
        ).to_numpy()
        bad_positions = np.flatnonzero(bad_mask)
        for fold in folds:
            fold["train_indices"] = np.setdiff1d(fold["train_indices"], bad_positions)
            fold["validation_indices"] = np.setdiff1d(fold["validation_indices"], bad_positions)
        print(f"exclude-bad-labels: dropped {len(bad_positions)} rows from folds")

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
    parser.add_argument("--folds-json", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type", default="huber", choices=["huber", "log_mse", "weighted_huber"]
    )
    parser.add_argument("--heavy-rain-weight-alpha", type=float, default=0.5)
    parser.add_argument("--heavy-rain-weight-scale", type=float, default=10.0)
    parser.add_argument("--heavy-rain-weight-max", type=float, default=2.0)
    parser.add_argument("--raw-huber-loss-weight", type=float, default=0.0)
    parser.add_argument("--raw-huber-beta", type=float, default=5.0)
    parser.add_argument("--raw-huber-max", type=float, default=100.0)
    parser.add_argument("--use-two-head", action="store_true")
    parser.add_argument("--rain-bce-weight-0-1", type=float, default=0.10)
    parser.add_argument("--rain-bce-weight-1", type=float, default=0.10)
    parser.add_argument("--rain-bce-weight-5", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stats-samples-per-satellite", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-subdir", default="swin_v2_temporal")
    parser.add_argument("--band-stats-root", default=None)
    parser.add_argument("--use-location-features", action="store_true")
    parser.add_argument("--location-metadata-path", default=None)
    parser.add_argument("--location-feature-mode", default="full", choices=["full", "local_time"])
    parser.add_argument("--sample-weight-path", default=None)
    parser.add_argument("--sample-weight-column", default="weight_sqrt_clipped")
    parser.add_argument(
        "--band-mode",
        default="matched6",
        choices=["legacy3", "matched6", "matched6_btd", "full16", "full16_btd"],
    )
    parser.add_argument("--encoder-name", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--flow-divergence", action="store_true")
    parser.add_argument("--flow-extrapolation", action="store_true")
    parser.add_argument("--temporal-context-steps", type=int, default=0)
    parser.add_argument("--temporal-frame-attention", action="store_true")
    parser.add_argument("--disable-temporal-features", action="store_true")
    parser.add_argument("--pseudo-label-npz", default=None)
    parser.add_argument("--pseudo-label-csv", default=None)
    parser.add_argument("--pseudo-sample-weight", type=float, default=1.0)
    parser.add_argument("--align-frames", action="store_true")
    parser.add_argument("--exclude-bad-labels", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
