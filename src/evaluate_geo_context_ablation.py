"""Evaluate context feature ablations using an already trained geo checkpoint.

This script does not retrain. It loads a fold checkpoint once per ablation mode
and zeros selected context feature groups during validation inference.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from kaggle_setup import ensure_kaggle_workspace
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    attach_location_metadata,
    get_device,
    load_fold_model,
    make_folds,
    make_loader,
    original_scale_rmse,
    prepare_metadata,
    satellite_directories,
)


ABLATIONS = {
    "full_geo": {},
    "no_geo_position": {"disable_geo_position_features": True},
    "no_local_time": {"disable_local_time_features": True},
    "no_geo_season": {"disable_geo_season_features": True},
    "local_time_only": {
        "disable_geo_position_features": True,
        "disable_geo_season_features": True,
    },
    "lat_lon_season_only": {"disable_local_time_features": True},
    "base_time_only": {
        "disable_geo_position_features": True,
        "disable_local_time_features": True,
        "disable_geo_season_features": True,
    },
}


def evaluate_validation_rmse(config: Config, dataframe: pd.DataFrame, fold: dict) -> float:
    device = get_device()
    model, stats = load_fold_model(config, fold["fold"], device)
    validation_frame = dataframe.iloc[fold["validation_indices"]].copy()
    validation_frame = attach_location_metadata(validation_frame, config)
    dataset = NowcastingDataset(
        validation_frame,
        satellite_directories(config, "train"),
        stats,
        config,
        has_target=True,
        augment=False,
    )
    loader = make_loader(dataset, config, device, shuffle=False)

    squared_error = 0.0
    pixel_count = 0
    model.eval()
    with torch.no_grad():
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            loader, desc=f"eval fold {fold['fold']}", leave=False
        ):
            prediction = model(
                image.to(device, non_blocking=True),
                satellite_id.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                missing.to(device, non_blocking=True),
            ).cpu()
            batch_squared_error, batch_pixels = original_scale_rmse(prediction, target)
            squared_error += batch_squared_error
            pixel_count += batch_pixels
    return math.sqrt(squared_error / pixel_count)


def parse_modes(value: str) -> list[str]:
    if value == "all":
        return list(ABLATIONS)
    modes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown ablation modes: {unknown}")
    return modes


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    if args.checkpoint_root:
        checkpoint_root = Path(args.checkpoint_root)
        model_dir = root / "models" / args.model_subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in [
            f"best_fold{args.fold}.pth",
            f"band_stats_fold{args.fold}.json",
        ]:
            source = checkpoint_root / filename
            if source.exists():
                destination = model_dir / filename
                if not destination.exists():
                    shutil.copy2(source, destination)

    config = Config(
        root=str(root),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        use_amp=not args.no_amp,
        use_temporal_differences=True,
        use_temporal_summary=True,
        use_location_features=True,
        location_metadata_path=args.location_metadata_path,
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(root / "models" / args.model_subdir),
    )

    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    folds = make_folds(dataframe, config.n_folds)
    fold = folds[args.fold]

    results = []
    for mode in parse_modes(args.modes):
        mode_config = replace(config, **ABLATIONS[mode])
        rmse = evaluate_validation_rmse(mode_config, dataframe, fold)
        results.append({"fold": args.fold, "mode": mode, "validation_rmse": rmse})
        print(f"fold={args.fold} mode={mode} val_rmse={rmse:.6f}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modes", default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-subdir", default="swin_v2_temporal_geo")
    parser.add_argument("--location-metadata-path", required=True)
    parser.add_argument("--output", default="outputs/swin_v2_temporal_geo/context_ablation.csv")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
