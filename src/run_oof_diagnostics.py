"""Create OOF predictions and diagnostics from fold checkpoints."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
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
    prepare_metadata,
    satellite_directories,
)


RAIN_BINS = [
    ("zero", 0.0, 0.1),
    ("weak", 0.1, 1.0),
    ("moderate", 1.0, 5.0),
    ("heavy", 5.0, 10.0),
    ("extreme", 10.0, np.inf),
]


def parse_folds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def copy_checkpoint_files(source_dir: Path, model_dir: Path, folds: list[int]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for filename in [f"best_fold{fold}.pth", f"band_stats_fold{fold}.json"]:
            source = source_dir / filename
            if not source.exists():
                raise FileNotFoundError(source)
            destination = model_dir / filename
            if not destination.exists():
                shutil.copy2(source, destination)


def target_original(target_log: torch.Tensor) -> np.ndarray:
    return torch.expm1(target_log).clamp(min=0).numpy()


def prediction_original(prediction_log: torch.Tensor) -> np.ndarray:
    return torch.expm1(prediction_log).clamp(min=0).cpu().numpy()


def binary_counts(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = prediction >= threshold
    actual = target >= threshold
    return {
        f"tp_{threshold:g}": int(np.logical_and(predicted, actual).sum()),
        f"fp_{threshold:g}": int(np.logical_and(predicted, ~actual).sum()),
        f"fn_{threshold:g}": int(np.logical_and(~predicted, actual).sum()),
        f"tn_{threshold:g}": int(np.logical_and(~predicted, ~actual).sum()),
    }


def summarize_array(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = prediction - target
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    row: dict[str, float | int] = {
        "pixel_count": int(target.size),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "bias_mean": float(error.mean()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "target_p95": float(np.quantile(target, 0.95)),
        "prediction_p95": float(np.quantile(prediction, 0.95)),
        "target_p99": float(np.quantile(target, 0.99)),
        "prediction_p99": float(np.quantile(prediction, 0.99)),
        "target_max": float(target.max()),
        "prediction_max": float(prediction.max()),
        "squared_error_sum": float(squared_error.sum()),
        "absolute_error_sum": float(absolute_error.sum()),
        "target_sum": float(target.sum()),
        "prediction_sum": float(prediction.sum()),
    }
    for threshold in [0.1, 1.0, 5.0, 10.0]:
        row[f"target_ge_{threshold:g}"] = float((target >= threshold).mean())
        row[f"prediction_ge_{threshold:g}"] = float((prediction >= threshold).mean())
        row.update(binary_counts(prediction, target, threshold))
    for name, lower, upper in RAIN_BINS:
        mask = (target >= lower) & (target < upper)
        row[f"bin_{name}_pixels"] = int(mask.sum())
        row[f"bin_{name}_squared_error_sum"] = float(squared_error[mask].sum())
        row[f"bin_{name}_absolute_error_sum"] = float(absolute_error[mask].sum())
        row[f"bin_{name}_bias_sum"] = float(error[mask].sum())
    return row


def aggregate_file_stats(frame: pd.DataFrame) -> pd.Series:
    pixels = frame["pixel_count"].sum()
    output = {
        "samples": len(frame),
        "pixels": int(pixels),
        "rmse": math.sqrt(frame["squared_error_sum"].sum() / pixels),
        "mae": frame["absolute_error_sum"].sum() / pixels,
        "bias": (frame["prediction_sum"].sum() - frame["target_sum"].sum()) / pixels,
        "target_mean": frame["target_sum"].sum() / pixels,
        "prediction_mean": frame["prediction_sum"].sum() / pixels,
        "target_p95_mean": frame["target_p95"].mean(),
        "prediction_p95_mean": frame["prediction_p95"].mean(),
        "target_max_p95": frame["target_max"].quantile(0.95),
        "prediction_max_p95": frame["prediction_max"].quantile(0.95),
    }
    for threshold in [0.1, 1.0, 5.0, 10.0]:
        tp = frame[f"tp_{threshold:g}"].sum()
        fp = frame[f"fp_{threshold:g}"].sum()
        fn = frame[f"fn_{threshold:g}"].sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        output[f"precision_ge_{threshold:g}"] = precision
        output[f"recall_ge_{threshold:g}"] = recall
        output[f"target_ge_{threshold:g}"] = frame[f"target_ge_{threshold:g}"].mean()
        output[f"prediction_ge_{threshold:g}"] = frame[f"prediction_ge_{threshold:g}"].mean()
    for name, _, _ in RAIN_BINS:
        bin_pixels = frame[f"bin_{name}_pixels"].sum()
        if bin_pixels:
            output[f"rmse_bin_{name}"] = math.sqrt(
                frame[f"bin_{name}_squared_error_sum"].sum() / bin_pixels
            )
            output[f"mae_bin_{name}"] = (
                frame[f"bin_{name}_absolute_error_sum"].sum() / bin_pixels
            )
            output[f"bias_bin_{name}"] = frame[f"bin_{name}_bias_sum"].sum() / bin_pixels
            output[f"ratio_bin_{name}"] = bin_pixels / pixels
        else:
            output[f"rmse_bin_{name}"] = np.nan
            output[f"mae_bin_{name}"] = np.nan
            output[f"bias_bin_{name}"] = np.nan
            output[f"ratio_bin_{name}"] = 0.0
    return pd.Series(output)


def evaluate_fold(config: Config, dataframe: pd.DataFrame, fold: dict) -> pd.DataFrame:
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

    rows = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            loader, desc=f"OOF fold {fold['fold']}", leave=False
        ):
            prediction_log = model(
                image.to(device, non_blocking=True),
                satellite_id.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                missing.to(device, non_blocking=True),
            )
            prediction = prediction_original(prediction_log)
            target_np = target_original(target)
            batch_size = prediction.shape[0]
            meta = validation_frame.iloc[offset : offset + batch_size].reset_index(drop=True)
            offset += batch_size
            for batch_index, row in meta.iterrows():
                stats_row = summarize_array(prediction[batch_index, 0], target_np[batch_index, 0])
                stats_row.update(
                    {
                        "fold": fold["fold"],
                        "unique_id": row["unique_id"],
                        "name_location": row["name_location"],
                        "satellite_target": row["satellite_target"],
                        "month": row["datetime"].month,
                        "hour": row["datetime"].hour,
                        "datetime": row["datetime"],
                    }
                )
                rows.append(stats_row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    folds_to_run = parse_folds(args.folds)
    model_dir = root / "models" / args.model_subdir
    if args.checkpoint_root:
        copy_checkpoint_files(Path(args.checkpoint_root), model_dir, folds_to_run)

    config = Config(
        root=str(root),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        use_amp=not args.no_amp,
        use_two_head=args.use_two_head,
        use_temporal_differences=args.use_temporal_differences,
        use_temporal_summary=args.use_temporal_summary,
        use_location_features=args.use_location_features,
        location_metadata_path=args.location_metadata_path,
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(model_dir),
    )

    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    folds = make_folds(dataframe, config.n_folds)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stats = []
    for fold_index in folds_to_run:
        file_stats.append(evaluate_fold(config, dataframe, folds[fold_index]))
    file_stats_frame = pd.concat(file_stats, ignore_index=True)
    file_stats_frame.to_csv(output_dir / "oof_file_stats.csv", index=False)

    groupings = {
        "overall": [],
        "fold": ["fold"],
        "location": ["fold", "name_location", "satellite_target"],
        "satellite": ["satellite_target"],
        "month": ["month"],
        "satellite_month": ["satellite_target", "month"],
        "fold_month": ["fold", "month"],
    }
    for name, columns in groupings.items():
        if columns:
            summary = (
                file_stats_frame.groupby(columns, observed=True)
                .apply(aggregate_file_stats, include_groups=False)
                .reset_index()
            )
        else:
            summary = aggregate_file_stats(file_stats_frame).to_frame().T
        summary.to_csv(output_dir / f"oof_{name}_summary.csv", index=False)

    print("Overall")
    print(pd.read_csv(output_dir / "oof_overall_summary.csv").to_string(index=False))
    print("\nFold")
    print(pd.read_csv(output_dir / "oof_fold_summary.csv").to_string(index=False))
    print(f"Saved: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/working")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-subdir", default="swin_v2_temporal_stable")
    parser.add_argument("--output-dir", default="outputs/oof_swin_v2_temporal_stable")
    parser.add_argument("--use-temporal-differences", action="store_true")
    parser.add_argument("--use-temporal-summary", action="store_true")
    parser.add_argument("--use-two-head", action="store_true")
    parser.add_argument("--use-location-features", action="store_true")
    parser.add_argument("--location-metadata-path", default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
