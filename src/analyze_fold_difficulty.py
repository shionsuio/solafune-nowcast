"""Aggregate target difficulty by location and validation fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm

from swin_nowcast_v2 import Config, make_folds, prepare_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/fold_difficulty")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(root=str(root))
    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    fold_by_location = {}
    for fold in make_folds(dataframe, config.n_folds):
        for location in fold["validation_locations"]:
            fold_by_location[location] = fold["fold"]
    dataframe["fold"] = dataframe["name_location"].map(fold_by_location)

    rows = []
    gpm_dir = config.paths.train_dir / "gpm_imerg"
    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc="Reading GPM targets",
    ):
        with rasterio.open(gpm_dir / row.gpm_imerg_filename) as src:
            target = src.read(1).astype(np.float32)
            if src.nodata is not None:
                target[target == src.nodata] = 0
        target = np.clip(target, 0, None)
        rows.append(
            {
                "unique_id": row.unique_id,
                "fold": row.fold,
                "name_location": row.name_location,
                "satellite_target": row.satellite_target,
                "pixel_count": target.size,
                "sum": float(target.sum()),
                "sum_sq": float(np.square(target).sum()),
                "positive_pixels": int((target > 0).sum()),
                "rain1_pixels": int((target >= 1).sum()),
                "rain5_pixels": int((target >= 5).sum()),
                "rain10_pixels": int((target >= 10).sum()),
                "max": float(target.max()),
            }
        )

    file_stats = pd.DataFrame(rows)
    file_stats.to_csv(output_dir / "target_file_stats.csv", index=False)

    def aggregate(frame: pd.DataFrame) -> pd.Series:
        pixels = frame["pixel_count"].sum()
        return pd.Series(
            {
                "samples": len(frame),
                "target_mean": frame["sum"].sum() / pixels,
                "zero_baseline_rmse": np.sqrt(frame["sum_sq"].sum() / pixels),
                "positive_ratio": frame["positive_pixels"].sum() / pixels,
                "rain_ge_1_ratio": frame["rain1_pixels"].sum() / pixels,
                "rain_ge_5_ratio": frame["rain5_pixels"].sum() / pixels,
                "rain_ge_10_ratio": frame["rain10_pixels"].sum() / pixels,
                "sample_max_median": frame["max"].median(),
                "sample_max_p95": frame["max"].quantile(0.95),
                "global_max": frame["max"].max(),
            }
        )

    location_stats = (
        file_stats.groupby(
            ["fold", "name_location", "satellite_target"], observed=True
        )
        .apply(aggregate, include_groups=False)
        .reset_index()
        .sort_values(["fold", "zero_baseline_rmse"], ascending=[True, False])
    )
    fold_stats = (
        file_stats.groupby("fold", observed=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
        .sort_values("fold")
    )
    location_stats.to_csv(output_dir / "location_stats.csv", index=False)
    fold_stats.to_csv(output_dir / "fold_stats.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    ordered = location_stats.sort_values("zero_baseline_rmse")
    colors = [f"C{fold}" for fold in ordered["fold"]]
    axes[0].barh(ordered["name_location"], ordered["zero_baseline_rmse"], color=colors)
    axes[0].set_xlabel("Zero-prediction RMSE")
    axes[0].set_title("Target difficulty by location")
    axes[1].bar(
        fold_stats["fold"].astype(str),
        fold_stats["zero_baseline_rmse"],
        color=[f"C{fold}" for fold in fold_stats["fold"]],
    )
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Zero-prediction RMSE")
    axes[1].set_title("Target difficulty by fold")
    figure.tight_layout()
    figure.savefig(output_dir / "fold_location_difficulty.png", dpi=160)
    print("\nFold summary")
    print(fold_stats.to_string(index=False))
    print("\nLocation summary")
    print(location_stats.to_string(index=False))


if __name__ == "__main__":
    main()
