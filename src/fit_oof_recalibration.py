"""Fit output recalibration on fold-2 OOF predictions and measure pooled RMSE.

Pointwise RMSE-optimal correction is E[target | prediction]. We estimate it with
isotonic regression and quantile-binned conditional means, cross-fitted over a
sample-level split of fold 2 so the reported RMSE is honest.

If a calibrator wins on OOF it can be applied directly to submission zip pixels
(no GPU) via apply_recalibration_to_zip.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RECIPES = {
    "wh100": lambda c: c["wh"],
    "wh50_twohead50": lambda c: 0.5 * c["wh"] + 0.5 * c["twohead"],
    "twohead100": lambda c: c["twohead"],
}

MODEL_DIRS = {
    "twohead": "swin_v2_temporal_two_head_oof",
    "wh": "swin_v2_temporal_weighted_huber_oof",
}


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return math.sqrt(float(np.mean((pred - target) ** 2)))


def fit_isotonic(pred: np.ndarray, target: np.ndarray, subsample: int, seed: int) -> IsotonicRegression:
    rng = np.random.default_rng(seed)
    idx = rng.choice(pred.size, size=min(subsample, pred.size), replace=False)
    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(pred[idx], target[idx])
    return iso


def fit_binned(pred: np.ndarray, target: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-binned conditional mean, returned as (bin_centers, bin_means)."""
    quantiles = np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles = np.unique(quantiles)
    bin_ids = np.clip(np.searchsorted(quantiles, pred, side="right") - 1, 0, len(quantiles) - 2)
    centers = np.zeros(len(quantiles) - 1)
    means = np.zeros(len(quantiles) - 1)
    for b in range(len(quantiles) - 1):
        mask = bin_ids == b
        if mask.any():
            centers[b] = pred[mask].mean()
            means[b] = target[mask].mean()
        else:
            centers[b] = 0.5 * (quantiles[b] + quantiles[b + 1])
            means[b] = centers[b]
    # enforce monotone map so interpolation is well-behaved in the tail
    means = np.maximum.accumulate(means)
    return centers, means


def apply_binned(pred: np.ndarray, centers: np.ndarray, means: np.ndarray) -> np.ndarray:
    return np.interp(pred, centers, means)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    pred_root = root / "outputs" / "oof_predictions"

    index = pd.read_csv(pred_root / "targets" / "oof_fold2.csv")
    target = np.load(pred_root / "targets" / "oof_fold2.npz")["target"].astype(np.float32)
    target = target.reshape(len(index), -1)

    components = {}
    for key, subdir in MODEL_DIRS.items():
        model_index = pd.read_csv(pred_root / subdir / "oof_fold2.csv")
        if not model_index["unique_id"].equals(index["unique_id"]):
            raise ValueError(f"unique_id mismatch: {key}")
        pred = np.load(pred_root / subdir / "oof_fold2.npz")["prediction"].astype(np.float32)
        components[key] = pred.reshape(len(index), -1)

    # location-grouped split: fold2 halves must not share locations, otherwise
    # shared extreme-rain sites leak into the calibration curve
    rng = np.random.default_rng(args.seed)
    locations = index["name_location"].to_numpy()
    unique_locations = rng.permutation(np.unique(locations))
    group_a = set(unique_locations[: len(unique_locations) // 2])
    half_a = np.array([loc in group_a for loc in locations])
    print(
        f"split: {half_a.sum()} samples / {len(group_a)} locations vs "
        f"{(~half_a).sum()} samples / {len(unique_locations) - len(group_a)} locations"
    )

    rows = []
    curves: dict[str, dict] = {}
    for name, fn in RECIPES.items():
        pred = fn(components)
        base = rmse(pred, target)
        calibrated_iso = np.empty_like(pred)
        calibrated_bin = np.empty_like(pred)
        for fit_mask in (half_a, ~half_a):
            apply_mask = ~fit_mask
            fit_pred = pred[fit_mask].ravel()
            fit_target = target[fit_mask].ravel()
            iso = fit_isotonic(fit_pred, fit_target, args.subsample, args.seed)
            calibrated_iso[apply_mask] = iso.predict(pred[apply_mask].ravel()).reshape(
                pred[apply_mask].shape
            )
            centers, means = fit_binned(fit_pred, fit_target, args.bins)
            calibrated_bin[apply_mask] = apply_binned(pred[apply_mask], centers, means)
        rows.append(
            {
                "recipe": name,
                "rmse_raw": base,
                "rmse_isotonic_cf": rmse(calibrated_iso, target),
                "rmse_binned_cf": rmse(calibrated_bin, target),
            }
        )
        # full-data curve for deployment (cross-fit only used for honest scoring)
        centers, means = fit_binned(pred.ravel(), target.ravel(), args.bins)
        curves[name] = {"centers": centers.tolist(), "means": means.tolist()}
        print(rows[-1])

    result = pd.DataFrame(rows)
    output_dir = root / "outputs" / "recalibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "fold2_recalibration_rmse.csv", index=False)
    with open(output_dir / "fold2_binned_curves.json", "w") as f:
        json.dump(curves, f)
    print("\n" + result.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/shionsuio/solafune-workspace")
    parser.add_argument("--subsample", type=int, default=2_000_000)
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
