"""Reproduce submission blend recipes on OOF and correlate eval-like CV with Public LB.

Requires per-pixel OOF predictions exported by export_oof_predictions.py under
outputs/oof_predictions/<model_subdir>/oof_fold{k}.npz.

Verified submission composition (via zip algebra + LB equality):
    swin80_temporal20 = 0.8 * base_swin + 0.2 * weighted_huber_fold2(single model)
    temporal100 (calibration) == submission_swin_temporal_weighted_fold2

The weighted-huber model is a single fold-2 model, so it is out-of-fold only on
fold 2. The calibration family is therefore scored on fold 2 only, where every
component (wh, two-head fold2) is exactly OOF. Caveat: submissions built on the
twohead_mixed zip contain a 5-model ensemble; the OOF proxy uses the fold-2
model alone (standard OOF-for-ensemble approximation).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PIXELS = 41 * 41

# Public LB scores for the fold2 calibration family.
PUBLIC_LB = {
    "wh100": 0.6695843605746573,          # == swin_temporal_weighted_fold2
    "wh50_twohead50": 0.667970843035207,
    "twohead100": 0.6674952809153677,     # twohead_mixed ensemble submission
    "wh80_twohead20": 0.6688021061203975,
    "wh_hg20": 0.6690275982996665,
}

MODEL_DIRS = {
    "stable": "swin_v2_temporal_stable",
    "twohead": "swin_v2_temporal_two_head_oof",
    "wh": "swin_v2_temporal_weighted_huber_oof",
}


def satellite_mask_blend(
    base: np.ndarray,
    other: np.ndarray,
    satellites: np.ndarray,
    weight: float,
    blend_satellites: tuple[str, ...],
) -> np.ndarray:
    blended = (1.0 - weight) * base + weight * other
    result = base.copy()
    mask = np.isin(satellites, blend_satellites)
    result[mask] = blended[mask]
    return result


# fold2-only family: every component is exactly OOF on fold 2
FOLD2_RECIPES = {
    "wh100": lambda c, s: c["wh"],
    "wh80_twohead20": lambda c, s: 0.8 * c["wh"] + 0.2 * c["twohead"],
    "wh50_twohead50": lambda c, s: 0.5 * c["wh"] + 0.5 * c["twohead"],
    "wh_hg20": lambda c, s: satellite_mask_blend(
        c["wh"], c["twohead"], s, 0.20, ("himawari", "goes")
    ),
    "twohead100": lambda c, s: c["twohead"],
    "stable100_f2": lambda c, s: c["stable"],
}


def load_fold(pred_root: Path, model_key: str, fold: int) -> tuple[pd.DataFrame, np.ndarray]:
    directory = pred_root / MODEL_DIRS[model_key]
    index = pd.read_csv(directory / f"oof_fold{fold}.csv")
    predictions = np.load(directory / f"oof_fold{fold}.npz")["prediction"].astype(np.float32)
    return index, predictions.reshape(len(index), -1)


def load_target(pred_root: Path, fold: int) -> tuple[pd.DataFrame, np.ndarray]:
    directory = pred_root / "targets"
    index = pd.read_csv(directory / f"oof_fold{fold}.csv")
    target = np.load(directory / f"oof_fold{fold}.npz")["target"].astype(np.float32)
    return index, target.reshape(len(index), -1)


def weighted_rmse(frame: pd.DataFrame, se_column: str, weight_column: str | None) -> float:
    if weight_column is None:
        return math.sqrt(frame[se_column].sum() / (len(frame) * PIXELS))
    weighted_error = (frame[se_column] * frame[weight_column]).sum()
    weighted_pixels = frame[weight_column].sum() * PIXELS
    return math.sqrt(weighted_error / weighted_pixels)


METRICS = [
    "rmse",
    "rmse_adv_weight_sqrt",
    "rmse_adv_weight",
    "rmse_top30",
    "rmse_top10",
]


def metric_row(frame: pd.DataFrame, se_column: str) -> dict[str, float]:
    top30 = frame[frame["adv_rank_pct_within_split"] >= 0.70]
    top10 = frame[frame["adv_rank_pct_within_split"] >= 0.90]
    return {
        "rmse": weighted_rmse(frame, se_column, None),
        "rmse_adv_weight_sqrt": weighted_rmse(frame, se_column, "weight_sqrt_clipped"),
        "rmse_adv_weight": weighted_rmse(frame, se_column, "weight_clipped"),
        "rmse_top30": weighted_rmse(top30, se_column, None),
        "rmse_top10": weighted_rmse(top10, se_column, None),
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    pred_root = root / "outputs" / "oof_predictions"

    index, target = load_target(pred_root, 2)
    components = {}
    for key in ["wh", "twohead", "stable"]:
        model_index, prediction = load_fold(pred_root, key, 2)
        if not model_index["unique_id"].equals(index["unique_id"]):
            raise ValueError(f"unique_id mismatch: {key} fold 2")
        components[key] = prediction

    satellites = index["satellite_target"].str.lower().to_numpy()
    frame = index[["unique_id", "satellite_target", "month"]].copy()
    for name, fn in FOLD2_RECIPES.items():
        blend = fn(components, satellites)
        frame[f"se_{name}"] = ((blend - target) ** 2).sum(axis=1)

    scores = pd.read_csv(
        root / "outputs" / "adversarial_diagnostics" / "adversarial_scores.csv",
        usecols=[
            "unique_id",
            "split",
            "weight_sqrt_clipped",
            "weight_clipped",
            "adv_rank_pct_within_split",
        ],
    )
    scores = scores[scores["split"] == "train"]
    frame = frame.merge(scores, on="unique_id", how="left", validate="one_to_one")
    if frame["weight_sqrt_clipped"].isna().any():
        raise ValueError("some OOF samples missing adversarial scores")

    rows = []
    for name in FOLD2_RECIPES:
        row = {"recipe": name, "public_lb": PUBLIC_LB.get(name, np.nan)}
        row.update(metric_row(frame, f"se_{name}"))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("public_lb")

    output_dir = root / "outputs" / "cv_lb_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "fold2_recipe_metrics.csv", index=False)
    print(f"fold 2 calibration family ({len(frame)} samples):\n")
    print(result.to_string(index=False))

    scored = result.dropna(subset=["public_lb"])
    if len(scored) >= 3:
        print(f"\ncorrelation with Public LB over {len(scored)} submissions:")
        correlation_rows = []
        for metric in METRICS:
            spearman = spearmanr(scored["public_lb"], scored[metric])
            pearson = pearsonr(scored["public_lb"], scored[metric])
            correlation_rows.append(
                {
                    "metric": metric,
                    "spearman": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "pearson": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                }
            )
        correlation = pd.DataFrame(correlation_rows)
        correlation.to_csv(output_dir / "fold2_lb_correlation.csv", index=False)
        print(correlation.to_string(index=False))
    else:
        print("\nfewer than 3 recipes with LB scores — correlation not computed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
