"""Build train/eval adversarial scores and eval-like OOF diagnostics.

This is for diagnosing CV/LB mismatch under covariate shift.  It uses only
provided train/evaluation inputs and OOF file-level prediction summaries.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SATELLITE_BANDS = {
    "himawari": (5, 6, 8, 10, 13, 15),
    "goes": (5, 6, 8, 10, 13, 15),
    "meteosat": (7, 8, 10, 11, 14, 15),
}
STAT_NAMES = ("mean", "std", "p05", "p50", "p95", "max")


def parse_observation_count(value: str) -> int:
    try:
        return len(ast.literal_eval(value))
    except Exception:
        return 0


def load_split(root: Path, split: str) -> pd.DataFrame:
    if split == "train":
        frame = pd.read_csv(root / "train_dataset" / "train_dataset.csv")
        base_dir = root / "train_dataset"
    elif split == "eval":
        frame = pd.read_csv(root / "evaluation_dataset" / "evaluation_target.csv")
        base_dir = root / "evaluation_dataset"
    else:
        raise ValueError(split)
    frame = frame.copy()
    frame["split"] = split
    frame["base_dir"] = str(base_dir)
    return frame


def statistic_columns() -> list[str]:
    return [f"b{band_index}_{stat}" for band_index in range(6) for stat in STAT_NAMES]


def read_last_observation_stats(row: pd.Series) -> np.ndarray:
    satellite = row["satellite_target"]
    bands = SATELLITE_BANDS[satellite]
    files = ast.literal_eval(row["last_30_minutes_observation_filename"])
    if not files:
        return np.full(len(bands) * len(STAT_NAMES), np.nan, dtype=np.float32)

    path = Path(row["base_dir"]) / satellite / files[-1]
    with rasterio.open(path) as src:
        readable_bands = [band for band in bands if band <= src.count]
        if not readable_bands:
            return np.full(len(bands) * len(STAT_NAMES), np.nan, dtype=np.float32)
        data = src.read(readable_bands).astype(np.float32)

    by_band = {band: data[index] for index, band in enumerate(readable_bands)}
    output: list[float] = []
    for band in bands:
        if band not in by_band:
            output.extend([np.nan] * len(STAT_NAMES))
            continue
        values = by_band[band]
        values = values[np.isfinite(values)]
        if values.size == 0:
            output.extend([np.nan] * len(STAT_NAMES))
            continue
        output.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.50)),
                float(np.quantile(values, 0.95)),
                float(values.max()),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def build_image_stats(root: Path, output_path: Path, force: bool) -> pd.DataFrame:
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    frame = pd.concat([load_split(root, "train"), load_split(root, "eval")], ignore_index=True)
    columns = statistic_columns()
    rows = []
    for index, row in frame.iterrows():
        if index % 1000 == 0:
            print(f"reading image stats {index}/{len(frame)}", flush=True)
        rows.append(read_last_observation_stats(row))

    stats = pd.DataFrame(rows, columns=columns)
    output = pd.concat([frame.drop(columns=["base_dir"]), stats], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def add_metadata_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    timestamp = pd.to_datetime(frame["datetime"])
    frame["year"] = timestamp.dt.year
    frame["month"] = timestamp.dt.month.astype(str)
    frame["month_num"] = timestamp.dt.month
    frame["dayofyear"] = timestamp.dt.dayofyear
    frame["hour_float"] = timestamp.dt.hour + timestamp.dt.minute / 60
    for column, period in [("dayofyear", 366), ("hour_float", 24)]:
        value = frame[column].to_numpy(float)
        frame[f"{column}_sin"] = np.sin(2 * np.pi * value / period)
        frame[f"{column}_cos"] = np.cos(2 * np.pi * value / period)
    frame["obs_count"] = frame["last_30_minutes_observation_filename"].map(
        parse_observation_count
    )
    frame["missing_obs"] = (frame["obs_count"] == 0).astype(int)
    return frame


def adversarial_predictions(
    frame: pd.DataFrame,
    model_name: str,
    random_state: int,
) -> tuple[np.ndarray, float]:
    stat_columns = statistic_columns()
    numeric_columns = [
        "year",
        "dayofyear_sin",
        "dayofyear_cos",
        "hour_float_sin",
        "hour_float_cos",
        "obs_count",
        "missing_obs",
        *stat_columns,
    ]
    categorical_columns = ["satellite_target", "month"]
    x = frame[categorical_columns + numeric_columns].copy()
    for column in numeric_columns:
        if x[column].isna().any():
            x[column] = x[column].fillna(x[column].median())
    y = (frame["split"] == "eval").astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("num", StandardScaler(), numeric_columns),
        ]
    )
    if model_name == "lr":
        classifier = LogisticRegression(max_iter=1000, C=1.0)
    elif model_name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=30,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(model_name)

    pipeline = make_pipeline(preprocessor, classifier)
    prediction = cross_val_predict(
        pipeline, x, y, cv=cv, method="predict_proba", n_jobs=None
    )[:, 1]
    auc = roc_auc_score(y, prediction)
    return prediction, auc


def build_adversarial_scores(
    image_stats: pd.DataFrame,
    output_path: Path,
    model_name: str,
    random_state: int,
) -> pd.DataFrame:
    frame = add_metadata_features(image_stats)
    prediction, auc = adversarial_predictions(frame, model_name, random_state)
    train_count = (frame["split"] == "train").sum()
    eval_count = (frame["split"] == "eval").sum()
    eps = 1e-4
    clipped_prediction = np.clip(prediction, eps, 1 - eps)
    density_ratio = clipped_prediction / (1 - clipped_prediction) * train_count / eval_count
    density_ratio = np.clip(density_ratio, 0.05, 20.0)
    weight_sqrt = np.sqrt(density_ratio)
    weight_sqrt_clipped = np.clip(weight_sqrt, 0.5, 3.0)
    weight_clipped = np.clip(density_ratio, 0.5, 3.0)

    output = frame[
        [
            "unique_id",
            "split",
            "name_location",
            "satellite_target",
            "datetime",
            "month_num",
            "hour_float",
            "obs_count",
        ]
    ].copy()
    output = output.rename(columns={"month_num": "month"})
    output["adv_score"] = prediction
    output["adv_auc_global"] = auc
    output["density_ratio"] = density_ratio
    output["weight_sqrt_clipped"] = weight_sqrt_clipped
    output["weight_clipped"] = weight_clipped
    output["adv_rank_pct_within_split"] = output.groupby("split")["adv_score"].rank(pct=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"adversarial AUC ({model_name}) = {auc:.6f}")
    return output


def weighted_rmse(frame: pd.DataFrame, weight_column: str | None = None) -> float:
    if weight_column is None:
        return math.sqrt(frame["squared_error_sum"].sum() / frame["pixel_count"].sum())
    weighted_error = (frame["squared_error_sum"] * frame[weight_column]).sum()
    weighted_pixels = (frame["pixel_count"] * frame[weight_column]).sum()
    return math.sqrt(weighted_error / weighted_pixels)


def summarize_oof(oof: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    groups = [((), oof)] if not group_columns else oof.groupby(group_columns, observed=True)
    for key, frame in groups:
        row = {}
        if group_columns:
            if not isinstance(key, tuple):
                key = (key,)
            row.update(dict(zip(group_columns, key)))
        row.update(
            {
                "samples": len(frame),
                "rmse": weighted_rmse(frame),
                "rmse_adv_weight_sqrt": weighted_rmse(frame, "weight_sqrt_clipped"),
                "rmse_adv_weight": weighted_rmse(frame, "weight_clipped"),
                "rmse_eval_like_top30": weighted_rmse(
                    frame[frame["adv_rank_pct_within_train"] >= 0.70]
                ),
                "rmse_eval_like_top10": weighted_rmse(
                    frame[frame["adv_rank_pct_within_train"] >= 0.90]
                ),
                "adv_score_mean": frame["adv_score"].mean(),
                "adv_score_p75": frame["adv_score"].quantile(0.75),
                "adv_score_p90": frame["adv_score"].quantile(0.90),
                "target_mean": frame["target_sum"].sum() / frame["pixel_count"].sum(),
                "prediction_mean": frame["prediction_sum"].sum() / frame["pixel_count"].sum(),
                "bias": (frame["prediction_sum"].sum() - frame["target_sum"].sum())
                / frame["pixel_count"].sum(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_oof_diagnostics(
    adversarial_scores: pd.DataFrame,
    oof_file_stats: Path,
    output_dir: Path,
) -> None:
    oof = pd.read_csv(oof_file_stats)
    train_scores = adversarial_scores[adversarial_scores["split"] == "train"].copy()
    train_scores = train_scores.rename(
        columns={"adv_rank_pct_within_split": "adv_rank_pct_within_train"}
    )
    merged = oof.merge(
        train_scores[
            [
                "unique_id",
                "adv_score",
                "density_ratio",
                "weight_sqrt_clipped",
                "weight_clipped",
                "adv_rank_pct_within_train",
            ]
        ],
        on="unique_id",
        how="left",
        validate="one_to_one",
    )
    if merged["adv_score"].isna().any():
        raise RuntimeError("OOF rows missing adversarial scores")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "oof_with_adversarial_scores.csv", index=False)

    summaries = {
        "overall": [],
        "fold": ["fold"],
        "satellite": ["satellite_target"],
        "month": ["month"],
        "fold_satellite": ["fold", "satellite_target"],
        "fold_month": ["fold", "month"],
        "location": ["fold", "name_location", "satellite_target"],
    }
    for name, columns in summaries.items():
        summarize_oof(merged, columns).to_csv(
            output_dir / f"oof_eval_like_{name}.csv", index=False
        )

    print("OOF eval-like overall")
    print(pd.read_csv(output_dir / "oof_eval_like_overall.csv").to_string(index=False))
    print("\nOOF eval-like by fold")
    print(pd.read_csv(output_dir / "oof_eval_like_fold.csv").to_string(index=False))


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    image_stats_path = output_dir / "adversarial_image_stats_all.csv"
    scores_path = output_dir / "adversarial_scores.csv"
    image_stats = build_image_stats(root, image_stats_path, args.force_image_stats)
    adversarial_scores = build_adversarial_scores(
        image_stats, scores_path, args.model, args.seed
    )
    build_oof_diagnostics(adversarial_scores, Path(args.oof_file_stats), output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--oof-file-stats",
        default="kaggle_outputs/swin_oof_diagnostics/outputs/oof_swin_v2_temporal_stable/oof_file_stats.csv",
    )
    parser.add_argument("--output-dir", default="outputs/adversarial_diagnostics")
    parser.add_argument("--model", choices=["lr", "rf"], default="rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-image-stats", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
