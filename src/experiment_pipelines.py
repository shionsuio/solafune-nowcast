"""Shared experiment orchestration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiment_utils import balanced_sample, build_folds
from swin_nowcast_v2 import Config, original_scale_rmse


def evaluate_rmse(model, loader, device) -> float:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    with torch.no_grad():
        for image, satellite_id, temporal, missing, target, _ in loader:
            prediction = model(
                image.to(device),
                satellite_id.to(device),
                temporal.to(device),
                missing.to(device),
            ).cpu()
            batch_error, batch_pixels = original_scale_rmse(prediction, target)
            squared_error += batch_error
            pixel_count += batch_pixels
    return float(np.sqrt(squared_error / pixel_count))


def build_sampled_location_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold_index: int,
    train_rows: int,
    validation_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    fold = build_folds(config, dataframe)[fold_index]
    train_frame = balanced_sample(
        dataframe.iloc[fold["train_indices"]], train_rows, seed
    )
    validation_frame = balanced_sample(
        dataframe.iloc[fold["validation_indices"]],
        validation_rows,
        seed + 1,
    )
    sampled = pd.concat([train_frame, validation_frame], ignore_index=True)
    sampled_fold = {
        "fold": fold_index,
        "train_indices": np.arange(len(train_frame)),
        "validation_indices": np.arange(len(train_frame), len(sampled)),
        "validation_locations": fold["validation_locations"],
    }
    return sampled, sampled_fold
