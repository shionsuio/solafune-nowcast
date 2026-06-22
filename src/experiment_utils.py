"""Shared helpers for Solafune experiments."""

from __future__ import annotations

import pandas as pd

from swin_nowcast_v2 import Config, make_folds, prepare_metadata


def load_train_dataframe(config: Config) -> pd.DataFrame:
    return prepare_metadata(config.paths.train_dir / "train_dataset.csv")


def build_folds(config: Config, dataframe: pd.DataFrame) -> list[dict]:
    return make_folds(dataframe, config.n_folds)


def balanced_sample(dataframe: pd.DataFrame, total_rows: int, seed: int) -> pd.DataFrame:
    per_satellite = max(total_rows // dataframe["satellite_target"].nunique(), 1)
    sampled = []
    for satellite, group in dataframe.groupby("satellite_target"):
        sampled.append(
            group.sample(
                min(per_satellite, len(group)),
                random_state=seed,
            )
        )
    return (
        pd.concat(sampled)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
