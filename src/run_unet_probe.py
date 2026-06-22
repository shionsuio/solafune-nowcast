"""Small local U-Net smoke test using a location-disjoint fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_feature_ablation import balanced_sample
from swin_nowcast_v2 import Config, get_device, make_folds, prepare_metadata
from unet_nowcast_v2 import UNetNowcaster, train_unet_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-rows", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-size", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = Config(
        root=str(root),
        encoder_size=args.encoder_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=0,
        pretrained=False,
        use_amp=False,
        stats_samples_per_satellite=300,
        unet_model_subdir="unet_probe",
    )
    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    original_fold = make_folds(dataframe, config.n_folds)[args.fold]
    train_frame = balanced_sample(
        dataframe.iloc[original_fold["train_indices"]], args.train_rows, config.seed
    )
    validation_frame = balanced_sample(
        dataframe.iloc[original_fold["validation_indices"]],
        args.validation_rows,
        config.seed + 1,
    )
    sampled = pd.concat([train_frame, validation_frame], ignore_index=True)
    fold = {
        "fold": args.fold,
        "train_indices": np.arange(len(train_frame)),
        "validation_indices": np.arange(len(train_frame), len(sampled)),
        "validation_locations": original_fold["validation_locations"],
    }

    model = UNetNowcaster(config, base_channels=args.base_channels)
    print("parameters", sum(parameter.numel() for parameter in model.parameters()))
    result = train_unet_fold(
        config,
        sampled,
        fold,
        device=get_device(),
        base_channels=args.base_channels,
    )
    print(result)


if __name__ == "__main__":
    main()
