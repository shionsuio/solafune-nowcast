"""Export two-head OOF predictions with rain-probability maps (return_aux).

Saves amount_log (softplus output, log1p space) and sigmoid probabilities for
the 0.1/1/5 mm thresholds so wet/dry gating strategies can be scored offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from export_oof_predictions import parse_folds
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    attach_location_metadata,
    get_device,
    load_fold_model,
    make_folds,
    prepare_metadata,
    satellite_directories,
)
from torch.utils.data import DataLoader


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    model_dir = root / "models" / args.model_subdir
    config = Config(
        root=str(root),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        use_two_head=True,
        use_temporal_differences=True,
        use_temporal_summary=True,
        use_location_features=False,
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(model_dir),
    )
    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    folds = make_folds(dataframe, config.n_folds)
    output_dir = root / "outputs" / "oof_predictions" / f"{args.model_subdir}_aux"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    for fold_index in parse_folds(args.folds):
        fold = folds[fold_index]
        model, stats = load_fold_model(config, fold_index, device)
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
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.workers > 0,
        )
        amounts, probs = [], []
        model.eval()
        with torch.no_grad():
            for image, satellite_id, temporal, missing, _, _ in tqdm(
                loader, desc=f"Aux fold {fold_index}", leave=False
            ):
                prediction, rain_logits = model(
                    image.to(device, non_blocking=True),
                    satellite_id.to(device, non_blocking=True),
                    temporal.to(device, non_blocking=True),
                    missing.to(device, non_blocking=True),
                    return_aux=True,
                )
                amount_log = prediction.float() / torch.sigmoid(
                    rain_logits[:, :1]
                ).float().clamp(min=1e-6)
                amounts.append(amount_log[:, 0].cpu().numpy().astype(np.float16))
                probs.append(
                    torch.sigmoid(rain_logits).float().cpu().numpy().astype(np.float16)
                )
        np.savez_compressed(
            output_dir / f"oof_fold{fold_index}.npz",
            amount_log=np.concatenate(amounts),
            rain_probs=np.concatenate(probs),
        )
        print(f"fold {fold_index}: saved {len(dataset)} samples -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-subdir", default="swin_v2_temporal_two_head_oof")
    parser.add_argument("--folds", default="2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
