"""Export per-pixel OOF predictions from fold checkpoints.

Saves one npz per fold with the full 41x41 prediction map per sample so that
submission blend recipes can be reproduced on OOF and scored with eval-like
weighted RMSE (see build_cv_lb_correlation.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kaggle_setup import ensure_kaggle_workspace
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    attach_location_metadata,
    extend_temporal_context,
    get_device,
    load_fold_model,
    make_folds,
    make_loader,
    prepare_metadata,
    satellite_directories,
)


def parse_folds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dihedral_transforms() -> list[tuple[int, bool]]:
    return [(k, flip) for flip in (False, True) for k in range(4)]


def apply_dihedral(image: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        image = torch.flip(image, dims=[3])
    if k:
        image = torch.rot90(image, k, dims=[2, 3])
    return image


def invert_dihedral(image: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if k:
        image = torch.rot90(image, -k, dims=[2, 3])
    if flip:
        image = torch.flip(image, dims=[3])
    return image


def export_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: dict,
    output_dir: Path,
    save_target: bool,
    limit: int | None,
    tta: bool = False,
) -> None:
    device = get_device()
    model, stats = load_fold_model(config, fold["fold"], device)
    validation_frame = dataframe.iloc[fold["validation_indices"]].copy()
    if limit:
        validation_frame = validation_frame.head(limit)
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

    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            loader, desc=f"Export fold {fold['fold']}", leave=False
        ):
            image = image.to(device, non_blocking=True)
            satellite_id = satellite_id.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            if tta:
                accumulated = None
                for k, flip in dihedral_transforms():
                    prediction_log = model(
                        apply_dihedral(image, k, flip), satellite_id, temporal, missing
                    )
                    restored = torch.expm1(
                        invert_dihedral(prediction_log, k, flip).float()
                    ).clamp(min=0)
                    accumulated = restored if accumulated is None else accumulated + restored
                prediction = (accumulated / len(dihedral_transforms())).cpu()
            else:
                prediction_log = model(image, satellite_id, temporal, missing)
                prediction = torch.expm1(prediction_log.float().cpu()).clamp(min=0)
            predictions.append(prediction[:, 0].numpy().astype(np.float16))
            if save_target:
                target_mm = torch.expm1(target).clamp(min=0)
                targets.append(target_mm[:, 0].numpy().astype(np.float16))

    prediction_array = np.concatenate(predictions)
    index = validation_frame.reset_index(drop=True)
    index_frame = pd.DataFrame(
        {
            "unique_id": index["unique_id"],
            "fold": fold["fold"],
            "name_location": index["name_location"],
            "satellite_target": index["satellite_target"],
            "month": index["datetime"].dt.month,
            "hour": index["datetime"].dt.hour,
            "datetime": index["datetime"],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"oof_fold{fold['fold']}.npz", prediction=prediction_array
    )
    index_frame.to_csv(output_dir / f"oof_fold{fold['fold']}.csv", index=False)
    if save_target:
        target_dir = output_dir.parent / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_dir / f"oof_fold{fold['fold']}.npz",
            target=np.concatenate(targets),
        )
        index_frame.to_csv(target_dir / f"oof_fold{fold['fold']}.csv", index=False)
    print(
        f"fold {fold['fold']}: saved {prediction_array.shape[0]} samples -> {output_dir}"
    )


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    context_steps = int(getattr(args, "temporal_context_steps", 0) or 0)
    model_dir = root / "models" / args.model_subdir
    config = Config(
        root=str(root),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        max_observations=3 * (1 + context_steps),
        use_two_head=args.use_two_head,
        use_temporal_differences=args.use_temporal_differences,
        use_temporal_summary=args.use_temporal_summary,
        use_location_features=False,
        swin_model_subdir=args.model_subdir,
        band_stats_root=str(model_dir),
        band_mode=args.band_mode,
    )

    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    if context_steps:
        dataframe = extend_temporal_context(dataframe, context_steps)
    folds = make_folds(dataframe, config.n_folds)
    output_dir = root / args.output_dir / args.model_subdir

    for fold_index in parse_folds(args.folds):
        export_fold(
            config,
            dataframe,
            folds[fold_index],
            output_dir,
            save_target=args.save_target,
            limit=args.limit,
            tta=getattr(args, "tta", False),
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--model-subdir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/oof_predictions")
    parser.add_argument("--use-temporal-differences", action="store_true")
    parser.add_argument("--use-temporal-summary", action="store_true")
    parser.add_argument("--use-two-head", action="store_true")
    parser.add_argument("--band-mode", default="matched6")
    parser.add_argument("--save-target", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--temporal-context-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
