"""Fast comparison of legacy bands, physical band mapping, and satellite correction."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from run_feature_ablation import balanced_sample
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    SwinNowcaster,
    compute_band_stats,
    get_band_mapping,
    get_device,
    load_stats,
    make_folds,
    make_loader,
    original_scale_rmse,
    prepare_metadata,
    satellite_directories,
    save_stats,
    seed_everything,
)


EXPERIMENTS = {
    "legacy3_shared": {
        "band_mode": "legacy3",
        "use_satellite_normalization": False,
        "use_satellite_stem": False,
        "use_satellite_embedding": False,
    },
    "matched6_shared": {
        "band_mode": "matched6",
        "use_satellite_normalization": False,
        "use_satellite_stem": False,
        "use_satellite_embedding": False,
    },
    "matched6_satellite_corrected": {
        "band_mode": "matched6",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
    },
}


def evaluate(model, loader, device) -> float:
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
    return math.sqrt(squared_error / pixel_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-rows", type=int, default=300)
    parser.add_argument("--stats-rows-per-satellite", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="outputs/band_ablation/results.csv"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    device = get_device()
    base_config = Config(
        root=str(root),
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=0,
        pretrained=False,
        use_amp=False,
        seed=args.seed,
    )
    dataframe = prepare_metadata(base_config.train_dir / "train_dataset.csv")
    fold = make_folds(dataframe, base_config.n_folds)[args.fold]
    fold_train = dataframe.iloc[fold["train_indices"]]
    train_frame = balanced_sample(fold_train, args.train_rows, args.seed)
    validation_frame = balanced_sample(
        dataframe.iloc[fold["validation_indices"]],
        args.validation_rows,
        args.seed + 1,
    )
    directories = satellite_directories(base_config, "train")
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    stats_by_mode = {}
    for band_mode in ("legacy3", "matched6"):
        stats_path = output_path.parent / f"stats_{band_mode}_fold{args.fold}.json"
        config = replace(base_config, band_mode=band_mode)
        if stats_path.exists():
            stats = load_stats(stats_path)
        else:
            stats = compute_band_stats(
                fold_train,
                directories,
                max_samples_per_satellite=args.stats_rows_per_satellite,
                seed=args.seed,
                band_mapping=get_band_mapping(config),
                include_shared=True,
            )
            save_stats(stats, stats_path)
        stats_by_mode[band_mode] = stats

    for name, flags in EXPERIMENTS.items():
        config = replace(base_config, **flags)
        seed_everything(args.seed)
        stats = stats_by_mode[config.band_mode]
        train_dataset = NowcastingDataset(
            train_frame, directories, stats, config, has_target=True, augment=True
        )
        validation_dataset = NowcastingDataset(
            validation_frame,
            directories,
            stats,
            config,
            has_target=True,
            augment=False,
        )
        train_loader = make_loader(train_dataset, config, device, shuffle=True)
        validation_loader = make_loader(
            validation_dataset, config, device, shuffle=False
        )
        model = SwinNowcaster(config).to(device)
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2e-4,
            weight_decay=1e-4,
        )
        criterion = nn.HuberLoss(delta=1.0)
        best_rmse = float("inf")
        started = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for image, satellite_id, temporal, missing, target, _ in tqdm(
                train_loader, desc=f"{name} epoch {epoch}", leave=False
            ):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    image.to(device),
                    satellite_id.to(device),
                    temporal.to(device),
                    missing.to(device),
                )
                loss = criterion(prediction, target.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * image.shape[0]
                seen += image.shape[0]

            validation_rmse = evaluate(model, validation_loader, device)
            best_rmse = min(best_rmse, validation_rmse)
            print(
                f"{name}: epoch={epoch} train_huber={running_loss/seen:.5f} "
                f"val_rmse={validation_rmse:.5f}"
            )

        results.append(
            {
                "experiment": name,
                "best_validation_rmse": best_rmse,
                "seconds": time.time() - started,
                **flags,
            }
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    result_frame = pd.DataFrame(results).sort_values("best_validation_rmse")
    result_frame.to_csv(output_path, index=False)
    (output_path.parent / "config.json").write_text(
        json.dumps(vars(args), indent=2)
    )
    print(result_frame.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
