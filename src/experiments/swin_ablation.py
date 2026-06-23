"""Swin ablation experiment for Kaggle notebook execution."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from experiment_pipelines import evaluate_rmse
from experiment_utils import balanced_sample, build_folds, load_train_dataframe
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    SwinNowcaster,
    compute_band_stats,
    get_band_mapping,
    get_device,
    load_stats,
    make_loader,
    save_stats,
    satellite_directories,
    seed_everything,
)


EXPERIMENTS = {
    "matched6_full": {
        "band_mode": "matched6",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
        "use_month_features": True,
        "use_hour_features": True,
        "use_missing_flag": True,
    },
    "matched6_shared": {
        "band_mode": "matched6",
        "use_satellite_normalization": False,
        "use_satellite_stem": False,
        "use_satellite_embedding": False,
        "use_month_features": True,
        "use_hour_features": True,
        "use_missing_flag": True,
    },
    "legacy3_full": {
        "band_mode": "legacy3",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
        "use_month_features": True,
        "use_hour_features": True,
        "use_missing_flag": True,
    },
    "matched6_no_hour": {
        "band_mode": "matched6",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
        "use_month_features": True,
        "use_hour_features": False,
        "use_missing_flag": True,
    },
    "matched6_no_month": {
        "band_mode": "matched6",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
        "use_month_features": False,
        "use_hour_features": True,
        "use_missing_flag": True,
    },
    "matched6_no_missing": {
        "band_mode": "matched6",
        "use_satellite_normalization": True,
        "use_satellite_stem": True,
        "use_satellite_embedding": True,
        "use_month_features": True,
        "use_hour_features": True,
        "use_missing_flag": False,
    },
}


def run(args) -> Path:
    root = Path(args.root).resolve()
    device = get_device()
    read_stats_dir = (
        Path(args.band_stats_root).resolve() if args.band_stats_root else None
    )
    write_stats_dir = root / "outputs" / "band_stats"
    experiment_output_dir = root / "outputs" / "swin_ablation"
    write_stats_dir.mkdir(parents=True, exist_ok=True)
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    base_config = Config(
        root=str(root),
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=0,
        pretrained=False,
        use_amp=False,
        seed=args.seed,
        band_stats_root=str(write_stats_dir),
    )
    dataframe = load_train_dataframe(base_config)
    fold = build_folds(base_config, dataframe)[args.fold]
    train_frame = balanced_sample(
        dataframe.iloc[fold["train_indices"]], args.train_rows, args.seed
    )
    validation_frame = balanced_sample(
        dataframe.iloc[fold["validation_indices"]],
        args.validation_rows,
        args.seed + 1,
    )

    directories = satellite_directories(base_config, "train")
    stats_by_band_mode = {}
    for band_mode in ("matched6", "legacy3"):
        stats_filename = f"swin_ablation_{band_mode}_fold{args.fold}.json"
        write_stats_path = write_stats_dir / stats_filename
        read_stats_path = (
            read_stats_dir / stats_filename if read_stats_dir is not None else None
        )
        config = replace(base_config, band_mode=band_mode)
        if write_stats_path.exists():
            stats = load_stats(write_stats_path)
        elif read_stats_path is not None and read_stats_path.exists():
            stats = load_stats(read_stats_path)
            if "shared" not in stats:
                raise ValueError(
                    f"Cached stats do not contain shared normalization: {read_stats_path}"
                )
        else:
            stats = compute_band_stats(
                dataframe.iloc[fold["train_indices"]],
                directories,
                max_samples_per_satellite=args.stats_rows_per_satellite,
                seed=args.seed,
                band_mapping=get_band_mapping(config),
                include_shared=True,
            )
            save_stats(stats, write_stats_path)
        stats_by_band_mode[band_mode] = stats

    results = []
    for name, flags in EXPERIMENTS.items():
        config = replace(base_config, **flags)
        seed_everything(args.seed)
        stats = stats_by_band_mode[config.band_mode]
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

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        best_rmse = float("inf")
        history = []
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

            validation_rmse = evaluate_rmse(model, validation_loader, device)
            best_rmse = min(best_rmse, validation_rmse)
            history.append(
                {
                    "epoch": epoch,
                    "train_huber": running_loss / seen,
                    "validation_rmse": validation_rmse,
                }
            )
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
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(
            experiment_output_dir / f"{name}_fold{args.fold}.csv",
            index=False,
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results).sort_values("best_validation_rmse")
    result_frame.to_csv(output_path, index=False)
    (output_path.parent / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(result_frame.to_string(index=False))
    print(f"Saved: {output_path}")
    return output_path
