"""ConvNeXt-Tiny + FPN precipitation nowcasting pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from swin_nowcast_v2 import (
    SATELLITES,
    Config,
    NowcastingDataset,
    SatelliteStem,
    compute_band_stats,
    get_band_mapping,
    get_device,
    load_stats,
    make_loader,
    original_scale_rmse,
    satellite_directories,
    save_stats,
    seed_everything,
)


class NCHWFPNDecoder(nn.Module):
    def __init__(self, input_channels: list[int], output_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, output_channels, 1) for channels in input_channels]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                    nn.GELU(),
                )
                for _ in input_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                output_channels * len(input_channels),
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        lateral = [
            layer(feature) for layer, feature in zip(self.lateral, features)
        ]
        for index in range(len(lateral) - 2, -1, -1):
            lateral[index] = lateral[index] + F.interpolate(
                lateral[index + 1],
                size=lateral[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        smoothed = [
            layer(feature) for layer, feature in zip(self.smooth, lateral)
        ]
        output_size = smoothed[0].shape[-2:]
        aligned = [
            F.interpolate(
                feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            for feature in smoothed
        ]
        return self.fuse(torch.cat(aligned, dim=1))


class ConvNeXtNowcaster(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        band_count = len(next(iter(get_band_mapping(config).values())))
        input_channels = band_count * config.max_observations
        self.config = config
        self.stem = SatelliteStem(input_channels, config.stem_channels)
        self.shared_stem = nn.Sequential(
            nn.Conv2d(
                input_channels, config.stem_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(config.stem_channels),
            nn.GELU(),
            nn.Conv2d(
                config.stem_channels,
                config.stem_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.stem_channels),
            nn.GELU(),
        )
        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=config.pretrained,
            features_only=True,
            in_chans=config.stem_channels,
        )
        self.decoder = NCHWFPNDecoder(
            self.encoder.feature_info.channels(), config.decoder_channels
        )
        condition_channels = config.decoder_channels
        self.satellite_embedding = nn.Embedding(
            len(SATELLITES), condition_channels
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(5, condition_channels),
            nn.GELU(),
            nn.Linear(condition_channels, condition_channels),
        )
        self.head = nn.Sequential(
            nn.Conv2d(condition_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        satellite_id: torch.Tensor,
        temporal_features: torch.Tensor,
        missing_flag: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.use_satellite_stem:
            image = self.stem(image, satellite_id)
        else:
            image = self.shared_stem(image)
        decoded = self.decoder(self.encoder(image))

        temporal = temporal_features.clone()
        if not self.config.use_month_features:
            temporal[:, :2] = 0
        if not self.config.use_hour_features:
            temporal[:, 2:] = 0
        if not self.config.use_missing_flag:
            missing_flag = torch.zeros_like(missing_flag)
        condition = self.context_mlp(
            torch.cat([temporal, missing_flag], dim=1)
        )
        if self.config.use_satellite_embedding:
            condition = condition + self.satellite_embedding(satellite_id)
        prediction = self.head(decoded + condition[:, :, None, None])
        return F.interpolate(
            prediction,
            size=(self.config.target_size, self.config.target_size),
            mode="bilinear",
            align_corners=False,
        )


def convnext_model_dir(config: Config) -> Path:
    return config.root_path / "models" / config.convnext_model_subdir


def train_convnext_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: dict,
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    fold_index = fold["fold"]
    seed_everything(config.seed + fold_index)
    model_dir = convnext_model_dir(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_frame = dataframe.iloc[fold["train_indices"]].copy()
    validation_frame = dataframe.iloc[fold["validation_indices"]].copy()
    directories = satellite_directories(config, "train")
    stats_path = model_dir / f"band_stats_fold{fold_index}.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
    else:
        stats = compute_band_stats(
            train_frame,
            directories,
            config.stats_samples_per_satellite,
            config.seed + fold_index,
            band_mapping=get_band_mapping(config),
        )
        save_stats(stats, stats_path)

    train_dataset = NowcastingDataset(
        train_frame, directories, stats, config, has_target=True, augment=True
    )
    validation_dataset = NowcastingDataset(
        validation_frame, directories, stats, config, has_target=True
    )
    train_loader = make_loader(train_dataset, config, device, shuffle=True)
    validation_loader = make_loader(
        validation_dataset, config, device, shuffle=False
    )
    model = ConvNeXtNowcaster(config).to(device)
    head_parameters = (
        list(model.stem.parameters())
        + list(model.shared_stem.parameters())
        + list(model.decoder.parameters())
        + list(model.satellite_embedding.parameters())
        + list(model.context_mlp.parameters())
        + list(model.head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.lr_encoder},
            {"params": head_parameters, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.HuberLoss(delta=config.huber_delta)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = model_dir / f"best_fold{fold_index}.pth"
    history = []
    best_rmse = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            train_loader,
            desc=f"ConvNeXt fold {fold_index} epoch {epoch}",
            leave=False,
        ):
            image = image.to(device, non_blocking=True)
            satellite_id = satellite_id.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(image, satellite_id, temporal, missing)
                loss = criterion(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * image.shape[0]
            sample_count += image.shape[0]

        model.eval()
        squared_error = 0.0
        pixel_count = 0
        with torch.no_grad():
            for image, satellite_id, temporal, missing, target, _ in validation_loader:
                prediction = model(
                    image.to(device, non_blocking=True),
                    satellite_id.to(device, non_blocking=True),
                    temporal.to(device, non_blocking=True),
                    missing.to(device, non_blocking=True),
                ).cpu()
                error, pixels = original_scale_rmse(prediction, target)
                squared_error += error
                pixel_count += pixels
        scheduler.step()
        validation_rmse = math.sqrt(squared_error / pixel_count)
        train_huber = loss_sum / sample_count
        history.append(
            {
                "epoch": epoch,
                "train_huber": train_huber,
                "validation_rmse": validation_rmse,
                "lr_encoder": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
            }
        )
        print(
            f"fold={fold_index} epoch={epoch:02d} "
            f"train_huber={train_huber:.5f} val_rmse={validation_rmse:.5f}"
        )
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            torch.save(
                {
                    "config": asdict(config),
                    "fold": fold_index,
                    "validation_locations": fold["validation_locations"],
                    "validation_rmse": validation_rmse,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

    pd.DataFrame(history).to_csv(
        model_dir / f"history_fold{fold_index}.csv", index=False
    )
    return {
        "fold": fold_index,
        "validation_rmse": best_rmse,
        "checkpoint": str(checkpoint_path),
    }


def predict_convnext_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: int,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_device()
    model_dir = convnext_model_dir(config)
    checkpoint = torch.load(
        model_dir / f"best_fold{fold}.pth",
        map_location=device,
        weights_only=False,
    )
    model = ConvNeXtNowcaster(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = load_stats(model_dir / f"band_stats_fold{fold}.json")
    dataset = NowcastingDataset(
        dataframe,
        satellite_directories(config, "evaluation"),
        stats,
        config,
        has_target=False,
    )
    loader = make_loader(dataset, config, device, shuffle=False)
    predictions = []
    with torch.no_grad():
        for image, satellite_id, temporal, missing, _, _ in tqdm(
            loader, desc=f"ConvNeXt inference fold {fold}"
        ):
            prediction = model(
                image.to(device, non_blocking=True),
                satellite_id.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                missing.to(device, non_blocking=True),
            )
            predictions.append(
                torch.expm1(prediction).clamp(min=0).cpu().numpy()[:, 0]
            )
    result = np.concatenate(predictions)
    np.save(
        model_dir / f"predictions_fold{fold}.npy", result.astype(np.float32)
    )
    return result


def ensemble_convnext_predict(
    config: Config,
    dataframe: pd.DataFrame,
    folds: list[int],
    device: torch.device | None = None,
) -> np.ndarray:
    predictions = [
        predict_convnext_fold(config, dataframe, fold, device) for fold in folds
    ]
    result = np.mean(predictions, axis=0)
    np.save(
        convnext_model_dir(config) / "predictions_ensemble.npy",
        result.astype(np.float32),
    )
    return result
