"""Lightweight U-Net baseline compatible with the Swin v2 data pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from swin_nowcast_v2 import (
    SATELLITES,
    Config,
    NowcastingDataset,
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


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class SatelliteInputStem(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.stems = nn.ModuleList(
            [ConvBlock(input_channels, output_channels) for _ in SATELLITES]
        )

    def forward(
        self, image: torch.Tensor, satellite_id: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        indices = []
        for identifier, stem in enumerate(self.stems):
            index = torch.nonzero(
                satellite_id == identifier, as_tuple=False
            ).flatten()
            if len(index):
                outputs.append(stem(image[index]))
                indices.append(index)
        combined_indices = torch.cat(indices)
        combined_outputs = torch.cat(outputs, dim=0)
        return combined_outputs[torch.argsort(combined_indices)]


class UNetNowcaster(nn.Module):
    def __init__(self, config: Config, base_channels: int = 32) -> None:
        super().__init__()
        band_count = len(next(iter(get_band_mapping(config).values())))
        input_channels = band_count * config.max_observations
        self.config = config
        self.stem = SatelliteInputStem(input_channels, base_channels)
        self.shared_stem = ConvBlock(input_channels, base_channels)

        self.encoder1 = ConvBlock(base_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.context = nn.Sequential(
            nn.Linear(5, base_channels * 8),
            nn.SiLU(),
            nn.Linear(base_channels * 8, base_channels * 8),
        )
        self.satellite_embedding = nn.Embedding(
            len(SATELLITES), base_channels * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)

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

        encoder1 = self.encoder1(image)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        temporal = temporal_features.clone()
        if not self.config.use_month_features:
            temporal[:, :2] = 0
        if not self.config.use_hour_features:
            temporal[:, 2:] = 0
        if not self.config.use_missing_flag:
            missing_flag = torch.zeros_like(missing_flag)
        condition = self.context(torch.cat([temporal, missing_flag], dim=1))
        if self.config.use_satellite_embedding:
            condition = condition + self.satellite_embedding(satellite_id)
        bottleneck = bottleneck + condition[:, :, None, None]

        decoder3 = self.up3(bottleneck)
        decoder3 = self.decoder3(torch.cat([decoder3, encoder3], dim=1))
        decoder2 = self.up2(decoder3)
        decoder2 = self.decoder2(torch.cat([decoder2, encoder2], dim=1))
        decoder1 = self.up1(decoder2)
        decoder1 = self.decoder1(torch.cat([decoder1, encoder1], dim=1))
        prediction = self.head(decoder1)
        return F.interpolate(
            prediction,
            size=(self.config.target_size, self.config.target_size),
            mode="bilinear",
            align_corners=False,
        )


def unet_model_dir(config: Config) -> Path:
    return config.root_path / "models" / config.unet_model_subdir


def train_unet_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: dict,
    device: torch.device | None = None,
    base_channels: int = 32,
) -> dict:
    device = device or get_device()
    fold_index = fold["fold"]
    seed_everything(config.seed + fold_index)
    model_dir = unet_model_dir(config)
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

    model = UNetNowcaster(config, base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_head, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.HuberLoss(delta=config.huber_delta)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = model_dir / f"best_fold{fold_index}.pth"
    best_rmse = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            train_loader,
            desc=f"U-Net fold {fold_index} epoch {epoch}",
            leave=False,
        ):
            image = image.to(device, non_blocking=True)
            satellite_id = satellite_id.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
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
                    image.to(device),
                    satellite_id.to(device),
                    temporal.to(device),
                    missing.to(device),
                ).cpu()
                error, pixels = original_scale_rmse(prediction, target)
                squared_error += error
                pixel_count += pixels
        scheduler.step()
        validation_rmse = math.sqrt(squared_error / pixel_count)
        history.append(
            {
                "epoch": epoch,
                "train_huber": loss_sum / sample_count,
                "validation_rmse": validation_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"fold={fold_index} epoch={epoch:02d} "
            f"train_huber={loss_sum/sample_count:.5f} "
            f"val_rmse={validation_rmse:.5f}"
        )
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            torch.save(
                {
                    "config": asdict(config),
                    "base_channels": base_channels,
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


def predict_unet_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: int,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_device()
    model_dir = unet_model_dir(config)
    checkpoint = torch.load(
        model_dir / f"best_fold{fold}.pth",
        map_location=device,
        weights_only=False,
    )
    model = UNetNowcaster(
        config, base_channels=checkpoint.get("base_channels", 32)
    ).to(device)
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
            loader, desc=f"U-Net inference fold {fold}"
        ):
            prediction = model(
                image.to(device),
                satellite_id.to(device),
                temporal.to(device),
                missing.to(device),
            )
            predictions.append(
                torch.expm1(prediction).clamp(min=0).cpu().numpy()[:, 0]
            )
    result = np.concatenate(predictions)
    np.save(model_dir / f"predictions_fold{fold}.npy", result.astype(np.float32))
    return result


def ensemble_unet_predict(
    config: Config,
    dataframe: pd.DataFrame,
    folds: list[int],
    device: torch.device | None = None,
) -> np.ndarray:
    predictions = [
        predict_unet_fold(config, dataframe, fold, device) for fold in folds
    ]
    result = np.mean(predictions, axis=0)
    np.save(
        unet_model_dir(config) / "predictions_ensemble.npy",
        result.astype(np.float32),
    )
    return result
