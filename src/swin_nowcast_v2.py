"""Swin-T precipitation nowcasting pipeline.

Design:
- six approximately matched physical channels across AHI / ABI / FCI
- satellite-specific normalization and input stems
- satellite embedding plus cyclical UTC time features
- GroupKFold by unseen location
- fold-specific normalization statistics
- log1p target with Huber loss, evaluated by original-scale RMSE
- fold ensemble inference and exact Solafune submission structure
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from project_paths import WorkspacePaths


SATELLITES = ("himawari", "goes", "meteosat")
SATELLITE_TO_ID = {name: idx for idx, name in enumerate(SATELLITES)}

# Channel order: 1.6, 2.2, 6.3, 7.3, 10.5, 12.3 micrometers.
# These are nearest available channels, not physically identical SRFs.
SATELLITE_BANDS = {
    "himawari": (5, 6, 8, 10, 13, 15),
    "goes": (5, 6, 8, 10, 13, 15),
    "meteosat": (7, 8, 10, 11, 14, 15),
}
LEGACY_THREE_BANDS = {satellite: (1, 2, 3) for satellite in SATELLITES}


@dataclass
class Config:
    root: str
    encoder_size: int = 224
    target_size: int = 41
    max_observations: int = 3
    stem_channels: int = 32
    decoder_channels: int = 192
    batch_size: int = 8
    epochs: int = 30
    n_folds: int = 5
    lr_encoder: float = 5e-5
    lr_head: float = 2e-4
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    workers: int = 2
    stats_samples_per_satellite: int | None = 1500
    seed: int = 42
    pretrained: bool = True
    use_amp: bool = True
    use_satellite_stem: bool = True
    use_satellite_embedding: bool = True
    use_month_features: bool = True
    use_hour_features: bool = True
    use_missing_flag: bool = True
    use_temporal_differences: bool = False
    use_temporal_summary: bool = False
    band_mode: str = "matched6"
    use_satellite_normalization: bool = True
    unet_model_subdir: str = "unet_v2"
    convnext_model_subdir: str = "convnext_v2"
    band_stats_root: str | None = None

    @property
    def paths(self) -> WorkspacePaths:
        return WorkspacePaths(self.root)

    @property
    def model_dir(self) -> Path:
        return self.paths.models_dir / "swin_v2"

    @property
    def band_stats_dir(self) -> Path:
        stats_root = self.band_stats_root or os.environ.get("SOLAFUNE_BAND_STATS_ROOT")
        return Path(stats_root) if stats_root else self.model_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["observation_files"] = df["last_30_minutes_observation_filename"].map(ast.literal_eval)
    df["missing_observation"] = df["observation_files"].map(lambda values: len(values) == 0)
    return df


def time_features(timestamp: pd.Timestamp) -> np.ndarray:
    month_phase = 2 * math.pi * (timestamp.month - 1) / 12
    hour = timestamp.hour + timestamp.minute / 60
    hour_phase = 2 * math.pi * hour / 24
    return np.asarray(
        [
            math.sin(month_phase),
            math.cos(month_phase),
            math.sin(hour_phase),
            math.cos(hour_phase),
        ],
        dtype=np.float32,
    )


def satellite_directories(config: Config, split: str) -> dict[str, Path]:
    base = config.paths.train_dir if split == "train" else config.paths.evaluation_dir
    return {satellite: base / satellite for satellite in SATELLITES}


def get_band_mapping(config: Config) -> dict[str, tuple[int, ...]]:
    if config.band_mode == "legacy3":
        return LEGACY_THREE_BANDS
    if config.band_mode == "matched6":
        return SATELLITE_BANDS
    raise ValueError(f"Unknown band_mode: {config.band_mode}")


def compute_band_stats(
    dataframe: pd.DataFrame,
    directories: dict[str, Path],
    max_samples_per_satellite: int | None,
    seed: int,
    band_mapping: dict[str, tuple[int, ...]] | None = None,
    include_shared: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute fold-train-only statistics for matched physical channels."""
    band_mapping = band_mapping or SATELLITE_BANDS
    stats: dict[str, dict[str, np.ndarray]] = {}
    aggregate_sums = None
    aggregate_squared_sums = None
    aggregate_counts = None

    for satellite in SATELLITES:
        rows = dataframe[dataframe["satellite_target"] == satellite]
        rows = rows[~rows["missing_observation"]]
        if max_samples_per_satellite and len(rows) > max_samples_per_satellite:
            rows = rows.sample(max_samples_per_satellite, random_state=seed)

        bands = band_mapping[satellite]
        sums = np.zeros(len(bands), dtype=np.float64)
        squared_sums = np.zeros_like(sums)
        counts = np.zeros_like(sums)

        for row in tqdm(
            rows.itertuples(index=False),
            total=len(rows),
            desc=f"Stats: {satellite}",
            leave=False,
        ):
            observation_files = row.observation_files
            # Use all available timestamps so the statistics match model input.
            for filename in observation_files:
                path = directories[satellite] / filename
                with rasterio.open(path) as src:
                    if src.count < max(bands):
                        continue
                    data = src.read(bands).astype(np.float64)
                flat = data.reshape(data.shape[0], -1)
                sums += flat.sum(axis=1)
                squared_sums += np.square(flat).sum(axis=1)
                counts += flat.shape[1]

        if not np.all(counts):
            raise RuntimeError(f"No valid pixels found while computing stats for {satellite}")

        mean = sums / counts
        variance = np.maximum(squared_sums / counts - np.square(mean), 1e-6)
        stats[satellite] = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
        }
        if aggregate_sums is None:
            aggregate_sums = sums.copy()
            aggregate_squared_sums = squared_sums.copy()
            aggregate_counts = counts.copy()
        else:
            aggregate_sums += sums
            aggregate_squared_sums += squared_sums
            aggregate_counts += counts

    if include_shared:
        mean = aggregate_sums / aggregate_counts
        variance = np.maximum(
            aggregate_squared_sums / aggregate_counts - np.square(mean), 1e-6
        )
        stats["shared"] = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
        }

    return stats


def save_stats(stats: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    serializable = {
        satellite: {key: value.tolist() for key, value in values.items()}
        for satellite, values in stats.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text())
    return {
        satellite: {
            key: np.asarray(value, dtype=np.float32) for key, value in values.items()
        }
        for satellite, values in raw.items()
    }


class NowcastingDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        directories: dict[str, Path],
        stats: dict[str, dict[str, np.ndarray]],
        config: Config,
        has_target: bool,
        augment: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.directories = directories
        self.stats = stats
        self.config = config
        self.has_target = has_target
        self.augment = augment
        self.gpm_dir = config.paths.gpm_dir
        self.band_mapping = get_band_mapping(config)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _read_observation(self, path: Path, satellite: str) -> np.ndarray:
        bands = self.band_mapping[satellite]
        with rasterio.open(path) as src:
            if src.count < max(bands):
                return np.zeros(
                    (
                        len(bands),
                        self.config.encoder_size,
                        self.config.encoder_size,
                    ),
                    dtype=np.float32,
                )
            data = src.read(bands).astype(np.float32)
        stats_key = satellite if self.config.use_satellite_normalization else "shared"
        mean = self.stats[stats_key]["mean"][:, None, None]
        std = self.stats[stats_key]["std"][:, None, None]
        data = (data - mean) / std
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.config.encoder_size, self.config.encoder_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).numpy()

    def _read_target(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            target = src.read(1).astype(np.float32)
            if src.nodata is not None:
                target[target == src.nodata] = 0
        target = np.clip(target, 0, None)
        return np.log1p(target)[None]

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        satellite = row["satellite_target"]
        files = row["observation_files"]
        missing = float(len(files) == 0)

        observations = []
        for observation_index in range(self.config.max_observations):
            if observation_index < len(files):
                path = self.directories[satellite] / files[observation_index]
                observations.append(self._read_observation(path, satellite))
            else:
                observations.append(
                    np.zeros(
                        (
                            len(self.band_mapping[satellite]),
                            self.config.encoder_size,
                            self.config.encoder_size,
                        ),
                        dtype=np.float32,
                    )
                )
        observation_stack = np.stack(observations, axis=0)
        image_features = [observation_stack.reshape(-1, *observation_stack.shape[-2:])]
        if self.config.use_temporal_differences:
            differences = np.diff(observation_stack, axis=0)
            image_features.append(differences.reshape(-1, *differences.shape[-2:]))
        if self.config.use_temporal_summary:
            image_features.extend(
                [
                    observation_stack.mean(axis=0),
                    observation_stack.std(axis=0),
                ]
            )
        image = np.concatenate(image_features, axis=0)

        if self.has_target:
            target = self._read_target(self.gpm_dir / row["gpm_imerg_filename"])
        else:
            target = np.zeros(
                (1, self.config.target_size, self.config.target_size), dtype=np.float32
            )

        if self.augment and self.has_target:
            if np.random.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                target = np.flip(target, axis=2).copy()
            if np.random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                target = np.flip(target, axis=1).copy()
            rotation = np.random.randint(4)
            if rotation:
                image = np.rot90(image, rotation, axes=(1, 2)).copy()
                target = np.rot90(target, rotation, axes=(1, 2)).copy()

        metadata = {
            "unique_id": row["unique_id"],
            "gpm_imerg_filename": row["gpm_imerg_filename"],
        }
        return (
            torch.from_numpy(image),
            torch.tensor(SATELLITE_TO_ID[satellite], dtype=torch.long),
            torch.from_numpy(time_features(row["datetime"])),
            torch.tensor([missing], dtype=torch.float32),
            torch.from_numpy(target),
            metadata,
        )


class SatelliteStem(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                )
                for _ in SATELLITES
            ]
        )

    def forward(self, image: torch.Tensor, satellite_id: torch.Tensor) -> torch.Tensor:
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


class FPNDecoder(nn.Module):
    def __init__(self, input_channels: Iterable[int], output_channels: int) -> None:
        super().__init__()
        input_channels = list(input_channels)
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
            nn.Conv2d(output_channels * len(input_channels), output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # timm Swin emits NHWC tensors.
        features = [feature.permute(0, 3, 1, 2).contiguous() for feature in features]
        lateral = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(lateral) - 2, -1, -1):
            lateral[index] = lateral[index] + F.interpolate(
                lateral[index + 1],
                size=lateral[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        smoothed = [layer(feature) for layer, feature in zip(self.smooth, lateral)]
        target_size = smoothed[0].shape[-2:]
        aligned = [
            F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
            for feature in smoothed
        ]
        return self.fuse(torch.cat(aligned, dim=1))


class SwinNowcaster(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        band_mapping = get_band_mapping(config)
        band_count = len(next(iter(band_mapping.values())))
        raw_channels = band_count * config.max_observations
        if config.use_temporal_differences:
            raw_channels += band_count * (config.max_observations - 1)
        if config.use_temporal_summary:
            raw_channels += band_count * 2
        self.config = config
        self.stem = SatelliteStem(raw_channels, config.stem_channels)
        self.shared_stem = nn.Sequential(
            nn.Conv2d(raw_channels, config.stem_channels, 3, padding=1, bias=False),
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
            "swin_tiny_patch4_window7_224",
            pretrained=config.pretrained,
            features_only=True,
            in_chans=config.stem_channels,
        )
        feature_channels = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(feature_channels, config.decoder_channels)
        condition_size = config.decoder_channels
        self.satellite_embedding = nn.Embedding(len(SATELLITES), condition_size)
        self.context_mlp = nn.Sequential(
            nn.Linear(5, condition_size),
            nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.head = nn.Sequential(
            nn.Conv2d(config.decoder_channels, 64, 3, padding=1),
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
        features = self.encoder(image)
        decoded = self.decoder(features)
        context = temporal_features.clone()
        if not self.config.use_month_features:
            context[:, :2] = 0
        if not self.config.use_hour_features:
            context[:, 2:] = 0
        if not self.config.use_missing_flag:
            missing_flag = torch.zeros_like(missing_flag)
        condition = self.context_mlp(torch.cat([context, missing_flag], dim=1))
        if self.config.use_satellite_embedding:
            condition = condition + self.satellite_embedding(satellite_id)
        decoded = decoded + condition[:, :, None, None]
        prediction = self.head(decoded)
        return F.interpolate(
            prediction,
            size=(self.config.target_size, self.config.target_size),
            mode="bilinear",
            align_corners=False,
        )


def make_folds(dataframe: pd.DataFrame, n_folds: int) -> list[dict]:
    """Location-disjoint folds stratified by satellite.

    Each location belongs to exactly one satellite in this dataset. Plain
    GroupKFold balances row counts but can produce a validation fold containing
    only one satellite. This greedy assignment distributes locations from each
    satellite across folds while balancing per-satellite sample counts.
    """
    location_table = (
        dataframe.groupby(["satellite_target", "name_location"])
        .size()
        .rename("rows")
        .reset_index()
    )
    assignments: list[list[str]] = [[] for _ in range(n_folds)]
    fold_satellite_rows = {
        satellite: np.zeros(n_folds, dtype=np.int64) for satellite in SATELLITES
    }
    fold_total_rows = np.zeros(n_folds, dtype=np.int64)

    for satellite in SATELLITES:
        locations = location_table[
            location_table["satellite_target"] == satellite
        ].sort_values(["rows", "name_location"], ascending=[False, True])
        if len(locations) < n_folds:
            raise ValueError(
                f"{satellite} has only {len(locations)} locations for {n_folds} folds"
            )
        for row in locations.itertuples(index=False):
            candidate = min(
                range(n_folds),
                key=lambda fold: (
                    fold_satellite_rows[satellite][fold],
                    fold_total_rows[fold],
                    fold,
                ),
            )
            assignments[candidate].append(row.name_location)
            fold_satellite_rows[satellite][candidate] += row.rows
            fold_total_rows[candidate] += row.rows

    folds = []
    all_indices = np.arange(len(dataframe))
    for fold, validation_locations in enumerate(assignments):
        validation_mask = dataframe["name_location"].isin(validation_locations).to_numpy()
        validation_indices = all_indices[validation_mask]
        train_indices = all_indices[~validation_mask]
        folds.append(
            {
                "fold": fold,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "validation_locations": sorted(validation_locations),
            }
        )
    return folds


def make_loader(
    dataset: Dataset,
    config: Config,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0 and device.type == "cuda",
    )


def original_scale_rmse(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[float, int]:
    predictions = torch.expm1(predictions).clamp(min=0)
    targets = torch.expm1(targets).clamp(min=0)
    squared_error = F.mse_loss(predictions, targets, reduction="sum").item()
    return squared_error, targets.numel()


def train_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: dict,
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    seed_everything(config.seed + fold["fold"])
    config.model_dir.mkdir(parents=True, exist_ok=True)

    train_frame = dataframe.iloc[fold["train_indices"]].copy()
    validation_frame = dataframe.iloc[fold["validation_indices"]].copy()
    train_directories = satellite_directories(config, "train")
    stats_path = config.band_stats_dir / f"band_stats_fold{fold['fold']}.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
    else:
        stats = compute_band_stats(
            train_frame,
            train_directories,
            config.stats_samples_per_satellite,
            config.seed + fold["fold"],
        )
        save_stats(stats, stats_path)

    train_dataset = NowcastingDataset(
        train_frame, train_directories, stats, config, has_target=True, augment=True
    )
    validation_dataset = NowcastingDataset(
        validation_frame,
        train_directories,
        stats,
        config,
        has_target=True,
        augment=False,
    )
    train_loader = make_loader(train_dataset, config, device, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, device, shuffle=False)

    model = SwinNowcaster(config).to(device)
    head_parameters = (
        list(model.stem.parameters())
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

    checkpoint_path = config.model_dir / f"best_fold{fold['fold']}.pth"
    history = []
    best_rmse = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for image, satellite_id, temporal, missing, target, _ in tqdm(
            train_loader, desc=f"Fold {fold['fold']} epoch {epoch}", leave=False
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
                image = image.to(device, non_blocking=True)
                satellite_id = satellite_id.to(device, non_blocking=True)
                temporal = temporal.to(device, non_blocking=True)
                missing = missing.to(device, non_blocking=True)
                prediction = model(image, satellite_id, temporal, missing).cpu()
                batch_squared_error, batch_pixels = original_scale_rmse(
                    prediction, target
                )
                squared_error += batch_squared_error
                pixel_count += batch_pixels

        scheduler.step()
        validation_rmse = math.sqrt(squared_error / pixel_count)
        train_huber = loss_sum / sample_count
        row = {
            "epoch": epoch,
            "train_huber": train_huber,
            "validation_rmse": validation_rmse,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"fold={fold['fold']} epoch={epoch:02d} "
            f"train_huber={train_huber:.5f} val_rmse={validation_rmse:.5f}"
        )

        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            torch.save(
                {
                    "config": asdict(config),
                    "fold": fold["fold"],
                    "validation_locations": fold["validation_locations"],
                    "validation_rmse": validation_rmse,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        config.model_dir / f"history_fold{fold['fold']}.csv", index=False
    )
    return {
        "fold": fold["fold"],
        "validation_rmse": best_rmse,
        "checkpoint": str(checkpoint_path),
        "stats": str(stats_path),
        "history": history_frame,
    }


def plot_histories(results: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        history = result["history"]
        axis.plot(history["epoch"], history["validation_rmse"], label="validation RMSE")
        axis.set_title(f"Fold {result['fold']}: {result['validation_rmse']:.4f}")
        axis.set_xlabel("epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()


def load_fold_model(
    config: Config, fold: int, device: torch.device
) -> tuple[SwinNowcaster, dict[str, dict[str, np.ndarray]]]:
    checkpoint = torch.load(
        config.model_dir / f"best_fold{fold}.pth",
        map_location=device,
        weights_only=False,
    )
    model = SwinNowcaster(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = load_stats(config.band_stats_dir / f"band_stats_fold{fold}.json")
    return model, stats


def predict_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: int,
    device: torch.device,
) -> np.ndarray:
    model, stats = load_fold_model(config, fold, device)
    dataset = NowcastingDataset(
        dataframe,
        satellite_directories(config, "evaluation"),
        stats,
        config,
        has_target=False,
        augment=False,
    )
    loader = make_loader(dataset, config, device, shuffle=False)
    predictions = []
    with torch.no_grad():
        for image, satellite_id, temporal, missing, _, _ in tqdm(
            loader, desc=f"Inference fold {fold}"
        ):
            prediction = model(
                image.to(device, non_blocking=True),
                satellite_id.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                missing.to(device, non_blocking=True),
            )
            prediction = torch.expm1(prediction).clamp(min=0).cpu().numpy()
            predictions.append(prediction[:, 0])
    return np.concatenate(predictions)


def ensemble_predict(
    config: Config,
    dataframe: pd.DataFrame,
    folds: Iterable[int],
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_device()
    prediction_sum = None
    fold_count = 0
    for fold in folds:
        fold_predictions = predict_fold(config, dataframe, fold, device)
        prediction_sum = (
            fold_predictions
            if prediction_sum is None
            else prediction_sum + fold_predictions
        )
        fold_count += 1
    if fold_count == 0:
        raise ValueError("At least one fold is required for inference")
    return prediction_sum / fold_count


def create_submission(
    config: Config,
    evaluation_frame: pd.DataFrame,
    predictions: np.ndarray,
    output_zip: Path,
) -> Path:
    """Create exact root/evaluation_target.csv + root/test_files/*.tif layout."""
    if len(evaluation_frame) != len(predictions):
        raise ValueError("Prediction count does not match evaluation rows")

    submission_dir = output_zip.with_suffix("")
    test_files_dir = submission_dir / "test_files"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    test_files_dir.mkdir(parents=True)

    # Preserve all supplied columns and row order exactly.
    source_csv = config.paths.evaluation_dir / "evaluation_target.csv"
    shutil.copy2(source_csv, submission_dir / "evaluation_target.csv")

    reference_dirs = [
        config.paths.sample_submission_dir / "test_files",
        config.paths.evaluation_dir / "test_files",
    ]
    for prediction, row in tqdm(
        zip(predictions, evaluation_frame.itertuples(index=False)),
        total=len(evaluation_frame),
        desc="Writing submission TIFs",
    ):
        filename = row.gpm_imerg_filename
        reference = next(
            (directory / filename for directory in reference_dirs if (directory / filename).is_file()),
            None,
        )
        if reference is None:
            raise FileNotFoundError(f"Reference target profile not found: {filename}")
        with rasterio.open(reference) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1)
        with rasterio.open(test_files_dir / filename, "w", **profile) as dst:
            dst.write(prediction.astype(np.float32), 1)

    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_dir / "evaluation_target.csv", "evaluation_target.csv")
        for path in sorted(test_files_dir.glob("*.tif")):
            archive.write(path, f"test_files/{path.name}")

    validate_submission(config, evaluation_frame, output_zip)
    return output_zip


def validate_submission(
    config: Config, evaluation_frame: pd.DataFrame, submission_zip: Path
) -> None:
    expected_names = set(evaluation_frame["gpm_imerg_filename"])
    with zipfile.ZipFile(submission_zip) as archive:
        names = archive.namelist()
        if "evaluation_target.csv" not in names:
            raise ValueError("evaluation_target.csv is missing from ZIP root")
        tif_names = {
            name.removeprefix("test_files/")
            for name in names
            if name.startswith("test_files/") and name.endswith(".tif")
        }
        if tif_names != expected_names:
            missing = expected_names - tif_names
            extra = tif_names - expected_names
            raise ValueError(
                f"Submission TIF mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        with archive.open("evaluation_target.csv") as stream:
            submitted_csv = pd.read_csv(stream)
    source_csv = pd.read_csv(config.paths.evaluation_dir / "evaluation_target.csv")
    if not submitted_csv.equals(source_csv):
        raise ValueError("Submitted evaluation_target.csv differs from supplied CSV")
