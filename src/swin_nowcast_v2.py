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
import re
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
FULL16_BANDS = {satellite: tuple(range(1, 17)) for satellite in SATELLITES}
LEGACY_THREE_BANDS = {satellite: (1, 2, 3) for satellite in SATELLITES}

# Brightness-temperature differences over matched6 channel indices:
# split-window (clean IR - dirty IR), WV-IR (overshooting tops), upper-lower WV.
BTD_PAIRS = ((4, 5), (2, 4), (2, 3))
# The same physical differences as BTD_PAIRS, expressed in raw 16-band
# channel indices for each sensor-specific band ordering.
FULL16_BTD_PAIRS = {
    "himawari": ((12, 14), (7, 12), (7, 9)),
    "goes": ((12, 14), (7, 12), (7, 9)),
    "meteosat": ((13, 14), (7, 13), (7, 9)),
}
IR_WINDOW_INDEX = 4  # matched6 position of the ~10.4um IR window channel
FLOW_GRID = 112


def flow_divergence_channels(
    ir_frames: np.ndarray, output_size: int
) -> np.ndarray:
    """Cloud-top divergence maps from Farneback optical flow between IR frames.

    Divergence of the cloud-top wind field is a proxy for updraft strength
    (storm-top outflow); returns one map per adjacent frame pair.
    """
    import cv2

    def to_uint8(image: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(image, [1, 99])
        scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
        return (scaled * 255).astype(np.uint8)

    small = [
        cv2.resize(frame, (FLOW_GRID, FLOW_GRID), interpolation=cv2.INTER_AREA)
        for frame in ir_frames
    ]
    maps = []
    for earlier, later in zip(small[:-1], small[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            to_uint8(earlier), to_uint8(later), None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )
        divergence = np.gradient(flow[..., 0], axis=1) + np.gradient(
            flow[..., 1], axis=0
        )
        divergence = cv2.GaussianBlur(divergence, (0, 0), sigmaX=3)
        # bring channel scale in line with normalized bands (raw std ~0.04)
        divergence *= 10.0
        maps.append(
            cv2.resize(
                divergence, (output_size, output_size),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    return np.stack(maps).astype(np.float32)


def flow_extrapolated_channels(observation_stack: np.ndarray) -> np.ndarray:
    """Advect the latest multiband frame one timestep using IR optical flow.

    The observation frames end ten minutes before the target timestamp.  This
    creates a physically motivated estimate at the target time, while leaving
    the network free to ignore it when advection is not a useful assumption.
    """
    import cv2

    def to_uint8(image: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(image, [1, 99])
        scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
        return (scaled * 255).astype(np.uint8)

    ir_frames = [
        cv2.resize(
            frame[IR_WINDOW_INDEX], (FLOW_GRID, FLOW_GRID), interpolation=cv2.INTER_AREA
        )
        for frame in observation_stack
    ]
    flows = []
    for earlier, later in zip(ir_frames[:-1], ir_frames[1:]):
        flows.append(
            cv2.calcOpticalFlowFarneback(
                to_uint8(earlier),
                to_uint8(later),
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
        )
    flow = np.mean(flows, axis=0)
    grid_y, grid_x = np.mgrid[:FLOW_GRID, :FLOW_GRID].astype(np.float32)
    # Farneback maps a source pixel from t-20 to t-10.  remap needs the source
    # coordinate for each destination pixel, hence the inverse-flow approximation.
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    extrapolated = []
    for channel in observation_stack[-1]:
        small = cv2.resize(
            channel, (FLOW_GRID, FLOW_GRID), interpolation=cv2.INTER_AREA
        )
        advanced = cv2.remap(
            small,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        extrapolated.append(
            cv2.resize(
                advanced,
                (channel.shape[1], channel.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    return np.stack(extrapolated).astype(np.float32)


def uses_btd(config: "Config") -> bool:
    return config.band_mode in ("matched6_btd", "full16_btd")


def btd_pairs_for(config: "Config", satellite: str) -> tuple[tuple[int, int], ...]:
    if config.band_mode == "full16_btd":
        return FULL16_BTD_PAIRS[satellite]
    return BTD_PAIRS


def append_btd_channels(
    data: np.ndarray, pairs: tuple[tuple[int, int], ...] = BTD_PAIRS
) -> np.ndarray:
    btd = np.stack([data[i] - data[j] for i, j in pairs])
    return np.concatenate([data, btd], axis=0)


def input_channel_count(config: "Config", satellite: str) -> int:
    count = len(get_band_mapping(config)[satellite])
    if uses_btd(config):
        count += len(btd_pairs_for(config, satellite))
    return count


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
    early_stopping_patience: int | None = None
    early_stopping_min_epochs: int = 1
    n_folds: int = 5
    lr_encoder: float = 5e-5
    lr_head: float = 2e-4
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    loss_type: str = "huber"
    heavy_rain_weight_alpha: float = 0.5
    heavy_rain_weight_scale: float = 10.0
    heavy_rain_weight_max: float = 2.0
    raw_huber_loss_weight: float = 0.0
    raw_huber_beta: float = 5.0
    raw_huber_max: float = 100.0
    use_two_head: bool = False
    rain_bce_weight_0_1: float = 0.10
    rain_bce_weight_1: float = 0.10
    rain_bce_weight_5: float = 0.05
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
    use_temporal_frame_attention: bool = False
    use_location_features: bool = False
    location_metadata_path: str | None = None
    location_feature_mode: str = "full"
    sample_weight_path: str | None = None
    sample_weight_column: str = "weight_sqrt_clipped"
    pseudo_label_npz: str | None = None
    pseudo_label_csv: str | None = None
    pseudo_sample_weight: float = 1.0
    disable_geo_position_features: bool = False
    disable_local_time_features: bool = False
    disable_geo_season_features: bool = False
    band_mode: str = "matched6"
    use_flow_divergence: bool = False
    use_flow_extrapolation: bool = False
    encoder_name: str = "swin_tiny_patch4_window7_224"
    use_satellite_normalization: bool = True
    swin_model_subdir: str = "swin_v2"
    unet_model_subdir: str = "unet_v2"
    convnext_model_subdir: str = "convnext_v2"
    band_stats_root: str | None = None

    @property
    def paths(self) -> WorkspacePaths:
        return WorkspacePaths(self.root)

    @property
    def model_dir(self) -> Path:
        return self.paths.models_dir / self.swin_model_subdir

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
    if os.environ.get("SOLAFUNE_DEVICE") == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
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


def align_observation_frames(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Repair rows with 1-2 observation frames by timestamp-aligned slot placement.

    Missing frames (mostly the latest t-10 slot) would otherwise be zero-padded
    at the wrong temporal position, so temporal-difference channels see fake
    motion. Instead each file is placed in its true slot (t-30/t-20/t-10) and
    empty slots are filled with the nearest available frame (zero motion).
    """
    df = dataframe.copy()
    aligned = []
    for ts, files in zip(df["datetime"], df["observation_files"]):
        if len(files) == 0 or len(files) >= 3:
            aligned.append(files)
            continue
        expected = [(ts - pd.Timedelta(minutes=m)).strftime("%Y%m%d_%H%M") for m in (30, 20, 10)]
        stamp_of = {}
        for name in files:
            match = re.search(r"(\d{8}_\d{4})", name)
            if match is None:
                stamp_of = None
                break
            stamp_of[match.group(1)] = name
        if stamp_of is None or any(stamp not in expected for stamp in stamp_of):
            aligned.append(files)
            continue
        slots = [stamp_of.get(stamp) for stamp in expected]
        filled = []
        for i, slot in enumerate(slots):
            if slot is not None:
                filled.append(slot)
            else:
                _, j = min((abs(j - i), j) for j, other in enumerate(slots) if other is not None)
                filled.append(slots[j])
        aligned.append(filled)
    df["observation_files"] = aligned
    return df


def extend_temporal_context(dataframe: pd.DataFrame, extra_steps: int) -> pd.DataFrame:
    """Prepend observation files from earlier rows of the same location chain.

    Rows are 30 minutes apart per location, so the previous row's three frames
    directly precede this row's frames. Chain starts are padded by repeating
    the oldest available frame.
    """
    df = dataframe.copy()
    lookup = {
        (loc, ts): files
        for loc, ts, files in zip(df["name_location"], df["datetime"], df["observation_files"])
        if len(files) > 0
    }
    step = pd.Timedelta(minutes=30)
    extended = []
    for loc, ts, files in zip(df["name_location"], df["datetime"], df["observation_files"]):
        if len(files) == 0:
            extended.append(files)
            continue
        chunks = [list(files)]
        for offset in range(1, extra_steps + 1):
            previous = lookup.get((loc, ts - offset * step))
            if previous and len(previous) == len(files):
                chunks.insert(0, list(previous))
            else:
                chunks.insert(0, [chunks[0][0]] * len(files))
        extended.append([name for chunk in chunks for name in chunk])
    df["observation_files"] = extended
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


def load_location_metadata(path: str | Path) -> pd.DataFrame:
    """Load manually curated location coordinates.

    Required columns:
    - name_location
    - latitude
    - longitude

    Coordinates are intentionally supplied by a local CSV so the training code
    does not perform geocoding or fetch any external dataset.
    """
    metadata = pd.read_csv(path)
    required = {"name_location", "latitude", "longitude"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Location metadata is missing columns: {sorted(missing)}")
    metadata = metadata[["name_location", "latitude", "longitude"]].copy()
    metadata["name_location"] = metadata["name_location"].astype(str)
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")
    if metadata[["latitude", "longitude"]].isna().any().any():
        bad = metadata[
            metadata[["latitude", "longitude"]].isna().any(axis=1)
        ]["name_location"].tolist()
        raise ValueError(f"Invalid coordinates for locations: {bad[:10]}")
    if metadata["name_location"].duplicated().any():
        duplicated = metadata.loc[
            metadata["name_location"].duplicated(), "name_location"
        ].tolist()
        raise ValueError(f"Duplicated location rows: {duplicated[:10]}")
    if not metadata["latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be in [-90, 90]")
    if not metadata["longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be in [-180, 180]")
    return metadata


def attach_location_metadata(
    dataframe: pd.DataFrame, config: Config, required: bool = True
) -> pd.DataFrame:
    """Attach coordinates to a dataframe when location features are enabled."""
    if not config.use_location_features:
        return dataframe
    if not config.location_metadata_path:
        raise ValueError("location_metadata_path is required when use_location_features=True")
    metadata = load_location_metadata(config.location_metadata_path)
    merged = dataframe.merge(metadata, on="name_location", how="left", validate="many_to_one")
    if required and merged[["latitude", "longitude"]].isna().any().any():
        missing_locations = sorted(
            merged.loc[
                merged[["latitude", "longitude"]].isna().any(axis=1),
                "name_location",
            ].unique()
        )
        raise ValueError(
            "Location metadata does not cover all rows. "
            f"Missing examples: {missing_locations[:20]}"
        )
    return merged


def attach_sample_weights(dataframe: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach optional per-sample training weights by unique_id."""
    dataframe = dataframe.copy()
    if not config.sample_weight_path:
        dataframe["sample_weight"] = 1.0
        return dataframe
    weights = pd.read_csv(config.sample_weight_path)
    required = {"unique_id", config.sample_weight_column}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Sample weight file is missing columns: {sorted(missing)}")
    weights = weights[["unique_id", config.sample_weight_column]].copy()
    weights = weights.rename(columns={config.sample_weight_column: "sample_weight"})
    weights["sample_weight"] = pd.to_numeric(
        weights["sample_weight"], errors="coerce"
    )
    merged = dataframe.merge(weights, on="unique_id", how="left", validate="one_to_one")
    if merged["sample_weight"].isna().any():
        missing_count = int(merged["sample_weight"].isna().sum())
        raise ValueError(f"Sample weights missing for {missing_count} rows")
    merged["sample_weight"] = merged["sample_weight"].clip(lower=0.05, upper=20.0)
    return merged


def load_pseudo_labels(config: Config) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Load transductive pseudo-labels for evaluation samples.

    Returns a metadata frame (rows from evaluation_target.csv, flagged with
    is_eval/pseudo_index/sample_weight) and the prediction array in mm/h.
    """
    if not config.pseudo_label_npz or not config.pseudo_label_csv:
        return None
    if config.use_location_features:
        raise ValueError("Pseudo labels are not supported with use_location_features")
    predictions = np.load(config.pseudo_label_npz)["predictions"].astype(np.float32)
    index_frame = pd.read_csv(config.pseudo_label_csv)
    if len(index_frame) != len(predictions):
        raise ValueError(
            f"Pseudo index rows ({len(index_frame)}) != predictions ({len(predictions)})"
        )
    evaluation_frame = prepare_metadata(
        config.paths.evaluation_dir / "evaluation_target.csv"
    ).set_index("unique_id")
    pseudo_frame = evaluation_frame.loc[index_frame["unique_id"]].reset_index()
    pseudo_frame["is_eval"] = True
    pseudo_frame["pseudo_index"] = np.arange(len(pseudo_frame))
    pseudo_frame["sample_weight"] = float(config.pseudo_sample_weight)
    return pseudo_frame, predictions


def location_features(row: pd.Series) -> np.ndarray:
    """Continuous geospatial/time features derived from manual coordinates."""
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    timestamp = row["datetime"]

    month_phase = 2 * math.pi * (timestamp.month - 1) / 12
    utc_hour = timestamp.hour + timestamp.minute / 60
    # Approximate solar local time from longitude. This avoids timezone lookup.
    local_hour = (utc_hour + longitude / 15.0) % 24
    local_hour_phase = 2 * math.pi * local_hour / 24
    lon_phase = math.radians(longitude)
    hemisphere = 1.0 if latitude >= 0 else -1.0

    return np.asarray(
        [
            latitude / 90.0,
            longitude / 180.0,
            abs(latitude) / 90.0,
            hemisphere,
            math.sin(lon_phase),
            math.cos(lon_phase),
            math.sin(local_hour_phase),
            math.cos(local_hour_phase),
            hemisphere * math.sin(month_phase),
            hemisphere * math.cos(month_phase),
        ],
        dtype=np.float32,
    )


def context_features(row: pd.Series, config: Config) -> np.ndarray:
    features = [time_features(row["datetime"])]
    if config.use_location_features:
        geo_features = location_features(row)
        if config.location_feature_mode == "full":
            features.append(geo_features)
        elif config.location_feature_mode == "local_time":
            features.append(geo_features[6:8])
        else:
            raise ValueError(f"Unknown location_feature_mode: {config.location_feature_mode}")
    return np.concatenate(features).astype(np.float32)


def context_feature_count(config: Config) -> int:
    count = len(time_features(pd.Timestamp("2000-01-01")))
    if config.use_location_features:
        if config.location_feature_mode == "full":
            count += 10
        elif config.location_feature_mode == "local_time":
            count += 2
        else:
            raise ValueError(f"Unknown location_feature_mode: {config.location_feature_mode}")
    return count


def satellite_directories(config: Config, split: str) -> dict[str, Path]:
    base = config.paths.train_dir if split == "train" else config.paths.evaluation_dir
    return {satellite: base / satellite for satellite in SATELLITES}


def get_band_mapping(config: Config) -> dict[str, tuple[int, ...]]:
    if config.band_mode == "legacy3":
        return LEGACY_THREE_BANDS
    if config.band_mode in ("matched6", "matched6_btd"):
        return SATELLITE_BANDS
    if config.band_mode in ("full16", "full16_btd"):
        return FULL16_BANDS
    raise ValueError(f"Unknown band_mode: {config.band_mode}")


def compute_band_stats(
    dataframe: pd.DataFrame,
    directories: dict[str, Path],
    max_samples_per_satellite: int | None,
    seed: int,
    band_mapping: dict[str, tuple[int, ...]] | None = None,
    include_shared: bool = False,
    append_btd: bool = False,
    btd_pairs_by_satellite: dict[str, tuple[tuple[int, int], ...]] | None = None,
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
        btd_pairs = (btd_pairs_by_satellite or {}).get(satellite, BTD_PAIRS)
        channel_count = len(bands) + (len(btd_pairs) if append_btd else 0)
        sums = np.zeros(channel_count, dtype=np.float64)
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
                # Keep statistics consistent with _read_observation: an
                # all-zero frame is a fully-clouded/missing observation, not
                # a valid low-valued satellite sample.
                if not data.any():
                    continue
                if append_btd:
                    data = append_btd_channels(data, btd_pairs)
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
        pseudo_targets: np.ndarray | None = None,
        eval_directories: dict[str, Path] | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.directories = directories
        self.stats = stats
        self.config = config
        self.has_target = has_target
        self.augment = augment
        self.pseudo_targets = pseudo_targets
        self.eval_directories = eval_directories
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
                        input_channel_count(self.config, satellite),
                        self.config.encoder_size,
                        self.config.encoder_size,
                    ),
                    dtype=np.float32,
                )
            data = src.read(bands).astype(np.float32)
        if not data.any():
            # fully clouded frame (all channels zero): raw zeros would normalize
            # to -3..-4 sigma outliers, so treat like a missing frame instead
            return np.zeros(
                (
                    input_channel_count(self.config, satellite),
                    self.config.encoder_size,
                    self.config.encoder_size,
                ),
                dtype=np.float32,
            )
        if uses_btd(self.config):
            data = append_btd_channels(
                data, btd_pairs_for(self.config, satellite)
            )
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

        directories = self.directories
        if self.eval_directories is not None and bool(row.get("is_eval", False)):
            directories = self.eval_directories

        observations = []
        for observation_index in range(self.config.max_observations):
            if observation_index < len(files):
                path = directories[satellite] / files[observation_index]
                observations.append(self._read_observation(path, satellite))
            else:
                observations.append(
                    np.zeros(
                        (
                            input_channel_count(self.config, satellite),
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
        if self.config.use_flow_divergence:
            if len(files) == self.config.max_observations:
                image_features.append(
                    flow_divergence_channels(
                        observation_stack[:, IR_WINDOW_INDEX],
                        self.config.encoder_size,
                    )
                )
            else:
                image_features.append(
                    np.zeros(
                        (
                            self.config.max_observations - 1,
                            self.config.encoder_size,
                            self.config.encoder_size,
                        ),
                        dtype=np.float32,
                    )
                )
        if self.config.use_flow_extrapolation:
            valid_frames = np.any(observation_stack != 0, axis=(1, 2, 3))
            if len(files) == self.config.max_observations and valid_frames.all():
                image_features.append(flow_extrapolated_channels(observation_stack))
            else:
                image_features.append(
                    np.zeros(
                        (
                            input_channel_count(self.config, satellite),
                            self.config.encoder_size,
                            self.config.encoder_size,
                        ),
                        dtype=np.float32,
                    )
                )
        image = np.concatenate(image_features, axis=0)

        pseudo_index = int(row.get("pseudo_index", -1))
        if self.pseudo_targets is not None and pseudo_index >= 0:
            pseudo = self.pseudo_targets[pseudo_index].astype(np.float32)
            target = np.log1p(np.clip(pseudo, 0, None))[None]
        elif self.has_target:
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
            "sample_weight": torch.tensor(
                float(row.get("sample_weight", 1.0)), dtype=torch.float32
            ),
        }
        return (
            torch.from_numpy(image),
            torch.tensor(SATELLITE_TO_ID[satellite], dtype=torch.long),
            torch.from_numpy(context_features(row, self.config)),
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
        # Keep routing IDs on CPU for MPS.  MPS advanced indexing accepts CPU
        # LongTensors, while repeatedly running nonzero on MPS can lose IDs.
        routing_ids = satellite_id.cpu() if satellite_id.device.type == "mps" else satellite_id
        for identifier, stem in enumerate(self.stems):
            index = torch.nonzero(routing_ids == identifier, as_tuple=False).flatten()
            if len(index):
                outputs.append(stem(image[index]))
                indices.append(index)
        combined_indices = torch.cat(indices)
        combined_outputs = torch.cat(outputs, dim=0)
        return combined_outputs[torch.argsort(combined_indices)]


class FPNDecoder(nn.Module):
    def __init__(
        self,
        input_channels: Iterable[int],
        output_channels: int,
        features_nhwc: bool = True,
    ) -> None:
        super().__init__()
        self.features_nhwc = features_nhwc
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
        if self.features_nhwc:
            # timm Swin emits NHWC tensors; ConvNeXt-style encoders emit NCHW.
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
        band_count = input_channel_count(config, next(iter(band_mapping)))
        raw_channels = band_count * config.max_observations
        if config.use_temporal_differences:
            raw_channels += band_count * (config.max_observations - 1)
        if config.use_temporal_summary:
            raw_channels += band_count * 2
        if config.use_flow_divergence:
            raw_channels += config.max_observations - 1
        if config.use_flow_extrapolation:
            raw_channels += band_count
        self.config = config
        self.band_count = band_count
        self.frame_channel_count = band_count * config.max_observations
        self.auxiliary_channel_count = raw_channels - self.frame_channel_count
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
        if config.use_temporal_frame_attention:
            self.frame_stem = SatelliteStem(band_count, config.stem_channels)
            self.shared_frame_stem = nn.Sequential(
                nn.Conv2d(band_count, config.stem_channels, 3, padding=1, bias=False),
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
            if self.auxiliary_channel_count:
                self.auxiliary_stem = SatelliteStem(
                    self.auxiliary_channel_count, config.stem_channels
                )
                self.shared_auxiliary_stem = nn.Sequential(
                    nn.Conv2d(
                        self.auxiliary_channel_count,
                        config.stem_channels,
                        3,
                        padding=1,
                        bias=False,
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
            else:
                self.auxiliary_stem = None
                self.shared_auxiliary_stem = None
            self.temporal_attention = nn.Sequential(
                nn.Conv2d(config.stem_channels, config.stem_channels // 2, 1),
                nn.GELU(),
                nn.Conv2d(config.stem_channels // 2, 1, 1),
            )
            self.temporal_position_bias = nn.Parameter(
                torch.zeros(config.max_observations)
            )
        self.encoder = timm.create_model(
            config.encoder_name,
            pretrained=config.pretrained,
            features_only=True,
            in_chans=config.stem_channels,
        )
        feature_channels = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(
            feature_channels,
            config.decoder_channels,
            features_nhwc="swin" in config.encoder_name,
        )
        condition_size = config.decoder_channels
        self.satellite_embedding = nn.Embedding(len(SATELLITES), condition_size)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_feature_count(config) + 1, condition_size),
            nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        output_channels = 4 if config.use_two_head else 1
        self.head = nn.Sequential(
            nn.Conv2d(config.decoder_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_channels, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        satellite_id: torch.Tensor,
        temporal_features: torch.Tensor,
        missing_flag: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.config.use_temporal_frame_attention:
            stacked_input = image
            batch_size, _, height, width = image.shape
            frames = stacked_input[:, : self.frame_channel_count].reshape(
                batch_size,
                self.config.max_observations,
                self.band_count,
                height,
                width,
            )
            flattened_frames = frames.reshape(
                batch_size * self.config.max_observations,
                self.band_count,
                height,
                width,
            )
            repeated_satellite_id = satellite_id.repeat_interleave(
                self.config.max_observations
            )
            if self.config.use_satellite_stem:
                encoded_frames = self.frame_stem(flattened_frames, repeated_satellite_id)
            else:
                encoded_frames = self.shared_frame_stem(flattened_frames)
            encoded_frames = encoded_frames.reshape(
                batch_size,
                self.config.max_observations,
                self.config.stem_channels,
                height,
                width,
            )
            attention_logits = self.temporal_attention(
                encoded_frames.reshape(
                    batch_size * self.config.max_observations,
                    self.config.stem_channels,
                    height,
                    width,
                )
            ).reshape(batch_size, self.config.max_observations, 1, height, width)
            attention_logits = attention_logits + self.temporal_position_bias[
                None, :, None, None, None
            ]
            valid_frames = frames.abs().sum(dim=(2, 3, 4), keepdim=True) > 0
            attention_logits = attention_logits.masked_fill(~valid_frames, -1e4)
            attention = torch.softmax(attention_logits, dim=1)
            image = (encoded_frames * attention).sum(dim=1)
            if self.auxiliary_channel_count:
                auxiliary = stacked_input[:, self.frame_channel_count :]
                if self.config.use_satellite_stem:
                    auxiliary = self.auxiliary_stem(auxiliary, satellite_id)
                else:
                    auxiliary = self.shared_auxiliary_stem(auxiliary)
                image = image + auxiliary
            image = image * valid_frames.any(dim=1).to(image.dtype)
        elif self.config.use_satellite_stem:
            image = self.stem(image, satellite_id)
        else:
            image = self.shared_stem(image)
        features = self.encoder(image)
        decoded = self.decoder(features)
        context = temporal_features.clone()
        if not self.config.use_month_features:
            context[:, :2] = 0
        if not self.config.use_hour_features:
            context[:, 2:4] = 0
        if self.config.use_location_features:
            if self.config.location_feature_mode == "full":
                if self.config.disable_geo_position_features:
                    context[:, 4:10] = 0
                if self.config.disable_local_time_features:
                    context[:, 10:12] = 0
                if self.config.disable_geo_season_features:
                    context[:, 12:14] = 0
            elif self.config.location_feature_mode == "local_time":
                if self.config.disable_local_time_features:
                    context[:, 4:6] = 0
        if not self.config.use_missing_flag:
            missing_flag = torch.zeros_like(missing_flag)
        condition = self.context_mlp(torch.cat([context, missing_flag], dim=1))
        if self.config.use_satellite_embedding:
            embedding_id = satellite_id.to(condition.device, non_blocking=True)
            condition = condition + self.satellite_embedding(embedding_id)
        decoded = decoded + condition[:, :, None, None]
        output = self.head(decoded)
        output = F.interpolate(
            output,
            size=(self.config.target_size, self.config.target_size),
            mode="bilinear",
            align_corners=False,
        )
        if not self.config.use_two_head:
            return output

        amount_log = F.softplus(output[:, :1])
        rain_logits = output[:, 1:]
        rain_probability = torch.sigmoid(rain_logits[:, :1])
        prediction = amount_log * rain_probability
        if return_aux:
            return prediction, rain_logits
        return prediction


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
        num_workers=config.workers if device.type in ("cuda", "xla") else 0,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0 and device.type in ("cuda", "xla"),
    )


def original_scale_rmse(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[float, int]:
    predictions = torch.expm1(predictions).clamp(min=0)
    targets = torch.expm1(targets).clamp(min=0)
    squared_error = F.mse_loss(predictions, targets, reduction="sum").item()
    return squared_error, targets.numel()


def make_training_loss(config: Config) -> nn.Module:
    if config.loss_type == "huber":
        return nn.HuberLoss(delta=config.huber_delta)
    if config.loss_type == "log_mse":
        return nn.MSELoss()
    if config.loss_type == "weighted_huber":
        return nn.HuberLoss(delta=config.huber_delta, reduction="none")
    raise ValueError(f"Unknown loss_type: {config.loss_type}")


def compute_training_loss(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    rain_logits: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_weight is not None:
        sample_weight = sample_weight.to(
            device=predictions.device, dtype=predictions.dtype
        ).view(-1)
        sample_weight = torch.clamp(sample_weight, min=0.05, max=20.0)
        weight_denominator = sample_weight.sum().clamp_min(1e-6)
    else:
        weight_denominator = None

    if config.loss_type != "weighted_huber":
        if sample_weight is None:
            base_loss = criterion(predictions, targets)
        else:
            if config.loss_type == "huber":
                element_loss = F.huber_loss(
                    predictions,
                    targets,
                    delta=config.huber_delta,
                    reduction="none",
                )
            elif config.loss_type == "log_mse":
                element_loss = F.mse_loss(predictions, targets, reduction="none")
            else:
                raise ValueError(f"Unknown loss_type: {config.loss_type}")
            sample_loss = element_loss.flatten(1).mean(dim=1)
            base_loss = (sample_loss * sample_weight).sum() / weight_denominator
    else:
        element_loss = criterion(predictions, targets)
        target_original = torch.expm1(targets).clamp(min=0)
        weight = 1.0 + config.heavy_rain_weight_alpha * torch.clamp(
            target_original / config.heavy_rain_weight_scale,
            min=0,
            max=config.heavy_rain_weight_max,
        )
        weighted_element_loss = element_loss * weight
        if sample_weight is None:
            base_loss = weighted_element_loss.mean()
        else:
            sample_loss = weighted_element_loss.flatten(1).mean(dim=1)
            base_loss = (sample_loss * sample_weight).sum() / weight_denominator

    if config.raw_huber_loss_weight > 0:
        prediction_original = torch.expm1(predictions).clamp(
            min=0, max=config.raw_huber_max
        )
        target_original = torch.expm1(targets).clamp(
            min=0, max=config.raw_huber_max
        )
        raw_element_loss = F.smooth_l1_loss(
            prediction_original,
            target_original,
            beta=config.raw_huber_beta,
            reduction="none",
        )
        if sample_weight is None:
            raw_loss = raw_element_loss.mean()
        else:
            raw_sample_loss = raw_element_loss.flatten(1).mean(dim=1)
            raw_loss = (raw_sample_loss * sample_weight).sum() / weight_denominator
        base_loss = base_loss + config.raw_huber_loss_weight * raw_loss

    if not config.use_two_head:
        return base_loss
    if rain_logits is None:
        raise ValueError("rain_logits are required when use_two_head=True")

    target_original = torch.expm1(targets).clamp(min=0)
    rain_targets = torch.cat(
        [
            (target_original >= 0.1).float(),
            (target_original >= 1.0).float(),
            (target_original >= 5.0).float(),
        ],
        dim=1,
    )
    bce = F.binary_cross_entropy_with_logits(rain_logits, rain_targets, reduction="none")
    bce_weights = torch.tensor(
        [
            config.rain_bce_weight_0_1,
            config.rain_bce_weight_1,
            config.rain_bce_weight_5,
        ],
        device=rain_logits.device,
        dtype=rain_logits.dtype,
    )[None, :, None, None]
    bce_loss = (bce * bce_weights).flatten(1).mean(dim=1)
    if sample_weight is None:
        return base_loss + bce_loss.mean()
    return base_loss + (bce_loss * sample_weight).sum() / weight_denominator


def train_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: dict,
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    seed_everything(config.seed + fold["fold"])
    config.model_dir.mkdir(parents=True, exist_ok=True)

    dataframe = attach_location_metadata(dataframe, config)
    dataframe = attach_sample_weights(dataframe, config)
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
            band_mapping=get_band_mapping(config),
            append_btd=uses_btd(config),
            btd_pairs_by_satellite={
                satellite: btd_pairs_for(config, satellite) for satellite in SATELLITES
            },
        )
        save_stats(stats, stats_path)

    pseudo_targets = None
    eval_directories = None
    pseudo = load_pseudo_labels(config)
    if pseudo is not None:
        pseudo_frame, pseudo_targets = pseudo
        train_frame = pd.concat([train_frame, pseudo_frame], ignore_index=True)
        train_frame["pseudo_index"] = (
            train_frame.get("pseudo_index", pd.Series(-1, index=train_frame.index))
            .fillna(-1)
            .astype(int)
        )
        train_frame["is_eval"] = (
            train_frame.get("is_eval", pd.Series(False, index=train_frame.index))
            .fillna(False)
            .astype(bool)
        )
        eval_directories = satellite_directories(config, "evaluation")
        print(
            f"Fold {fold['fold']}: appended {len(pseudo_frame)} pseudo-labeled eval samples "
            f"(weight={config.pseudo_sample_weight})"
        )

    train_dataset = NowcastingDataset(
        train_frame,
        train_directories,
        stats,
        config,
        has_target=True,
        augment=True,
        pseudo_targets=pseudo_targets,
        eval_directories=eval_directories,
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
    criterion = make_training_loss(config)
    amp_enabled = config.use_amp and device.type in ("cuda", "xla")
    # bf16 on TPU needs no gradient scaling; fp16 on CUDA does
    amp_dtype = torch.bfloat16 if device.type == "xla" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    if device.type == "xla":
        # without a per-step barrier the XLA lazy graph grows across steps
        # (recompiles every step, then host OOM)
        import torch_xla.core.xla_model as xm

        mark_step = xm.mark_step
    else:
        mark_step = lambda: None

    checkpoint_path = config.model_dir / f"best_fold{fold['fold']}.pth"
    history = []
    best_rmse = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for image, satellite_id, temporal, missing, target, metadata in tqdm(
            train_loader, desc=f"Fold {fold['fold']} epoch {epoch}", leave=False
        ):
            image = image.to(device, non_blocking=True)
            if device.type != "mps":
                satellite_id = satellite_id.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sample_weight = metadata.get("sample_weight")
            if sample_weight is not None:
                sample_weight = sample_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if config.use_two_head:
                    prediction, rain_logits = model(
                        image,
                        satellite_id,
                        temporal,
                        missing,
                        return_aux=True,
                    )
                else:
                    prediction = model(image, satellite_id, temporal, missing)
                    rain_logits = None
                loss = compute_training_loss(
                    criterion,
                    prediction,
                    target,
                    config,
                    rain_logits,
                    sample_weight=sample_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            mark_step()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite training loss at fold={fold['fold']} epoch={epoch}"
                )
            loss_sum += loss_value * image.shape[0]
            sample_count += image.shape[0]

        model.eval()
        squared_error = 0.0
        pixel_count = 0
        with torch.no_grad():
            for image, satellite_id, temporal, missing, target, _ in validation_loader:
                image = image.to(device, non_blocking=True)
                if device.type != "mps":
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
        if not math.isfinite(validation_rmse):
            raise FloatingPointError(
                f"Non-finite validation RMSE at fold={fold['fold']} epoch={epoch}"
            )
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
            epochs_without_improvement = 0
            torch.save(
                {
                    "config": asdict(config),
                    "fold": fold["fold"],
                    "validation_locations": fold["validation_locations"],
                    "validation_rmse": validation_rmse,
                    # move to CPU so checkpoints load anywhere (XLA tensors
                    # cannot be deserialized without torch_xla installed)
                    "model_state_dict": {
                        key: value.cpu() for key, value in model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience is not None
            and epoch >= config.early_stopping_min_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"early-stop: fold={fold['fold']} epoch={epoch:02d} "
                f"best_rmse={best_rmse:.5f} "
                f"patience={config.early_stopping_patience}"
            )
            break

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


def predict_fold(
    config: Config,
    dataframe: pd.DataFrame,
    fold: int,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    model, stats = load_fold_model(config, fold, device)
    dataframe = attach_location_metadata(dataframe, config)
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
            image = image.to(device, non_blocking=True)
            if device.type != "mps":
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
                prediction = (accumulated / len(dihedral_transforms())).cpu().numpy()
            else:
                prediction = model(image, satellite_id, temporal, missing)
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
