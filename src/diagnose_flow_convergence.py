"""Visual diagnostic: optical-flow moisture-flux-convergence proxy vs rain target.

Computes, per sample:
- IR-window brightness (matched6 idx4) for 3 frames
- Farneback optical flow (frame0->2 averaged), its divergence
- clear-sky split-window BTD (idx4-idx5) as moisture proxy q
- convergence * q map (moisture flux convergence proxy)
Then correlates each map (downsampled to 41x41) with the GPM target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from swin_nowcast_v2 import (
    SATELLITE_BANDS,
    Config,
    prepare_metadata,
    satellite_directories,
)

IR_IDX = 4  # matched6 position: IR window (~10.4um)
DIRTY_IDX = 5  # matched6 position: dirty window (~12um)


def read_bands(path: Path, satellite: str) -> np.ndarray | None:
    bands = SATELLITE_BANDS[satellite]
    with rasterio.open(path) as src:
        if src.count < max(bands):
            return None
        return src.read(bands).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def compute_maps(frames: list[np.ndarray], grid: int = 128):
    ir_frames = [
        cv2.resize(f[IR_IDX], (grid, grid), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    dirty_last = cv2.resize(
        frames[-1][DIRTY_IDX], (grid, grid), interpolation=cv2.INTER_AREA
    )
    flows = []
    for a, b in zip(ir_frames[:-1], ir_frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            to_uint8(a), to_uint8(b), None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )
        flows.append(flow)
    flow = np.mean(flows, axis=0)
    du_dx = np.gradient(flow[..., 0], axis=1)
    dv_dy = np.gradient(flow[..., 1], axis=0)
    divergence = cv2.GaussianBlur(du_dx + dv_dy, (0, 0), sigmaX=3)

    ir_last = ir_frames[-1]
    # moisture proxy: split-window BTD, valid only where sky is warm/clearish
    btd = ir_last - dirty_last
    warm = ir_last > np.percentile(ir_last, 60)
    q_proxy = np.where(warm, btd, np.nan)
    q_filled = np.nan_to_num(q_proxy, nan=float(np.nanmedian(q_proxy)))
    q_smooth = cv2.GaussianBlur(q_filled, (0, 0), sigmaX=8)

    convergence = -divergence
    mfc = q_smooth * convergence
    return ir_last, flow, convergence, q_smooth, mfc


def downsample41(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (41, 41), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--per-satellite", type=int, default=3)
    parser.add_argument("--scan", type=int, default=400, help="rows scanned per satellite for heavy rain")
    parser.add_argument("--out", default="outputs/flow_diagnostics")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = Config(root=str(root))
    frame = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    frame = frame[~frame["missing_observation"]]
    frame = frame[frame["observation_files"].map(len) == 3]
    directories = satellite_directories(config, "train")
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    correlations = []
    for satellite in SATELLITE_BANDS:
        rows = frame[frame["satellite_target"] == satellite]
        rows = rows.sample(min(args.scan, len(rows)), random_state=42)
        means = []
        for _, row in rows.iterrows():
            with rasterio.open(config.paths.gpm_dir / row["gpm_imerg_filename"]) as src:
                target = src.read(1).astype(np.float32)
            target = np.clip(np.nan_to_num(target), 0, None)
            means.append(target.mean())
        rows = rows.assign(rain_mean=means).nlargest(args.per_satellite, "rain_mean")

        for _, row in rows.iterrows():
            frames = []
            ok = True
            for name in row["observation_files"]:
                data = read_bands(directories[satellite] / name, satellite)
                if data is None:
                    ok = False
                    break
                frames.append(data)
            if not ok:
                continue
            with rasterio.open(config.paths.gpm_dir / row["gpm_imerg_filename"]) as src:
                target = np.clip(np.nan_to_num(src.read(1).astype(np.float32)), 0, None)

            ir, flow, convergence, q_proxy, mfc = compute_maps(frames)
            maps41 = {
                "convergence": downsample41(convergence),
                "q_proxy": downsample41(q_proxy),
                "mfc": downsample41(mfc),
                "cold_ir": downsample41(-ir),
            }
            target_log = np.log1p(target)
            sample_corr = {
                key: float(np.corrcoef(value.ravel(), target_log.ravel())[0, 1])
                for key, value in maps41.items()
            }
            correlations.append(
                {"satellite": satellite, "uid": row["unique_id"], **sample_corr}
            )

            fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
            axes[0].imshow(ir, cmap="gray_r")
            step = 8
            ys, xs = np.mgrid[0 : ir.shape[0] : step, 0 : ir.shape[1] : step]
            axes[0].quiver(
                xs, ys, flow[::step, ::step, 0], flow[::step, ::step, 1],
                color="red", scale=40,
            )
            axes[0].set_title(f"IR + flow ({row['satellite_target']})")
            for ax, (key, cmap) in zip(
                axes[1:],
                [("convergence", "RdBu_r"), ("q_proxy", "viridis"), ("mfc", "RdBu_r")],
            ):
                data = maps41[key]
                limit = np.percentile(np.abs(data), 99) or 1
                kwargs = (
                    {"vmin": -limit, "vmax": limit} if cmap == "RdBu_r" else {}
                )
                ax.imshow(data, cmap=cmap, **kwargs)
                ax.set_title(f"{key} (r={sample_corr[key]:.2f})")
            axes[4].imshow(target_log, cmap="Blues")
            axes[4].set_title(f"target log1p (mean {row['rain_mean']:.2f})")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"flow_{satellite}_{row['unique_id']}.png", dpi=110)
            plt.close(fig)

    import pandas as pd

    result = pd.DataFrame(correlations)
    result.to_csv(out_dir / "correlations.csv", index=False)
    print(result.to_string(index=False))
    print("\nmean correlation vs log1p target:")
    print(result[["convergence", "q_proxy", "mfc", "cold_ir"]].mean().to_string())


if __name__ == "__main__":
    main()
