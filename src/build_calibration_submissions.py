"""Build calibration submission zips from existing submission zips (no GPU).

The temporal submit component t is recovered algebraically from two blends of
the lost base swin model s:
    A = swin90_temporal10 = 0.9 s + 0.1 t
    B = swin80_temporal20 = 0.8 s + 0.2 t
    -> t = 9B - 8A   (verified: 2A - B reproduces submission.zip to 3e-6)

Outputs exactly reproducible on OOF by build_cv_lb_correlation.py recipes:
    temporal100, temporal80_twohead20, temporal50_twohead50, temporal_hg20
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from tqdm.auto import tqdm


def load_zip(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict], bytes]:
    arrays: dict[str, np.ndarray] = {}
    profiles: dict[str, dict] = {}
    csv_bytes = b""
    with zipfile.ZipFile(path) as archive:
        for name in tqdm(archive.namelist(), desc=f"load {path.name}", leave=False):
            data = archive.read(name)
            if name.endswith(".csv"):
                csv_bytes = data
                continue
            with MemoryFile(data) as memory, memory.open() as source:
                arrays[name] = source.read(1).astype(np.float32)
                profiles[name] = source.profile
    return arrays, profiles, csv_bytes


def write_zip(
    path: Path,
    arrays: dict[str, np.ndarray],
    profiles: dict[str, dict],
    csv_name: str,
    csv_bytes: bytes,
) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(csv_name, csv_bytes)
        for name in tqdm(sorted(arrays), desc=f"write {path.name}", leave=False):
            profile = profiles[name]
            with MemoryFile() as memory:
                with memory.open(**profile) as destination:
                    destination.write(arrays[name], 1)
                archive.writestr(name, memory.read())
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    blend_dir = root / "outputs" / "submission_blends"
    output_dir = root / "outputs" / "submission_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    a_arrays, profiles, csv_bytes = load_zip(blend_dir / "submission_swin90_temporal10.zip")
    b_arrays, _, _ = load_zip(blend_dir / "submission_swin80_temporal20.zip")
    th_arrays, _, _ = load_zip(Path(args.twohead_zip))

    names = sorted(a_arrays)
    assert names == sorted(b_arrays) == sorted(th_arrays), "zip file lists differ"

    evaluation = pd.read_csv(root / "evaluation_dataset" / "evaluation_target.csv")
    satellite_by_file = {
        f"test_files/{row.gpm_imerg_filename}": row.satellite_target.lower()
        for row in evaluation.itertuples()
    }
    missing = [name for name in names if name not in satellite_by_file]
    assert not missing, f"files without satellite mapping: {missing[:3]}"

    temporal = {
        name: np.clip(9.0 * b_arrays[name] - 8.0 * a_arrays[name], 0.0, None)
        for name in names
    }

    recipes: dict[str, dict[str, np.ndarray]] = {
        "submission_temporal100": temporal,
        "submission_temporal80_twohead20": {
            name: 0.8 * temporal[name] + 0.2 * th_arrays[name] for name in names
        },
        "submission_temporal50_twohead50": {
            name: 0.5 * temporal[name] + 0.5 * th_arrays[name] for name in names
        },
        "submission_temporal_hg20": {
            name: (
                0.8 * temporal[name] + 0.2 * th_arrays[name]
                if satellite_by_file[name] in ("himawari", "goes")
                else temporal[name]
            )
            for name in names
        },
    }

    for recipe_name, arrays in recipes.items():
        write_zip(
            output_dir / f"{recipe_name}.zip",
            arrays,
            profiles,
            "evaluation_target.csv",
            csv_bytes,
        )

    # consistency check: 0.8 s + 0.2 t must reproduce B (s = 2A - B)
    check = np.random.default_rng(0).choice(names, 50, replace=False)
    max_diff = max(
        float(
            np.abs(
                0.8 * (2.0 * a_arrays[n] - b_arrays[n]) + 0.2 * temporal[n] - b_arrays[n]
            ).max()
        )
        for n in check
    )
    print(f"consistency max diff vs swin80_temporal20: {max_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--twohead-zip",
        default="/Users/shionsuio/Downloads/submission_swin_twohead_fold02_stable134.zip",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
