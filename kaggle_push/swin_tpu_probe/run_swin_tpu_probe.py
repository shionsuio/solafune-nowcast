"""TPU feasibility probe: run the real train_fold code path on a subsample.

Measures XLA compile overhead + steady-state step rate so we can decide how
to spend the 20h TPU quota (full clean-data retrain vs waiting for GPU).
"""

from pathlib import Path
import os
import subprocess
import sys
import time


def install_torch_xla() -> None:
    try:
        import torch_xla

        print("torch_xla preinstalled:", torch_xla.__version__)
        return
    except ImportError:
        pass
    import torch

    version = torch.__version__.split("+")[0]
    print(f"installing torch_xla matching torch {version}")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            f"torch_xla[tpu]=={version}",
            "-f", "https://storage.googleapis.com/libtpu-releases/index.html",
            "-f", "https://storage.googleapis.com/libtpu-wheels/index.html",
        ]
    )
    if result.returncode != 0:
        print("exact match failed, falling back to pinned torch 2.5.1 + torch_xla 2.5.1")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "torch==2.5.1", "torch_xla[tpu]==2.5.1",
                "-f", "https://storage.googleapis.com/libtpu-releases/index.html",
                "-f", "https://storage.googleapis.com/libtpu-wheels/index.html",
            ],
            check=True,
        )


def main() -> None:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    install_torch_xla()

    repo = Path("/kaggle/working/solafune-nowcast")
    if repo.exists():
        subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "https://github.com/shionsuio/solafune-nowcast.git", str(repo)],
            check=True,
        )
    sys.path.insert(0, str(repo / "src"))

    os.environ["SOLAFUNE_DEVICE"] = "xla"

    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm

    from kaggle_setup import ensure_kaggle_workspace
    from run_swin_temporal_full import BAD_LABEL_LOCATIONS
    from swin_nowcast_v2 import (
        Config,
        align_observation_frames,
        get_device,
        make_folds,
        prepare_metadata,
        train_fold,
    )

    data_root = Path("/kaggle/input/solafune-dataset-v2")
    if not data_root.exists():
        data_root = Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")
    ensure_kaggle_workspace(Path("/kaggle/working"), data_root)

    stats_dataset = Path("/kaggle/input/solafune-stat")
    if not stats_dataset.exists():
        stats_dataset = Path("/kaggle/input/datasets/suioshion/solafune-stat")

    device = get_device()
    print("device:", device)

    config = Config(
        root="/kaggle/working",
        epochs=1,
        batch_size=8,
        workers=4,
        lr_encoder=2e-5,
        lr_head=1e-4,
        loss_type="huber",
        use_two_head=True,
        use_temporal_differences=True,
        use_temporal_summary=True,
        band_mode="matched6",
        use_amp=True,  # bf16 autocast on XLA
        swin_model_subdir="swin_tpu_probe",
        band_stats_root=str(stats_dataset) if stats_dataset.exists() else None,
    )

    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    dataframe = align_observation_frames(dataframe)
    folds = make_folds(dataframe, config.n_folds)
    fold = folds[0]

    import pandas as pd

    bad_mask = (
        (dataframe["datetime"] == pd.Timestamp("2023-01-01 00:00:00"))
        & dataframe["name_location"].isin(BAD_LABEL_LOCATIONS)
    ).to_numpy()
    bad_positions = np.flatnonzero(bad_mask)
    fold["train_indices"] = np.setdiff1d(fold["train_indices"], bad_positions)
    fold["validation_indices"] = np.setdiff1d(fold["validation_indices"], bad_positions)

    # subsample: enough steps to reach steady state past XLA compilation
    rng = np.random.default_rng(0)
    probe_fold = {
        "fold": 0,
        "validation_locations": fold["validation_locations"],
        "train_indices": rng.choice(fold["train_indices"], 2000, replace=False),
        "validation_indices": rng.choice(fold["validation_indices"], 400, replace=False),
    }
    print(
        f"probe: {len(probe_fold['train_indices'])} train / "
        f"{len(probe_fold['validation_indices'])} val samples, bs={config.batch_size}"
    )

    start = time.time()
    result = train_fold(config, dataframe, probe_fold, device=device)
    elapsed = time.time() - start
    steps = len(probe_fold["train_indices"]) / config.batch_size
    print(f"epoch wall time: {elapsed/60:.1f} min ({steps:.0f} steps)")
    print(f"probe val_rmse: {result['validation_rmse']:.5f}")

    full_steps = len(fold["train_indices"]) / config.batch_size
    # first ~30 steps are compilation; assume the rest is steady state
    steady = elapsed / steps
    print(
        f"rough projection: {steady:.2f}s/step -> "
        f"{full_steps * steady / 3600:.2f}h per full fold0 epoch (upper bound, includes compile)"
    )
    print(xm.get_memory_info(device) if hasattr(xm, "get_memory_info") else "no mem info")


if __name__ == "__main__":
    main()
