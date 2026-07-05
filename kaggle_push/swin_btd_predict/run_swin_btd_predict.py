from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def install_p100_torch_if_needed() -> str:
    try:
        gpu = subprocess.check_output(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
            ],
            text=True,
        ).strip()
    except Exception:
        gpu = ""
    if "P100" in gpu:
        print(f"Installing a Pascal-compatible PyTorch build for {gpu}")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "pillow==11.3.0",
            ],
            check=True,
        )
    return gpu


def install_runtime_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "timm", "rasterio"],
        check=True,
    )


def clone_repo() -> Path:
    repo = Path("/kaggle/working/solafune-nowcast")
    if repo.exists():
        subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/shionsuio/solafune-nowcast.git",
                str(repo),
            ],
            check=True,
        )
    sys.path.insert(0, str(repo / "src"))
    return repo


def copy_model_inputs() -> tuple[Path, Path]:
    checkpoints = sorted(Path("/kaggle/input").rglob("best_fold*.pth"))
    stats = sorted(Path("/kaggle/input").rglob("band_stats_fold*.json"))
    btd_checkpoints = [p for p in checkpoints if "btd" in str(p).lower().replace("-", "_")]
    btd_stats = [p for p in stats if "btd" in str(p).lower().replace("-", "_")]
    print("btd checkpoints:", len(btd_checkpoints), "btd stats:", len(btd_stats))
    assert len(btd_checkpoints) == 5 and len(btd_stats) == 5

    model_dir = Path("/kaggle/working/models/swin_v2_temporal_two_head_btd")
    stats_dir = Path("/kaggle/working/outputs/band_stats_btd")
    model_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    for path in btd_checkpoints:
        shutil.copy2(path, model_dir / path.name)
    for path in btd_stats:
        shutil.copy2(path, stats_dir / path.name)
    return model_dir, stats_dir


def main() -> None:
    gpu_name = install_p100_torch_if_needed()
    install_runtime_deps()
    clone_repo()

    import numpy as np
    import torch

    from kaggle_setup import ensure_kaggle_workspace
    from swin_nowcast_v2 import Config, create_submission, predict_fold, prepare_metadata

    data_root = Path("/kaggle/input/solafune-dataset-v2")
    if not data_root.exists():
        data_root = Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")
    ensure_kaggle_workspace(Path("/kaggle/working"), data_root)

    print("GPU:", gpu_name or torch.cuda.get_device_name(0), "torch:", torch.__version__)
    assert torch.cuda.is_available(), "GPU is required for this kernel"

    model_dir, stats_dir = copy_model_inputs()
    config = Config(
        root="/kaggle/working",
        batch_size=8,
        workers=2,
        pretrained=False,
        use_amp=False,
        use_two_head=True,
        use_temporal_differences=True,
        use_temporal_summary=True,
        band_mode="matched6_btd",
        swin_model_subdir=model_dir.name,
        band_stats_root=str(stats_dir),
    )

    evaluation_frame = prepare_metadata(
        config.paths.evaluation_dir / "evaluation_target.csv"
    )
    device = torch.device("cuda")
    prediction_sum = None
    for fold in range(5):
        fold_predictions = predict_fold(config, evaluation_frame, fold, device)
        np.save(
            f"/kaggle/working/prediction_btd_fold{fold}.npy",
            fold_predictions.astype(np.float16),
        )
        prediction_sum = (
            fold_predictions
            if prediction_sum is None
            else prediction_sum + fold_predictions
        )
        print(
            f"fold={fold} pred mean={fold_predictions.mean():.6f} "
            f"p95={np.quantile(fold_predictions, 0.95):.6f} max={fold_predictions.max():.6f}"
        )

    predictions = prediction_sum / 5
    np.save("/kaggle/working/prediction_btd_ensemble.npy", predictions.astype(np.float16))
    print(
        f"ensemble mean={predictions.mean():.6f} "
        f"p95={np.quantile(predictions, 0.95):.6f} max={predictions.max():.6f}"
    )

    output_zip = Path("/kaggle/working/submission_swin_btd_5fold.zip")
    create_submission(config, evaluation_frame, predictions, output_zip)
    print("Saved:", output_zip)


if __name__ == "__main__":
    main()
