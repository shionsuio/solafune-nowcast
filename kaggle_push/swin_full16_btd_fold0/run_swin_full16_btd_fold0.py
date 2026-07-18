from pathlib import Path
import subprocess
import sys


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
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--force-reinstall",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "pillow==11.3.0"],
            check=True,
        )
    return gpu


gpu_name = install_p100_torch_if_needed()

repo = Path("/kaggle/working/solafune-nowcast")
if repo.exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
else:
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "codex/flow-warp-kaggle",
            "https://github.com/shionsuio/solafune-nowcast.git",
            str(repo),
        ],
        check=True,
    )

sys.path.insert(0, str(repo / "src"))

import torch

print("GPU:", gpu_name or torch.cuda.get_device_name(0), "torch:", torch.__version__)
assert torch.cuda.is_available(), "GPU is required"

from kaggle_setup import ensure_kaggle_workspace
from run_swin_temporal_full import run

data_root = Path("/kaggle/input/solafune-dataset-v2")
if not data_root.exists():
    data_root = Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")

ensure_kaggle_workspace(Path("/kaggle/working"), data_root)


class Args:
    root = "/kaggle/working"
    kaggle_input_root = None
    folds = "0"
    folds_json = repo / "outputs/cv_schemes/eval_aligned/folds.json"
    epochs = 10
    early_stopping_patience = 3
    early_stopping_min_epochs = 7
    batch_size = 8
    workers = 2
    lr_encoder = 2e-5
    lr_head = 1e-4
    loss_type = "huber"
    heavy_rain_weight_alpha = 0.5
    heavy_rain_weight_scale = 10.0
    heavy_rain_weight_max = 2.0
    raw_huber_loss_weight = 0.0
    raw_huber_beta = 5.0
    raw_huber_max = 100.0
    use_two_head = True
    rain_bce_weight_0_1 = 0.10
    rain_bce_weight_1 = 0.10
    rain_bce_weight_5 = 0.05
    stats_samples_per_satellite = 1500
    seed = 42
    model_subdir = "swin_btd_twohead_eval_aligned_full16_btd_f0"
    band_stats_root = None
    band_mode = "full16_btd"
    use_location_features = False
    location_metadata_path = None
    location_feature_mode = "full"
    no_pretrained = False
    no_amp = True
    disable_temporal_features = False
    align_frames = True
    exclude_bad_labels = True


summary_path = run(Args())
print(summary_path)
