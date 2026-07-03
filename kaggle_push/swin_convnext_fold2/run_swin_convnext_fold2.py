from pathlib import Path
import subprocess
import sys


def install_p100_torch_if_needed():
    try:
        gpu = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"],
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
    subprocess.run(["git", "clone", "https://github.com/shionsuio/solafune-nowcast.git", str(repo)], check=True)

sys.path.insert(0, str(repo / "src"))

import torch

print("GPU:", gpu_name or torch.cuda.get_device_name(0), "torch:", torch.__version__)
assert torch.cuda.is_available(), "GPU is required"

from run_swin_temporal_full import run

data_root = Path("/kaggle/input/solafune-dataset-v2")
if not data_root.exists():
    data_root = Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")

stats_dataset = Path("/kaggle/input/solafune-stat")
if not stats_dataset.exists():
    stats_dataset = Path("/kaggle/input/datasets/suioshion/solafune-stat")
band_stats_root = str(stats_dataset) if stats_dataset.exists() else None


class Args:
    root = "/kaggle/working"
    kaggle_input_root = str(data_root)
    folds = "2"
    epochs = 8
    batch_size = 8
    workers = 2
    lr_encoder = 2e-5
    lr_head = 1e-4
    loss_type = "huber"
    heavy_rain_weight_alpha = 0.5
    heavy_rain_weight_scale = 10.0
    heavy_rain_weight_max = 2.0
    use_two_head = True
    rain_bce_weight_0_1 = 0.10
    rain_bce_weight_1 = 0.10
    rain_bce_weight_5 = 0.05
    stats_samples_per_satellite = 1500
    seed = 42
    model_subdir = "swin_v2_temporal_two_head_convnext_fold2"
    band_stats_root = band_stats_root
    use_location_features = False
    location_metadata_path = None
    location_feature_mode = "full"
    encoder_name = "convnext_tiny"
    no_pretrained = False
    no_amp = True


summary_path = run(Args())
print(summary_path)
