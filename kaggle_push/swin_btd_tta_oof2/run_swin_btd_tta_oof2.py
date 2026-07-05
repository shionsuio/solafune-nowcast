from pathlib import Path
import shutil
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
                sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "pillow==11.3.0"],
            check=True,
        )
    return gpu


gpu_name = install_p100_torch_if_needed()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "rasterio"], check=True)

repo = Path("/kaggle/working/solafune-nowcast")
if repo.exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/shionsuio/solafune-nowcast.git", str(repo)], check=True)

sys.path.insert(0, str(repo / "src"))

import numpy as np
import torch

print("GPU:", gpu_name or torch.cuda.get_device_name(0), "torch:", torch.__version__)
assert torch.cuda.is_available(), "GPU is required"

MODEL_SUBDIR = "swin_v2_temporal_two_head_btd_tta"
model_dir = Path("/kaggle/working/models") / MODEL_SUBDIR
model_dir.mkdir(parents=True, exist_ok=True)
checkpoints = [p for p in Path("/kaggle/input").rglob("best_fold*.pth") if "btd" in str(p).lower().replace("-", "_")]
stats = [p for p in Path("/kaggle/input").rglob("band_stats_fold*.json") if "btd" in str(p).lower().replace("-", "_")]
print("btd checkpoints:", len(checkpoints), "stats:", len(stats))
assert len(checkpoints) == 5 and len(stats) == 5
for path in checkpoints + stats:
    shutil.copy2(path, model_dir / path.name)

from export_oof_predictions import run


class Args:
    root = "/kaggle/working"
    kaggle_input_root = str(
        Path("/kaggle/input/solafune-dataset-v2")
        if Path("/kaggle/input/solafune-dataset-v2").exists()
        else Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")
    )
    model_subdir = MODEL_SUBDIR
    folds = "2"
    batch_size = 16
    workers = 2
    output_dir = "outputs/oof_predictions"
    use_temporal_differences = True
    use_temporal_summary = True
    use_two_head = True
    band_mode = "matched6_btd"
    save_target = True
    limit = None
    tta = True


output_dir = run(Args())

prediction = np.load(output_dir / "oof_fold2.npz")["prediction"].astype(np.float32)
target = np.load(output_dir.parent / "targets" / "oof_fold2.npz")["target"].astype(np.float32)
tile = np.sqrt(((prediction - target) ** 2).reshape(len(prediction), -1).mean(axis=1)).mean()
pooled = np.sqrt(((prediction - target) ** 2).mean())
print(f"TTA fold2: tile_rmse={tile:.6f} pooled_rmse={pooled:.6f}")
print("reference (no TTA, btd fold2): tile 0.5922-family — compare locally against stored OOF")
