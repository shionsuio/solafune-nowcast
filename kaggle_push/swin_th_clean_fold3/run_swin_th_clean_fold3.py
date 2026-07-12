from pathlib import Path
import subprocess
import sys

repo = Path("/kaggle/working/solafune-nowcast")
if repo.exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/shionsuio/solafune-nowcast.git", str(repo)], check=True)

sys.path.insert(0, str(repo / "src"))

import torch

print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)
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
    folds = "3"
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
    model_subdir = "swin_th_clean_fold3"
    band_stats_root = band_stats_root
    use_location_features = False
    location_metadata_path = None
    location_feature_mode = "full"
    no_pretrained = False
    no_amp = True
    align_frames = True
    exclude_bad_labels = True


summary_path = run(Args())
print(summary_path)
