from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


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
            ["git", "clone", "https://github.com/shionsuio/solafune-nowcast.git", str(repo)],
            check=True,
        )
    sys.path.insert(0, str(repo / "src"))
    return repo


def ensure_workspace(data_root: Path) -> None:
    from kaggle_setup import ensure_kaggle_workspace

    ensure_kaggle_workspace(Path("/kaggle/working"), data_root)


def all_input_files(pattern: str) -> list[Path]:
    return sorted(Path("/kaggle/input").rglob(pattern))


def path_score(path: Path, keywords: tuple[str, ...], reject: tuple[str, ...]) -> int:
    text = str(path).lower().replace("-", "_")
    if any(word in text for word in reject):
        return -10_000
    return sum(1000 for word in keywords if word in text) + len(text)


def pick_file(filename: str, keywords: tuple[str, ...], reject: tuple[str, ...] = ()) -> Path:
    matches = [path for path in all_input_files(filename) if path.name == filename]
    if not matches:
        raise FileNotFoundError(filename)
    matches = sorted(matches, key=lambda path: path_score(path, keywords, reject))
    selected = matches[-1]
    if path_score(selected, keywords, reject) < 0:
        raise FileNotFoundError(f"No acceptable match for {filename}: {matches}")
    print(f"picked {filename}: {selected}")
    return selected


def prepare_model_dir(
    model_name: str,
    folds: list[int],
    keywords: tuple[str, ...],
    reject: tuple[str, ...],
) -> Path:
    model_dir = Path("/kaggle/working/models") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        checkpoint = pick_file(f"best_fold{fold}.pth", keywords, reject)
        stats = pick_file(f"band_stats_fold{fold}.json", keywords, reject)
        shutil.copy2(checkpoint, model_dir / checkpoint.name)
        shutil.copy2(stats, model_dir / stats.name)
    return model_dir


def run_oof_variant(
    *,
    model_name: str,
    folds: list[int],
    use_two_head: bool,
    keywords: tuple[str, ...],
    reject: tuple[str, ...] = (),
) -> None:
    from run_oof_diagnostics import run

    model_dir = prepare_model_dir(model_name, folds, keywords, reject)

    print(f"\n=== OOF {model_name} folds={folds} use_two_head={use_two_head} ===")
    print("model_dir:", model_dir)
    run(
        SimpleNamespace(
            root="/kaggle/working",
            kaggle_input_root=None,
            checkpoint_root=None,
            folds=",".join(str(fold) for fold in folds),
            batch_size=16,
            workers=2,
            model_subdir=model_name,
            output_dir=f"outputs/oof_compare/{model_name}",
            use_temporal_differences=True,
            use_temporal_summary=True,
            use_two_head=use_two_head,
            use_location_features=False,
            location_metadata_path=None,
            no_amp=False,
        )
    )


def main() -> None:
    gpu = install_p100_torch_if_needed()
    install_runtime_deps()
    clone_repo()

    import torch

    print("GPU:", gpu or torch.cuda.get_device_name(0), "torch:", torch.__version__)
    assert torch.cuda.is_available(), "GPU is required"

    data_root = Path("/kaggle/input/solafune-dataset-v2")
    if not data_root.exists():
        data_root = Path("/kaggle/input/datasets/suioshion/solafune-dataset-v2")
    ensure_workspace(data_root)

    print("Input roots:")
    for root in sorted(Path("/kaggle/input").glob("*")):
        print(" ", root)

    run_oof_variant(
        model_name="swin_v2_temporal_two_head_oof_folds012",
        folds=[0, 1, 2],
        use_two_head=True,
        keywords=("two_head",),
        reject=(),
    )
    run_oof_variant(
        model_name="swin_v2_temporal_weighted_huber_oof_fold2",
        folds=[2],
        use_two_head=False,
        keywords=("weighted_huber",),
        reject=("two_head",),
    )


if __name__ == "__main__":
    main()
