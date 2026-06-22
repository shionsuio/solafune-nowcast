"""Fail fast when the baseline/experiment environment is incomplete."""

from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = {
    "numpy": "NumPy",
    "pandas": "pandas",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "Matplotlib",
    "seaborn": "seaborn",
    "rasterio": "rasterio",
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "lightgbm": "LightGBM",
    "timm": "timm",
    "einops": "einops",
    "torchmetrics": "torchmetrics",
    "albumentations": "Albumentations",
    "cv2": "OpenCV",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    failures: list[str] = []
    for module_name, display_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {display_name}: {version}")
        except Exception as exc:
            failures.append(f"{display_name}: {exc}")
            print(f"[ERROR] {display_name}: {exc}")

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        print(f"CUDA available: {torch.cuda.is_available()}")
        mps = getattr(torch.backends, "mps", None)
        print(f"MPS built: {bool(mps and mps.is_built())}")
        print(f"MPS available: {bool(mps and mps.is_available())}")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
