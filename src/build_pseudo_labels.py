"""Generate transductive pseudo-labels for the evaluation set.

For fold-honest screening the generator must only use models that never saw
the screening fold's training locations. Default: fold2 two-head + fold2
stable checkpoints, blended 50/50 (the validated submission recipe).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swin_nowcast_v2 import Config, get_device, predict_fold, prepare_metadata


def build_config(root: Path, model_subdir: str, use_two_head: bool, args) -> Config:
    return Config(
        root=str(root),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        use_amp=False,
        use_two_head=use_two_head,
        use_temporal_differences=True,
        use_temporal_summary=True,
        swin_model_subdir=model_subdir,
        band_stats_root=str(root / "models" / model_subdir),
    )


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    device = get_device()
    print("device:", device)

    two_config = build_config(root, args.two_head_subdir, True, args)
    stable_config = build_config(root, args.stable_subdir, False, args)

    evaluation_frame = prepare_metadata(
        two_config.paths.evaluation_dir / "evaluation_target.csv"
    )
    print("evaluation samples:", len(evaluation_frame))

    two_predictions = predict_fold(two_config, evaluation_frame, args.fold, device)
    stable_predictions = predict_fold(stable_config, evaluation_frame, args.fold, device)
    blended = (
        args.two_head_weight * two_predictions
        + (1.0 - args.two_head_weight) * stable_predictions
    )
    print(
        f"blend mean={blended.mean():.6f} "
        f"p95={np.quantile(blended, 0.95):.6f} max={blended.max():.6f}"
    )

    output_dir = root / "outputs" / "pseudo_labels" / f"fold{args.fold}_blend50"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "pseudo_predictions.npz"
    csv_path = output_dir / "pseudo_index.csv"
    np.savez_compressed(npz_path, predictions=blended.astype(np.float16))
    evaluation_frame[["unique_id", "name_location", "satellite_target"]].to_csv(
        csv_path, index=False
    )
    print("Saved:", npz_path, csv_path)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--two-head-subdir", default="swin_v2_temporal_two_head_oof")
    parser.add_argument("--stable-subdir", default="swin_v2_temporal_stable")
    parser.add_argument("--two-head-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
