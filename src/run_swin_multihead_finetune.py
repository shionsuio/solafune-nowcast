"""W4C23-style multi-head + gate-controller fine-tune on a trained checkpoint.

Attaches 5 parallel prediction heads (dropout p=0/0.1/0.2/0.3/0.4, warm-started
from the base head) and a sigmoid-gated per-pixel controller to a frozen
SwinNowcaster body, then trains in two stages:
  stage 2: frozen body, each head trained with its own two-head loss
  stage 3: frozen body+heads, controller trained on the gated combination

Reference: Weather4cast 2023 winner (arXiv:2401.09424) — the 5-head+controller
mechanism was their entire winning margin, concentrated at heavy rain.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from kaggle_setup import ensure_kaggle_workspace
from swin_nowcast_v2 import (
    Config,
    NowcastingDataset,
    SwinNowcaster,
    attach_location_metadata,
    attach_sample_weights,
    compute_training_loss,
    get_device,
    load_stats,
    make_folds,
    make_loader,
    make_training_loss,
    original_scale_rmse,
    prepare_metadata,
    satellite_directories,
    seed_everything,
)

DROPOUT_PROBS = (0.0, 0.1, 0.2, 0.3, 0.4)


class MultiHeadGateNowcaster(nn.Module):
    def __init__(self, base: SwinNowcaster) -> None:
        super().__init__()
        assert base.config.use_two_head, "multi-head fine-tune expects a two-head base"
        self.base = base
        self.config = base.config
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout2d(p), copy.deepcopy(base.head))
                for p in DROPOUT_PROBS
            ]
        )
        self.controller = nn.Sequential(
            nn.Conv2d(self.config.decoder_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, len(DROPOUT_PROBS), 1),
        )
        # sigmoid(-1.386) ~= 0.2 so the initial gated sum ~= mean of the 5 heads
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.constant_(self.controller[-1].bias, -1.386)

    def forward_decoded(
        self,
        image: torch.Tensor,
        satellite_id: torch.Tensor,
        temporal_features: torch.Tensor,
        missing_flag: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base
        config = self.config
        if config.use_satellite_stem:
            image = base.stem(image, satellite_id)
        else:
            image = base.shared_stem(image)
        features = base.encoder(image)
        decoded = base.decoder(features)
        context = temporal_features.clone()
        if not config.use_month_features:
            context[:, :2] = 0
        if not config.use_hour_features:
            context[:, 2:4] = 0
        if not config.use_missing_flag:
            missing_flag = torch.zeros_like(missing_flag)
        condition = base.context_mlp(torch.cat([context, missing_flag], dim=1))
        if config.use_satellite_embedding:
            condition = condition + base.satellite_embedding(satellite_id)
        return decoded + condition[:, :, None, None]

    def head_outputs(
        self, decoded: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        size = (self.config.target_size, self.config.target_size)
        for head in self.heads:
            raw = F.interpolate(
                head(decoded), size=size, mode="bilinear", align_corners=False
            )
            amount_log = F.softplus(raw[:, :1])
            rain_logits = raw[:, 1:]
            prediction = amount_log * torch.sigmoid(rain_logits[:, :1])
            outputs.append((prediction, rain_logits))
        return outputs

    def gates(self, decoded: torch.Tensor) -> torch.Tensor:
        size = (self.config.target_size, self.config.target_size)
        logits = F.interpolate(
            self.controller(decoded), size=size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(logits)

    def combine(
        self, predictions: list[torch.Tensor], gates: torch.Tensor
    ) -> torch.Tensor:
        stacked = torch.cat(predictions, dim=1)
        return (gates * stacked).sum(dim=1, keepdim=True)

    def forward(
        self,
        image: torch.Tensor,
        satellite_id: torch.Tensor,
        temporal_features: torch.Tensor,
        missing_flag: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self.forward_decoded(image, satellite_id, temporal_features, missing_flag)
        outputs = self.head_outputs(decoded)
        return self.combine([pred for pred, _ in outputs], self.gates(decoded))


def build_config_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> Config:
    stored = checkpoint["config"]
    field_names = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in stored.items() if k in field_names}
    kwargs.update(
        root=str(Path(args.root).resolve()),
        batch_size=args.batch_size,
        workers=args.workers,
        pretrained=False,
        use_amp=False,
        swin_model_subdir=args.model_subdir,
        band_stats_root=None,
        location_metadata_path=None,
        sample_weight_path=None,
        pseudo_label_npz=None,
        pseudo_label_csv=None,
    )
    return Config(**kwargs)


def validate(
    model: MultiHeadGateNowcaster,
    loader,
    device: torch.device,
    per_head: bool = False,
) -> dict:
    model.eval()
    n_heads = len(model.heads)
    combined_sq = mean_sq = 0.0
    head_sq = [0.0] * n_heads
    pixels = 0
    with torch.no_grad():
        for image, satellite_id, temporal, missing, target, _ in loader:
            image = image.to(device, non_blocking=device.type == "cuda")
            satellite_id = satellite_id.to(device, non_blocking=device.type == "cuda")
            temporal = temporal.to(device, non_blocking=device.type == "cuda")
            missing = missing.to(device, non_blocking=device.type == "cuda")
            decoded = model.forward_decoded(image, satellite_id, temporal, missing)
            outputs = model.head_outputs(decoded)
            predictions = [pred for pred, _ in outputs]
            combined = model.combine(predictions, model.gates(decoded)).cpu()
            sq, count = original_scale_rmse(combined, target)
            combined_sq += sq
            pixels += count
            mean_prediction = torch.stack(predictions).mean(dim=0).cpu()
            sq, _ = original_scale_rmse(mean_prediction, target)
            mean_sq += sq
            if per_head:
                for i, prediction in enumerate(predictions):
                    sq, _ = original_scale_rmse(prediction.cpu(), target)
                    head_sq[i] += sq
    result = {
        "combined": math.sqrt(combined_sq / pixels),
        "mean": math.sqrt(mean_sq / pixels),
    }
    if per_head:
        result["heads"] = [math.sqrt(sq / pixels) for sq in head_sq]
    return result


def train_stage(
    model: MultiHeadGateNowcaster,
    stage: str,
    epochs: int,
    lr: float,
    config: Config,
    train_loader,
    validation_loader,
    device: torch.device,
    history: list[dict],
    checkpoint_path: Path,
    best_rmse: float,
) -> float:
    if stage == "heads":
        parameters = list(model.heads.parameters())
    else:
        parameters = list(model.controller.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = make_training_loss(config)
    stage3_config = dataclasses.replace(config, use_two_head=False)

    for epoch in range(1, epochs + 1):
        model.eval()  # keep BatchNorm in the frozen body on running stats
        if stage == "heads":
            model.heads.train()
        else:
            model.controller.train()
        loss_sum = 0.0
        sample_count = 0
        for image, satellite_id, temporal, missing, target, metadata in tqdm(
            train_loader, desc=f"{stage} epoch {epoch}", leave=False
        ):
            image = image.to(device, non_blocking=device.type == "cuda")
            satellite_id = satellite_id.to(device, non_blocking=device.type == "cuda")
            temporal = temporal.to(device, non_blocking=device.type == "cuda")
            missing = missing.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            sample_weight = metadata.get("sample_weight")
            if sample_weight is not None:
                sample_weight = sample_weight.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                decoded = model.forward_decoded(image, satellite_id, temporal, missing)
            if stage == "heads":
                outputs = model.head_outputs(decoded)
                loss = sum(
                    compute_training_loss(
                        criterion, prediction, target, config, rain_logits,
                        sample_weight=sample_weight,
                    )
                    for prediction, rain_logits in outputs
                ) / len(outputs)
            else:
                with torch.no_grad():
                    outputs = model.head_outputs(decoded)
                combined = model.combine(
                    [pred for pred, _ in outputs], model.gates(decoded)
                )
                loss = compute_training_loss(
                    criterion, combined, target, stage3_config, None,
                    sample_weight=sample_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {stage} epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += loss.item() * image.shape[0]
            sample_count += image.shape[0]
        scheduler.step()

        metrics = validate(
            model, validation_loader, device, per_head=(stage == "heads")
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "val_rmse_combined": metrics["combined"],
            "val_rmse_mean": metrics["mean"],
        }
        if "heads" in metrics:
            for i, value in enumerate(metrics["heads"]):
                row[f"val_rmse_head{i}"] = value
        history.append(row)
        heads_note = (
            " heads=" + ",".join(f"{v:.4f}" for v in metrics["heads"])
            if "heads" in metrics
            else ""
        )
        print(
            f"{stage} epoch={epoch:02d} train_loss={row['train_loss']:.5f} "
            f"val_combined={metrics['combined']:.5f} val_mean={metrics['mean']:.5f}"
            + heads_note
        )
        tracked = metrics["mean"] if stage == "heads" else metrics["combined"]
        if tracked < best_rmse:
            best_rmse = tracked
            torch.save(
                {
                    "config": asdict(config),
                    "stage": stage,
                    "epoch": epoch,
                    "validation_rmse": tracked,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
    return best_rmse


def run(args: argparse.Namespace) -> dict:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))
    device = get_device()

    source_dir = root / "models" / args.model_subdir
    checkpoint = torch.load(
        source_dir / f"best_fold{args.fold}.pth", map_location=device, weights_only=False
    )
    config = build_config_from_checkpoint(checkpoint, args)
    seed_everything(config.seed + args.fold)

    base = SwinNowcaster(config).to(device)
    base.load_state_dict(checkpoint["model_state_dict"])
    model = MultiHeadGateNowcaster(base).to(device)
    print(
        f"loaded {args.model_subdir} fold{args.fold} "
        f"(stored val_rmse={checkpoint.get('validation_rmse', float('nan')):.5f})"
    )

    stats = load_stats(source_dir / f"band_stats_fold{args.fold}.json")
    dataframe = prepare_metadata(config.paths.train_dir / "train_dataset.csv")
    folds = make_folds(dataframe, config.n_folds)
    fold = folds[args.fold]
    dataframe = attach_location_metadata(dataframe, config)
    dataframe = attach_sample_weights(dataframe, config)
    train_frame = dataframe.iloc[fold["train_indices"]].copy()
    validation_frame = dataframe.iloc[fold["validation_indices"]].copy()
    if args.limit:
        train_frame = train_frame.head(args.limit)
        validation_frame = validation_frame.head(args.limit)
    directories = satellite_directories(config, "train")
    train_loader = make_loader(
        NowcastingDataset(train_frame, directories, stats, config, has_target=True, augment=True),
        config, device, shuffle=True,
    )
    validation_loader = make_loader(
        NowcastingDataset(validation_frame, directories, stats, config, has_target=True, augment=False),
        config, device, shuffle=False,
    )

    output_dir = root / "models" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    baseline = validate(model, validation_loader, device, per_head=True)
    print(
        f"before fine-tune: val_combined={baseline['combined']:.5f} "
        f"val_mean={baseline['mean']:.5f} heads="
        + ",".join(f"{v:.4f}" for v in baseline["heads"])
    )
    history.append(
        {
            "stage": "init", "epoch": 0, "train_loss": float("nan"),
            "val_rmse_combined": baseline["combined"], "val_rmse_mean": baseline["mean"],
            **{f"val_rmse_head{i}": v for i, v in enumerate(baseline["heads"])},
        }
    )

    checkpoint_path = output_dir / f"best_fold{args.fold}.pth"
    best = train_stage(
        model, "heads", args.heads_epochs, args.lr_heads, config,
        train_loader, validation_loader, device, history, checkpoint_path,
        best_rmse=float("inf"),
    )
    print(f"stage 2 done: best mean-ensemble val_rmse={best:.5f}")
    best = train_stage(
        model, "controller", args.controller_epochs, args.lr_controller, config,
        train_loader, validation_loader, device, history, checkpoint_path,
        best_rmse=best,
    )
    print(f"stage 3 done: best val_rmse={best:.5f}")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / f"history_fold{args.fold}.csv", index=False)
    stats_source = source_dir / f"band_stats_fold{args.fold}.json"
    stats_target = output_dir / f"band_stats_fold{args.fold}.json"
    if not stats_target.exists():
        stats_target.write_bytes(stats_source.read_bytes())
    return {"fold": args.fold, "validation_rmse": best, "history": history_frame}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--model-subdir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--heads-epochs", type=int, default=3)
    parser.add_argument("--controller-epochs", type=int, default=3)
    parser.add_argument("--lr-heads", type=float, default=2e-4)
    parser.add_argument("--lr-controller", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
