"""Probe-style smoke tests for alternative backbones."""

from __future__ import annotations

from pathlib import Path

from convnext_nowcast_v2 import ConvNeXtNowcaster, train_convnext_fold
from experiment_pipelines import build_sampled_location_fold
from experiment_utils import load_train_dataframe
from swin_nowcast_v2 import Config, get_device
from unet_nowcast_v2 import UNetNowcaster, train_unet_fold


def run_unet_probe(args) -> dict:
    root = Path(args.root).resolve()
    config = Config(
        root=str(root),
        encoder_size=args.encoder_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=0,
        pretrained=False,
        use_amp=False,
        stats_samples_per_satellite=300,
        unet_model_subdir="unet_probe",
    )
    dataframe = load_train_dataframe(config)
    sampled, fold = build_sampled_location_fold(
        config,
        dataframe,
        args.fold,
        args.train_rows,
        args.validation_rows,
        config.seed,
    )
    model = UNetNowcaster(config, base_channels=args.base_channels)
    print("parameters", sum(parameter.numel() for parameter in model.parameters()))
    result = train_unet_fold(
        config,
        sampled,
        fold,
        device=get_device(),
        base_channels=args.base_channels,
    )
    print(result)
    return result


def run_convnext_probe(args) -> dict:
    root = Path(args.root).resolve()
    config = Config(
        root=str(root),
        encoder_size=args.encoder_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=0,
        pretrained=False,
        use_amp=False,
        stats_samples_per_satellite=200,
        convnext_model_subdir="convnext_probe",
    )
    dataframe = load_train_dataframe(config)
    sampled, fold = build_sampled_location_fold(
        config,
        dataframe,
        args.fold,
        args.train_rows,
        args.validation_rows,
        config.seed,
    )
    model = ConvNeXtNowcaster(config)
    print("parameters", sum(parameter.numel() for parameter in model.parameters()))
    result = train_convnext_fold(config, sampled, fold, device=get_device())
    print(result)
    return result
