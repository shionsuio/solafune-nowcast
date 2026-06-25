"""Create a manual coordinate CSV template for Solafune locations.

The generated file intentionally leaves latitude/longitude blank. Fill the
coordinates only if the competition host confirms manually curated coordinates
are allowed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kaggle_setup import ensure_kaggle_workspace
from project_paths import WorkspacePaths


def prepare_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame


def summarize_locations(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for (location, satellite), group in frame.groupby(
        ["name_location", "satellite_target"], sort=True
    ):
        months = ",".join(str(value) for value in sorted(group["datetime"].dt.month.unique()))
        rows.append(
            {
                "name_location": location,
                "split": split,
                "satellite_target": satellite,
                "samples": len(group),
                "months": months,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if args.kaggle_input_root:
        ensure_kaggle_workspace(root, Path(args.kaggle_input_root))

    paths = WorkspacePaths(root)
    train = prepare_metadata(paths.train_dir / "train_dataset.csv")
    evaluation = prepare_metadata(paths.evaluation_dir / "evaluation_target.csv")

    summary = pd.concat(
        [
            summarize_locations(train, "train"),
            summarize_locations(evaluation, "evaluation"),
        ],
        ignore_index=True,
    )
    summary = (
        summary.groupby(["name_location", "satellite_target"], as_index=False)
        .agg(
            split=("split", lambda values: ",".join(sorted(set(values)))),
            samples=("samples", "sum"),
            months=(
                "months",
                lambda values: ",".join(
                    str(month)
                    for month in sorted(
                        {int(value) for value in ",".join(values).split(",") if value}
                    )
                ),
            ),
        )
        .sort_values(["satellite_target", "name_location"])
    )
    summary.insert(2, "latitude", "")
    summary.insert(3, "longitude", "")
    summary["notes"] = ""

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Locations: {len(summary)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--kaggle-input-root", default=None)
    parser.add_argument("--output", default="data/location_coordinates_template.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
