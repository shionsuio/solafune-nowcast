#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -d kaggle_upload/train_dataset
test -d kaggle_upload/evaluation_dataset
test -d kaggle_upload/sample_submission
test -f kaggle_upload/dataset-metadata.json

echo "Kaggle account:"
kaggle config view
echo
echo "Creating private dataset: suioshion/solafune-nowcasting-private-20260622"
echo "Source directories are symlinks; the original data is not duplicated."

kaggle datasets create \
  --path kaggle_upload \
  --dir-mode zip \
  --keep-tabular
