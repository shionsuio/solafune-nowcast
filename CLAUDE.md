# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Solafune 降水ナウキャストコンペのワークスペース。静止衛星画像（過去30分・3枚、Himawari/GOES/Meteosat の3衛星）から GPM-IMERG 降水量マップ（41×41）を予測する回帰タスク。外部データ禁止。

**セッション開始時は必ず `PROJECT_HANDOFF.md` を読むこと。** 最新のLBスコア、実験結果、次のアクション、注意点がすべてそこに記録されている。実験の判断・結果が更新されたら PROJECT_HANDOFF.md も更新する。

## コマンド

```bash
# Python環境はローカル .venv（python3.12）を使う
./.venv/bin/python <script>

# 変更後の構文チェック（テストスイートは存在しない）
./.venv/bin/python -m py_compile src/swin_nowcast_v2.py src/run_swin_temporal_full.py

# JupyterLab（ローカル）
./scripts/start_lab.sh

# JupyterLab（Docker、GDAL等ネイティブ依存込み）
docker compose up
```

### Kaggle ワークフロー

学習・推論は Kaggle GPU カーネルで実行する。ローカルは実験コードの編集と診断のみ。

```bash
# カーネル push（kaggle_push/<experiment>/ に kernel-metadata.json + notebook）
kaggle kernels push -p kaggle_push/<experiment>

# データセット更新（kaggle_upload/<dataset>/ に dataset-metadata.json + files）
kaggle datasets version -p kaggle_upload/<dataset> -m "message"
```

- カーネルの notebook は `src/` のモジュールをセルに埋め込む or dataset 経由で参照する形式
- モデル重み・band stats は `kaggle_upload/` 経由で Kaggle dataset 化してカーネルに渡す
- PyTorch 2.10 は P100 非対応、T4 を使う

## アーキテクチャ

### コア構造

- `src/swin_nowcast_v2.py` — 本線モデルの中核。`Config` / `prepare_metadata` / `make_folds` / `train_fold` を提供。Swin-T エンコーダ + 衛星別 normalization/stem/embedding、location-disjoint 5-fold、Huber/two-head loss 対応
- `src/run_swin_temporal_full.py` — 本線の学習エントリポイント。swin_nowcast_v2 を import し、temporal差分+mean/stdチャネルを追加した設定で学習。CLI引数で loss / fold / sample weight 等を制御
- `src/run_*.py` — 各種実験・診断のエントリポイント（ablation, probe, OOF診断, adversarial診断）
- `src/experiments/` — ablation系の共有ロジック
- `src/project_paths.py` — パス解決の一元管理。`WorkspacePaths` がローカルレイアウトを、`find_solafune_input_root` が `/kaggle/input` 配下の自動検出を担う。新スクリプトは必ずここを使う
- `src/kaggle_setup.py` — `ensure_kaggle_workspace` で Kaggle 環境にローカルと同じレイアウトを再現（同一コードがローカル/Kaggle両対応になる仕組み）

### ディレクトリ

- `train_dataset/` / `evaluation_dataset/` / `sample_submission/` — コンペデータ（git管理外）
- `models/<model_subdir>/` — fold別チェックポイント（`best_fold{n}.pth`）と band stats
- `outputs/` — 診断CSV、提出zip（`submission_blends/`, `submission_local_blends/`, `adversarial_diagnostics/` 等）
- `kaggle_push/<experiment>/` — Kaggleカーネル一式（実験ごとに1ディレクトリ）
- `kaggle_upload/<dataset>/` — Kaggle dataset アップロード用ステージング

### コンペ固有の重要事実

- **CV/LB乖離が大きい**（adversarial AUC 0.934）。通常CVは楽観的で、eval-like weighted RMSE / top30 RMSE も併せて評価する
- 衛星ごとに解像度・分布が異なる（GOES/Meteosat は train/eval 乖離大、Himawari は近い）。衛星別のブレンド・後処理が有効
- location は train/eval で overlap 0。location 直接特徴は危険
- fold は location-disjoint。fold2/fold3 は eval-like 評価で悪化しやすい
- Public LB は test の35%のみ。Public への過剰適合に注意
