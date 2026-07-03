# Solafune Nowcast 実験引き継ぎメモ

最終更新: 2026-07-03  
目的: セッションを跨いでも、これまでの判断・結果・次アクションを復元できるようにする。

**重要: 2026-07-03にLB全体が再計算された**（評価関数が欠損チャネル画像のサンプルを採点対象外に変更）。本ドキュメントのLB値はすべて再計算後の値。詳細は「I. 評価メトリクス変更」節。

## 現在の結論

現在の Public LB best は以下。

| submission | Public LB |
|---|---:|
| `submission_swin80_twohead_hg20.zip` | `0.6649110102093378` |

この best は、Public 最良だった `submission_swin80_temporal20.zip` をベースに、two-head モデルの差分を Himawari/GOES のみに 20% 入れたもの。

```text
Himawari / GOES:
  pred = 0.8 * swin80_temporal20 + 0.2 * twohead

Meteosat:
  pred = swin80_temporal20
```

Meteosat に two-head を入れると悪化する可能性が高い。現時点では Meteosat は base 維持。

## CV-LB相関検証（2026-07-02、重要）

eval-like CVがLB順位を予測できるかを、OOFで正確に再現できる提出5件（fold2キャリブレーションファミリー）で検証した。

### 前提の訂正

zip代数（`2×swin90_temporal10 − swin80_temporal20 = submission.zip`、誤差3e-6）とLB一致で確認した事実:

- blend系の「temporal」成分は mixed ensemble ではなく **weighted_huber fold2 単体モデル**（`submission_swin_temporal_weighted_fold2.zip` と同一）
- `swin80_temporal20 = 0.8 × base_swin + 0.2 × weighted_huber_fold2(単体)`
- base swin_v2（非temporal）の重みはColab学習でDrive未保存のため**消失**。swin系提出のOOF遡及再現は不可能

### キャリブレーション結果（fold2、7260サンプル）

| recipe | Public LB（再計算後） | fold2 RMSE | eval-like sqrt | top30 |
|---|---:|---:|---:|---:|
| twohead100 | 0.667537 | 1.2268 | 1.2748 | 1.5515 |
| wh50_twohead50 | 0.668032 | 1.2212 | 1.2670 | 1.5366 |
| wh80_twohead20 | 0.668873 | 1.2294 | 1.2714 | 1.5303 |
| wh_hg20 | 0.669099 | 1.2345 | 1.2751 | 1.5303 |
| wh100 | 0.669659 | 1.2396 | 1.2781 | 1.5271 |

（再計算後も順位は完全に保存 → Spearman相関の結論は不変）

### LBとの相関（n=5）

| metric | Spearman | p |
|---|---:|---:|
| **通常 pooled RMSE** | **+0.9** | 0.037 |
| eval-like sqrt weighted | +0.7 | 0.19 |
| eval-like weighted | +0.4 | 0.50 |
| **top30** | **−0.9** | 0.037 |
| **top10** | **−0.9** | 0.039 |

### 結論

1. **提出選別は通常 pooled RMSE を使う**（Spearman +0.9）。
2. **top30/top10（adversarial上位サブセット）はLBと有意に逆相関**。この指標で選ぶとLBで悪化する。adversarial weighted trainingの優先度も下げるべき（weightの方向がLB改善と逆の可能性）。
3. **LBはtwohead比率に単調改善**（wh100 0.6696 → twohead100 0.6675）。過去の「two-headはPublic悪化」はbase swin成分の欠落と交絡していた。two-head自体はLB善玉。
4. HG局所blendのhg05→hg20単調改善とも一貫。**hg20より先（hg30/hg40、Meteosatへの小さいtwohead注入）を試す価値が高い**。既存zipの代数だけでGPU不要で作れる。

関連ファイル:

- `src/export_oof_predictions.py` — OOFピクセル予測エクスポート（stable 5fold / twohead f012 / wh f2 済み、`outputs/oof_predictions/`）
- `src/build_cv_lb_correlation.py` — fold2ファミリーの指標計算と相関
- `src/build_calibration_submissions.py` — キャリブレーションzip生成（zip代数）
- `outputs/cv_lb_correlation/` — 結果CSV
- `outputs/lb_history.csv` — 全提出LB履歴

## 主要スコア（2026-07-03 再計算後）

| submission | Public LB |
|---|---:|
| `submission_swin80_twohead_hg20.zip` | `0.6649110102093378` |
| `submission_swin80_5x5_hg30.zip` | `0.6649399690321076` |
| `submission_swin80_twohead_hg15.zip` | `0.6649643832317802` |
| `submission_swin80_5x5_hg20.zip` | `0.6649722355255615` |
| `submission_swin80_twohead_hg10.zip` | `0.6650369920990741` |
| `submission_swin80_twohead_hg05.zip` | `0.6651288273839465` |
| `submission_swin80_twohead_goes10.zip` | `0.6651446978678026` |
| `submission_swin80_twohead_goes05.zip` | `0.6651878276969626` |
| `submission_swin80_temporal20.zip` | `0.6652398764998774` |
| `submission_swin70_temporal30.zip` | `0.6652739510069082` |
| `submission_swin90_temporal10.zip` | `0.665354069142379` |
| `submission.zip` | `0.6656162651494983` |
| `submission_twohead50_stable50_5x5.zip` | `0.6671302640502683` |
| `submission_swin_twohead_fold02_stable134.zip` | `0.6675368076867854` |
| `submission_pseudo_fold2.zip` | `0.6728310563265988` |
| `submission_unet_v2.zip` | `0.694462043324256` |

## 局所 blend の推移

Meteosat を固定し、Himawari/GOES だけ two-head 差分を注入した。

| blend | Public LB |
|---|---:|
| HG05 | `0.6651288273839465` |
| HG10 | `0.6650369920990741` |
| HG15 | `0.6649643832317802` |
| HG20 | `0.6649110102093378` |

単調改善だが改善幅は小さい。

```text
HG05 → HG10: -0.000092
HG10 → HG15: -0.000073
HG15 → HG20: -0.000053
```

この系統は安全に少し稼ぐ枠。0.64台前半を狙う主戦場ではない。

## モデルごとの判断

### Swin temporal

本線。`submission_swin80_temporal20.zip` が長く Public best だった。

### U-Net

弱い。

| model | Public LB |
|---|---:|
| U-Net single | `0.694462043324256` |
| Swin80 + U-Net20 | `0.6671065678179597` |
| Swin70 + U-Net30 | `0.6686899338875454` |

本線から外してよい。

### ConvNeXt — ネガティブ確定（probe済み、再試行不要）

過去のprobeで大敗: fold0 best 1.4453（Swin stable 1.2848）、fold1 best 1.3032（Swin stable 1.0523）。

**アーキ知見: U-Net弱 + ConvNeXt大敗 → CNN系エンコーダはこのタスクに構造的に不向き。window attention（Swin）が本質的に効いている。** アーキ多様性を狙うなら同系統の容量増し（Swin-S、`kaggle_push/swin_small_fold2/` 準備済み、`encoder_name` Config対応済み）。MaxViT等のconvハイブリッドは期待値低。ConvNeXt fold2カーネルは投入前に中止・削除した。

### two-head

fold CV では改善するが、提出でそのまま使うと悪化した。

| fold | stable | two-head | 差分 |
|---:|---:|---:|---:|
| 0 | `1.2848` | `1.2742` | 改善 |
| 1 | `1.0523` | `1.0427` | 改善 |
| 2 | `1.2500` | `1.2268` | 改善 |

ただし full/mixed two-head 提出:

```text
submission_swin_twohead_fold02_stable134.zip
0.6674952809153677
```

悪化。原因は Meteosat を下げすぎた可能性が高い。

## train/eval 分布乖離

adversarial validation を実施。  
metadata + last observation image stats で train/eval を分類。

```text
Adversarial AUC = 0.934292
```

かなり高い。train/eval は明確に分布が違う。CV/LB 乖離の大きな原因。

### 初期 AUC

| 特徴量 | AUC |
|---|---:|
| metadata no location | `0.772` |
| image stats only | `0.634` |
| satellite + image stats | `0.648` |
| metadata + image stats LR | `0.799` |
| metadata + image stats RF | `0.879` |
| 全件 RF | `0.934` |

### 衛星別 image stats AUC

| satellite | AUC |
|---|---:|
| GOES | `0.895` |
| Himawari | `0.629` |
| Meteosat | `0.816` |

GOES と Meteosat は train/eval の画像分布がかなり違う。Himawari は比較的近い。

## 月分布のズレ

| month | eval | train |
|---:|---:|---:|
| 1 | `40.7%` | `25.5%` |
| 4 | `19.7%` | `7.0%` |
| 6 | `9.9%` | `0.0%` |
| 7 | `0.0%` | `3.6%` |
| 8 | `0.0%` | `7.3%` |
| 9 | `4.8%` | `13.9%` |
| 10 | `0.0%` | `7.2%` |
| 11 | `14.8%` | `21.1%` |
| 12 | `5.1%` | `3.6%` |

eval は 1月/4月/6月が多い。train には eval にない 7/8/10月がある。

## eval-like CV

通常 OOF stable:

```text
RMSE = 1.076242
```

eval-like weighted:

```text
sqrt weighted RMSE = 1.131733
weighted RMSE      = 1.135664
top30 RMSE         = 1.248549
top10 RMSE         = 1.239301
```

通常 CV は楽観的。test/eval っぽい train 領域では誤差が大きい。

### fold別 eval-like

| fold | normal RMSE | eval-like sqrt RMSE | top30 RMSE |
|---:|---:|---:|---:|
| 0 | `1.2848` | `1.2881` | `1.3046` |
| 1 | `1.0523` | `1.0708` | `0.9621` |
| 2 | `1.2500` | `1.2922` | `1.5487` |
| 3 | `0.8691` | `0.9809` | `1.5144` |
| 4 | `0.8858` | `0.9057` | `0.9616` |

fold2 と fold3 は eval-like でかなり悪化する。

## two-head 差分診断

`submission_swin80_temporal20.zip` と `submission_swin_twohead_fold02_stable134.zip` を比較。

全体:

```text
base mean    = 0.259381
twohead mean = 0.254818
diff mean    = -0.004564
```

two-head は全体的に予測を下げている。

### 衛星別

| satellite | base mean | twohead mean | diff |
|---|---:|---:|---:|
| GOES | `0.3615` | `0.3632` | `+0.0017` |
| Himawari | `0.2968` | `0.2964` | `-0.0004` |
| Meteosat | `0.1424` | `0.1286` | `-0.0139` |

Meteosat を大きく下げている。full two-head 提出悪化の主因候補。

### 月別

| month | base mean | twohead mean | diff |
|---:|---:|---:|---:|
| 11 | `0.6500` | `0.6422` | `-0.0079` |
| 6 | `0.3356` | `0.3337` | `-0.0020` |
| 4 | `0.2848` | `0.2756` | `-0.0093` |
| 1 | `0.1157` | `0.1119` | `-0.0037` |
| 12 | `0.0561` | `0.0517` | `-0.0044` |

eval-like で重要な 1/4/11/12月を下げている。Public 悪化と整合的。

## 重要なファイル

### 診断

- `src/build_adversarial_diagnostics.py`
- `outputs/adversarial_diagnostics/adversarial_scores.csv`
- `outputs/adversarial_diagnostics/oof_with_adversarial_scores.csv`
- `outputs/adversarial_diagnostics/oof_eval_like_overall.csv`
- `outputs/adversarial_diagnostics/oof_eval_like_fold.csv`
- `outputs/adversarial_model_compare/stable_vs_twohead_folds012_eval_like.csv`
- `outputs/submission_diff_diagnostics/swin80_vs_twohead_file_diff.csv`
- `outputs/submission_diff_diagnostics/swin80_vs_twohead_location_diff.csv`

### 提出 zip

- `outputs/submission_blends/submission_swin80_temporal20.zip`
- `outputs/submission_local_blends/submission_swin80_twohead_hg20.zip`
- `outputs/submission_local_blends/submission_swin80_twohead_hg15.zip`
- `outputs/submission_local_blends/submission_swin80_twohead_hg10.zip`
- `outputs/submission_local_blends/submission_swin80_twohead_hg05.zip`

### Downloads 側にある提出

- `/Users/shionsuio/Downloads/submission_swin_twohead_fold02_stable134.zip`
- `/Users/shionsuio/Downloads/submission.zip`
- `/Users/shionsuio/Downloads/submission_unet_v2.zip`
- `/Users/shionsuio/Downloads/submission_swin70_unet30.zip`
- `/Users/shionsuio/Downloads/submission_swin80_unet20.zip`

## 実装済み変更

以下は repo に入っている/入れ始めた。

- `src/build_adversarial_diagnostics.py`
- `src/run_oof_diagnostics.py` に `--use-two-head` 対応
- `src/swin_nowcast_v2.py` に sample weight 対応を追加中
- `src/run_swin_temporal_full.py` に sample weight 引数を追加中
- `kaggle_push/swin_oof_model_compare/`

直近の GitHub push:

```text
42c7344 add adversarial oof model comparison
```

注意: sample weight 対応は実装途中で、Kaggle dataset upload がユーザー側で中断された。再開前に `git diff` と `py_compile` を確認すること。

## 次にやること

### 1. adversarial weighted training

目的: train/eval 乖離を学習に反映する。

設計:

- `adversarial_scores.csv` の `weight_sqrt_clipped` を `unique_id` で train に結合
- Huber loss を sample weight 付きにする
- weight は強すぎると危険なのでまず `sqrt(density_ratio)` + clip `[0.5, 3.0]`
- まず fold0 or fold2 のみ

候補:

```text
model_subdir = swin_v2_temporal_adv_weighted
folds = 0 or 2
epochs = 8〜10
loss_type = huber
sample_weight_column = weight_sqrt_clipped
```

評価:

- 通常 RMSE
- eval-like weighted RMSE
- top30 RMSE
- Public 提出での反応

### 2. Meteosat 専用改善

理由:

- two-head は Meteosat を下げすぎて Public 悪化
- Meteosat image stats AUC は `0.816`
- train/eval の入力分布差が大きい

候補:

- Meteosat だけ base 維持/補正
- Meteosat 専用 normalization
- Meteosat の postprocess
- Meteosat を除外した two-head/local blend は既に効いた

### 3. 強雨 underprediction 対策

OOF 診断では強雨を潰している。

ただし単純 boost は Public で悪化済み。

過去に試したもの:

| postprocess | Public LB |
|---|---:|
| p95boost105 | `0.6654586023866743` |
| p95boost110 | `0.6658180834127551` |
| GOES+June p95boost105 | `0.6652116670003753` |

単純 boost は本線ではない。条件付きが必要。

## 2026-07-02 5案一斉実行の結果

「全部やろう」で開始した5ワークストリームの記録。

### A. OOF再キャリブレーション → 確定的ネガティブ

`src/fit_oof_recalibration.py`（fold2、isotonic/binned/affine/scale）。

- ランダムsplitのクロスフィットでは大幅改善に見える（twohead 1.2268→1.1606）が、**地点グループ分割にすると全手法悪化**（isotonic 1.2517、LOLO scale 1.3318）
- 地点別最適スケール: dhaka 0.12 / cape_town 0.46 / borno 1.17 / jamaica 1.17 / central_vietnam 2.27 — 残差バイアスは完全に地点固有
- **結論: 出力の大域的再キャリブレーションは未知地点（eval）に転移しない。提出しない。** 過去のp95boost系がLBで悪化した事実と整合。ランダムsplitでの見かけの改善は空間リーク

### B. IMERGターゲット構造EDA

- ターゲットは**0.01 mm/h刻みに完全量子化**（300 tifで偏差ゼロ確認）
- ゼロ率 ~80%（train全体82%）、tail: q99=6.06, q999=18.3, max~96.5
- RMSE指標では条件付き期待値が最適なので**グリッドスナッピングは無効**。ゼロ膨張はtwo-headのBCEヘッドが既に対応

### C. BTD物理特徴量（実装済み、GPU検証待ち）

- `band_mode="matched6_btd"`: matched6に BTD 3ch追加 — split-window(ch4-ch5)、WV-IR(ch2-ch4、対流overshoot)、WV差(ch2-ch3)
- 雲頂冷却率は `use_temporal_differences` が既にカバー
- 実装: swin_nowcast_v2.py（BTD_PAIRS, append_btd_channels, input_channel_count）+ run_swin_temporal_full.py `--band-mode`。ローカルsmoke test済み（63ch入力でforward OK、matched6は挙動不変）
- カーネル: `kaggle_push/swin_two_head_btd_fold2/`（band_stats_root=None必須 — 既存statsは6ch）。**GitHub push後に投入**
- 期待値は低〜中（BTDは線形結合なので理論上学習可能）。fold2 two-head val RMSEで直接比較

### D. pseudo-labeling — fold2スクリーニングで**ネガティブ確定**

- **結果: fold2 val RMSE 1.2382（ep8, weight 0.3）vs baseline 1.2268 → +0.0114悪化**
- train_huberは0.0372→0.0298に低下 = pseudoターゲット（自分自身の予測）にフィットしただけ。回帰self-trainingのconfirmation bias
- val_rmseはep7→8でまだ微減中だったが、best同士でも明確に劣後。weight 0.1等の再試行は優先度低（GPUコスト対比で見込み薄）
- 注: スクリーニング設計上、生成器はfold2単独モデル（弱い教師）。全ensembleならpseudo品質は上がるが、その場合fold2で正直に評価できない。**現時点では非採用**

- 運営がtransductive pseudo-labelingを**承認**（「pseudoはok出ました」）
- 実装（f7a437f）: `Config.pseudo_label_npz/csv/pseudo_sample_weight` + `load_pseudo_labels()` + train_foldフック。pseudo行はtrainのみに追加（validationは非汚染）、`is_eval`フラグでeval画像ディレクトリに切替、`pseudo_index`でnpzのターゲット参照
- **汚染回避設計**: fold2スクリーニングではpseudo生成器をfold2モデルのみ（twohead f2 + stable f2の50/50）に限定。他foldモデルを混ぜるとfold2の正解がリークする
- `src/build_pseudo_labels.py` でローカルMPS推論（29,090 eval、~60分）→ blend mean 0.265 / p95 1.38 / max 25.4
- Kaggle dataset: `solafune-pseudo-fold2`（npz 82MB + index CSV）
- カーネル `solafune-swin-pseudo-fold2` 投入済み（fold2 two-head + pseudo weight 0.3、8ep、~5.5h見込み）。**val RMSE 1.2268との比較で判定**

### E. two-head fold3/4 学習 → 完了、5-foldブレンドがCV最良

- fold3 val RMSE 0.8936（ep6）、fold4 0.8871（ep7）。checkpoint回収済み → `models/swin_v2_temporal_two_head_oof/`（5fold揃った）
- OOF export済み（fold3/4追加）。**5-fold pooled RMSE比較:**

| 構成 | pooled RMSE |
|---|---:|
| stable×5 | 1.076244 |
| twohead×5 | 1.072928 |
| 現行mixed（th f0,f2 + st f1,f3,f4）= LB 0.66750 | 1.068755 |
| **twohead50% + stable50%（10モデル）** | **1.063764** |

- weight掃引はw=0.5〜0.55でフラット。衛星別最適（hima 0.5 / goes 0.7 / meteo 0.4）は差が小さく過学習リスクがあるため一律0.5を採用
- 提出カーネル: `kaggle_push/swin_two_head_full_submit/`（出力: submission_twohead50_stable50_5x5.zip）
- Kaggle dataset: `swin-two-head-full-models`（two-head 5fold）+ `swin-temporal-stable-models`（stable 5fold、初アップロード）

### F. BTDアブレーション → ポジティブ（fold2 val −0.012）

- カーネル `solafune-swin-btd-fold2` 完了（スラッグは two-head-btd-fold2 だと "Notebook not found" になったためリネーム）
- **fold2 val RMSE 1.2149（ep4）vs two-head baseline 1.2268 → −0.0119改善**。ただしepoch間の変動大（1.21〜1.37）なので単fold結果は过信しない
- 出力: `kaggle_outputs/swin_btd_fold2/models/swin_v2_temporal_two_head_btd_fold2/`（checkpoint + 9ch band stats）
- 次の判断: pseudo結果と合わせて、有望なら全fold BTD再学習を検討（10モデル再学習のコストと相談）

### G. 5x5提出 → LB 0.66707（純モデル系で最良、CV-LB相関また的中）

- `submission_twohead50_stable50_5x5.zip` → **Public LB 0.6670709661506823**
- 純モデルブレンド系の比較: mixed（pooled 1.068755）= LB 0.66750 → 5x5（pooled 1.063764）= **LB 0.66707（−0.00043）**。pooled RMSE選別が3例連続で方向一致
- **ただし全体bestは依然 `submission_swin80_twohead_hg20.zip` = 0.66484**（swin80_temporal20ベース+two-head 20% HG限定）。5x5はベースブレンドなしの素の10モデル
- 次の一手候補: swin80_temporal20ベースに5x5系差分を混ぜる（`prediction_two_head_5fold.npy` / `prediction_stable_5fold.npy` 各196MBが `kaggle_outputs/swin_two_head_full_submit/` に回収済みで、Kaggle再実行なしでローカルブレンド可能）

### H. 5x5注入ブレンド（2026-07-03、`src/build_5x5_local_blends.py`）

- 4本生成（hg20/hg30/hg40/hg30_m10、`outputs/submission_local_blends/`）
- **hg30 = LB 0.66494、hg20 = 0.66497 → どちらも旧hg20（0.66484）に届かず**
- 同一重み比較（hg20同士）で**5x5成分は旧twohead成分に劣る**と確定。単体では5x5が上（0.66707 vs 0.66750）なのに注入では逆転 — ベースが既にstable系temporal成分を含むため、5x5内のstable半分が多様性を薄めていると解釈
- 5x5系はhg20→hg30で微改善（重みは30%側が正解）
- **th5f仮説は提出前に棄却**: zip代数検証で旧hg20注入成分 ≡ mixed zip（corr=1.0）と判明。純two-headではなくmixedが正解成分であり、注入のtwohead比率を上げるほど悪化する（base が既にstable系temporal成分を含むため）。`swin80_th5f_hg20/hg30.zip` は**提出禁止**
- `submission_pseudo_fold2.zip` → **LB 0.6728（大幅悪化）**。CV悪化（+0.0114）と方向一致の4例目。pseudo-labeling路線は完全クローズ

### I. 評価メトリクス変更（2026-07-03、LB全体再計算）

運営が評価関数を変更: **欠損チャネルのある衛星画像のサンプルを採点対象外に**。LB全提出が再計算された。

- 欠損はevalの**GOESのみ45ファイル → 影響29行/29,090行（0.1%）**。内訳: upper_midwest 16行、rio_grande_do_sul 11行、peru 2行。Himawari/Meteosatは全16ch完備（ローカルスキャンで確認、運営のdiscussionリストとも一致）
- 欠損パターン: 4ch(282×282)が28件、12〜15ch(141×141)が17件。tifからは「どのチャネルが欠けているか」は判別不可
- スコア変動は ~+0.0001（例: hg20 0.664844→0.664911）。**全提出の順位はほぼ保存され、既存の結論（mixed注入>5x5注入、pooled RMSE選別、HG単調改善）はすべて新メトリクスでも成立**
- 再計算の適用は7/2夜〜7/3朝。7/3以降の提出は最初から新メトリクスで採点
- 我々のパイプラインは `src.count < max(bands)` の画像をゼロ埋めしており（swin_nowcast_v2 `_read_observation`）、該当行が採点外になったのは中立〜微プラス。ただし**15ch GOESファイルはチェックを通過してチャネルずれのまま読まれる**（採点外なのでLB影響なし、trainにあればノイズ源）
- train側の欠損: **GOESのみ10ファイル**（15ch×6、14ch×2、4ch×2 / 30,778中）。Himawari/Meteosatは完備。15ch×6はチャネルずれのままtrainに入っているが10/30,778で無視できる規模。CV validation から除外して新LBメトリクスに揃える改修は優先度低

## 注意点

- Public は test 全体の 35%。Private は全 test。
- Public に寄せすぎると危険。
- ただし現時点では Public で一貫して two-head Meteosat が悪いので、Meteosat に two-head を入れるのは避ける。
- location は train/eval overlap 0。location 直接特徴（lat/lon raw）は危険。
- **ジオコーディングは2026-07-02に運営承認済み**（地点名→座標=特徴量変換。条件: フリー・再現可能ソース、文書化、EPSG:4326）。Nominatim実施済み → `data/location_coordinates_geocoded.csv` + `data/GEOCODING.md`。fold2スクリーニング2本（local_time/full）実行中。標高等の外部DEMルックアップは承認範囲外の可能性が高く、要追加確認。
- eval 入力を使う adversarial weighting / normalization は、外部データではないが transductive preprocessing に近い。ルール上の安全性は要確認。

## 再開時の最初のコマンド

```bash
cd /Users/shionsuio/solafune-workspace
git status --short
./.venv/bin/python -m py_compile src/swin_nowcast_v2.py src/run_swin_temporal_full.py src/build_adversarial_diagnostics.py
```

sample weight dataset を Kaggle に上げるなら:

```bash
mkdir -p kaggle_upload/solafune_adversarial_scores
cp outputs/adversarial_diagnostics/adversarial_scores.csv kaggle_upload/solafune_adversarial_scores/adversarial_scores.csv
```

`kaggle_upload/solafune_adversarial_scores/dataset-metadata.json` は作成済み。

