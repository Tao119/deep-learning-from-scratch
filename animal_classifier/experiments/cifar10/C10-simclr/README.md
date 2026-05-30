# C10 - SimCLR / CIFAR-10

## 概要

SimCLR（Simple Framework for Contrastive Learning of Visual Representations）による自己教師あり学習。ラベルなしデータで表現を学習し、少量のラベル（1%）で線形分類器を評価する。

## SimCLRの仕組み

1. 1枚の画像に異なるaugmentationを2回適用し、ポジティブペア（x_i, x_j）を作る
2. エンコーダ（backbone）＋プロジェクションヘッド（2層MLP）で埋め込みzを得る
3. NT-Xent lossで同一画像の2ビューを引き寄せ、他の画像を遠ざける

```
l(i, j) = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )
```

プリトレーニング後はプロジェクションヘッドを捨て、エンコーダ出力の上に線形層だけを重ねて評価する。

## SimCLR Augmentation Pipeline

| 変換 | 詳細 |
|------|------|
| Random crop | pad=4, reflect padding |
| Horizontal flip | 確率0.5 |
| Color jitter | brightness±0.4, contrast±0.4, saturation±0.2 |
| Gaussian blur | sigma∈[0.1, 1.0], 確率0.5 |

## アーキテクチャ

### エンコーダ (VGG-like, 小型)
- Conv-BN-ReLU × 4（64ch → 64ch → 128ch → 128ch）+ MaxPool × 2
- 出力: 128×8×8 = 8192次元の特徴マップをflatten

### プロジェクションヘッド
- Affine(8192→512) → ReLU → Affine(512→proj_dim)
- proj_dim=128（デフォルト）

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| Pretrain epochs | 20 |
| Temperature τ | 0.5 |
| Projection dim | 128 |
| Batch size | 256 |
| Optimizer | Adam (lr=0.001) |
| Label fraction | 1% (500 samples) |
| Linear eval epochs | 30 |

## 実行方法

```bash
cd <repo_root>
PYTHONPATH=. python3 animal_classifier/experiments/cifar10/C10-simclr/simclr.py \
  --pretrain_epochs 20 \
  --eval_epochs 30 \
  --batch 256 \
  --temperature 0.5 \
  --label_fraction 0.01
```

学習済みエンコーダが存在する場合はプリトレーニングをスキップ:

```bash
PYTHONPATH=. python3 .../simclr.py --skip_pretrain
```

## 評価方法

プリトレーニング後:
1. エンコーダを凍結し、表現の上に線形分類器を学習（1%ラベル）
2. 同じ1%ラベルでVGGWithBNをゼロから教師あり学習（ベースライン）
3. 両者の精度を比較し、自己教師あり表現の品質を評価

## 期待される結果

| 手法 | ラベル率 | 精度 |
|------|---------|------|
| SimCLR linear eval | 1% | 実行後に記入 |
| 教師あり baseline | 1% | 実行後に記入 |
| VGGWithBN (C02, 全ラベル) | 100% | 90.58% |

自己教師あり事前学習により、少量ラベルで教師あり学習を上回ることを期待する。
