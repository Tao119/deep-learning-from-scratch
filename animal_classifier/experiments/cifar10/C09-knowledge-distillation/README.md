# C09 - Knowledge Distillation / CIFAR-10

## 概要

大きな教師モデル（VGGWithBN, C02: 90.58%）から、小さな生徒モデル（StudentNet, 3-Conv + 2-FC）へ知識を転移する。

## Knowledge Distillationの仕組み

通常のクロスエントロピーはone-hotラベルしか使わないが、KDでは教師の出力する確率分布（soft labels）を学習ターゲットに加える。

```
KD loss = alpha * KL(student_soft || teacher_soft) + (1 - alpha) * CE(student, hard_label)

soft_i = softmax(logits_i / T)
```

温度パラメータTを大きくすると、教師の確率分布がより平滑化され、クラス間の類似関係の情報が強調される。

## アーキテクチャ

### 教師モデル (VGGWithBN)
- Conv-BN-ReLU × 6 + FC × 2
- パラメータ数: ~5M
- CIFAR-10 精度: 90.58% (C02)

### 生徒モデル (StudentNet)
- Conv-ReLU × 3（32ch → 64ch → 128ch）+ MaxPool × 3 + FC × 2（256次元）+ Dropout(0.4)
- パラメータ数: ~1.5M（教師の約30%）

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| Temperature T | 4.0 |
| alpha (KD weight) | 0.7 |
| Epochs | 30 |
| Batch size | 256 |
| Optimizer | Adam (lr=0.001) |
| LR decay | epoch 20, 25 → ×0.1 |
| Augmentation | mild (flip + crop) |

## 実行方法

```bash
cd <repo_root>
PYTHONPATH=. python3 animal_classifier/experiments/cifar10/C09-knowledge-distillation/distill.py \
  --epochs 30 --batch 256 --alpha 0.7 --T 4.0
```

教師のpklが存在しない場合、スクリプトが自動的に教師を10エポック訓練する。

## 期待される結果

生徒モデルをゼロから学習した場合と比較して、KDにより精度が向上することを確認する。

| モデル | 精度 |
|--------|------|
| 教師 (VGGWithBN C02) | 90.58% |
| 生徒 (KD) | 実行後に記入 |
| 生徒 (scratch) | 実行後に記入 |
| KD改善幅 | 実行後に記入 |
