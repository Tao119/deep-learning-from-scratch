# C11 - Label Smoothing / CIFAR-10

## 概要

C02（VGGWithBN + standard CE、90.58%）に対し、ラベルスムージングを適用した場合の精度変化を検証する。

ラベルスムージングは、one-hotターゲットを少しだけ均等分布に近づける正則化手法。

```
y_smooth = (1 - eps) * one_hot + eps / K
```

損失はKLダイバージェンスとして計算する。

```
L = -sum_k( y_smooth_k * log(p_k) )
  = (1-eps)*CE + eps * H_uniform
```

## 構成

- モデル: VGGWithBN（C02と同じアーキテクチャ）
- 入力サイズ: 32×32 RGB
- クラス数: 10
- eps: 0.1
- Augmentation: mild（flip + crop pad=2）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 20 → 1e-4, epoch 25 → 1e-5
- Epochs: 30 / Batch: 128

## 正則化効果

通常の学習では、正解クラスのlogitを無限大に押し上げることが最適になるため、モデルが過信（overconfidence）しやすい。ラベルスムージングはターゲットを0/1から離すことで、この過信を構造的に抑制する。

結果として：

- 過学習が緩和され、汎化精度が向上することが多い
- 予測確率の分散が小さくなる（キャリブレーションの改善）
- 知識蒸留の教師モデルとしての品質が高まる（Hinton et al.）

## 実行方法

```bash
cd <repo_root>
python animal_classifier/experiments/cifar10/C11-label-smoothing/train_smooth.py
```

## 結果

| 実験 | 損失関数 | Best Test Acc |
|------|---------|--------------|
| C02  | Standard CE | 90.58% |
| C11  | Label Smoothing (eps=0.1) | 実行後に記入 |
