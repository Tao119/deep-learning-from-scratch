# Deep Learning from Scratch — 画像認識実験プロジェクト

純 NumPy で実装したディープラーニングフレームワーク + CIFAR-10 / Oxford Pets 実験。  
外部 DL ライブラリ一切不使用（NumPy + Matplotlib のみ）。

## 構成

```
deep-learning-from-scratch/
├── common/              # 純NumPy DLフレームワーク
│   ├── layers.py        # Affine, BN, Dropout, Softmax等
│   ├── conv_layers.py   # Conv2D, Pooling (im2col/col2im)
│   ├── optimizer.py     # SGD, Adam, AdaGrad
│   └── layers_ext.py    # 4DテンソルBN
│
├── animal_classifier/   # 実験プロジェクト
│   ├── models/          # VGGLike, VGGWithBN, ResNet, SEResNet, DenseNet, MobileNet
│   ├── dataset/         # CIFAR-10, Oxford Pets, 拡張手法
│   ├── utils/           # Grad-CAM, Ensemble, 可視化
│   └── experiments/
│       └── cifar10/     # C01〜C17 実験フォルダ
│
├── ch01〜ch08/          # 教科書章別実装
└── REPORT.md            # 実験レポート
```

## 実験結果サマリ (CIFAR-10)

| 実験 | モデル | Top-1 | Top-5 |
|------|--------|-------|-------|
| C01 | VGGLike | 87.75% | 99.50% |
| **C02** | **VGGWithBN** | **90.58%** | **99.70%** |
| C04 | VGGWithBN (noaug) | 87.37% | 99.31% |
| C05 | MobileNet | 85.40% | 99.37% |
| **C08** | **VGGWithBN + Mixup** | **90.25%** (peak **91.9%**) | **99.65%** |
| C09 | Knowledge Distillation | KD=84.9% > Scratch=83.3% | — |
| C12 | WarmupCosine (best sched) | 42.97% (15ep) | — |

詳細: [REPORT.md](REPORT.md)

## 実行方法

```bash
# CIFAR-10 ダウンロード + キャッシュ
cd animal_classifier && PYTHONPATH=.. python3 -c "from dataset.cifar10 import load_cifar10; load_cifar10()"

# VGGWithBN 学習 (C02 相当)
PYTHONPATH=.. python3 train_cifar10.py --model vgg_bn --epochs 30 --batch 256 --aug mild

# Mixup 学習 (C08 相当)
PYTHONPATH=.. python3 train_cifar10_mixup.py --mix mixup --epochs 30 --batch 256

# 知識蒸留 (C09)
PYTHONPATH=.. python3 experiments/cifar10/C09-knowledge-distillation/distill.py

# LR スケジュール比較 (C12)
PYTHONPATH=.. python3 experiments/cifar10/C12-lr-schedule/compare_schedules.py
```

## キー技術

- **Batch Normalization**: 学習安定化 + 精度 +2.8%
- **Mixup**: ソフトラベル正則化。LR decay 後に効果が顕在化
- **Knowledge Distillation**: 小モデルに大モデルの知識転移 (+1.57%)
- **WarmupCosine LR**: フル拡張環境で最良の収束
- **Depthwise Separable Conv**: MobileNet で 1/5 パラメータで 85.4%
- **Grad-CAM**: モデルが注目している領域の可視化

## 教科書実装 (ch01〜ch08)

| 章 | 内容 |
|----|------|
| ch01 | NumPy基礎・パーセプトロン |
| ch02 | 論理ゲート |
| ch03 | ニューラルネットワーク推論 |
| ch04 | ニューラルネットワーク学習 |
| ch05 | 誤差逆伝播法 |
| ch06 | 最適化・正規化技術 |
| ch07 | CNN (畳み込みNN) |
| ch08 | DeepConvNet + 実装まとめ |
