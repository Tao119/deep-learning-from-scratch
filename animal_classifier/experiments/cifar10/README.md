# CIFAR-10 Classification Experiments

## データセット

**CIFAR-10**（カナダ・トロント大学）

| 項目 | 値 |
|------|-----|
| Train | 50,000枚（5,000/class） |
| Test | 10,000枚（1,000/class） |
| 解像度 | 32×32 RGB |
| クラス数 | 10 |
| クラス | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

Oxford Petsと比べてクラスあたり50倍のデータ量。

## 目的

- データ量がモデル精度に与える影響の定量化（Pets top10: 100枚/class → CIFAR-10: 5000枚/class）
- 同じ32×32解像度・同じアーキテクチャで公平比較
- CIFAR-10ベンチマーク精度との比較（VGG系: ~85-90%, ResNet: ~90-93%）

## 実験一覧

| ID | フォルダ | モデル | Aug | Top-1 | Top-5 | 状態 |
|----|---------|--------|-----|-------|-------|------|
| C01 | vgglike-baseline | VGGLike | mild | 87.75% | 99.50% | ✅ 完了 |
| C02 | vggbn-batchnorm | VGGWithBN | mild | **90.58%** | 99.70% | ✅ 完了 |
| C03 | resnet-skipconn | ResNet | mild | ~88%† | — | ⚠️ 17ep中止 |
| C04 | vggbn-noaug | VGGWithBN | none | 87.37% | 99.31% | ✅ 完了 |

† epoch 14時点のsampling acc。pure NumPy ResNetは1epoch≈2時間のため中止。

## 共通設定

- Optimizer: Adam (lr=0.001)
- Batch size: 128（Petsの32から拡大、50kデータに適切）
- LR decay: epoch 30 → 1e-4, epoch 40 → 1e-5
- Epochs: 50
- Mild aug: horizontal flip + crop(pad=2)
