# Oxford-IIIT Pet 分類実験一覧

純NumPy実装のCNNでOxford-IIIT Pet Dataset（37犬猫種）の分類精度を段階的に改善する実験群。

## データセット

- Oxford-IIIT Pet Dataset（37犬猫種）
- Train: 3680枚 / Test: 3669枚
- 実験によりクラス数・解像度・サンプル数を変更

## 実験一覧

| ID | フォルダ | モデル | クラス | 解像度 | Aug | Top-1 | Top-5 | 状態 |
|----|---------|--------|--------|--------|-----|-------|-------|------|
| 01 | vgglike-baseline | VGGLike | 37 | 32px | full | 3.87% | 17.68% | 完了 |
| 02 | vggbn-batchnorm | VGGWithBN | 37 | 32px | full | 4.03% | 20.67% | 完了 |
| 03 | resnet-skipconn | ResNet | 37 | 32px | full | 5.07% | 18.97% | 完了 |
| 04 | vggnoaug-overfit | VGGLike | 37 | 32px | none | **15.62%** | **48.36%** | 完了 |
| 05 | vggbn-catdog-binary | VGGWithBN | 2 | 64px | mild | — | — | 予定 |
| 06 | vggbn-top10-64px | VGGWithBN | 10 | 64px | mild | — | — | 実行中 |
| 07 | resnet-top10-64px | ResNet | 10 | 64px | mild | — | — | 予定 |

## 主要な発見

### 実験01〜04（32px, 37クラス）

- **Augmentationが逆効果**：32×32という低解像度では flip/crop/color jitter が情報を破壊し過学習より精度が下がる
- **BNの効果は限定的**：37クラスでは解像度不足がボトルネック
- **ResNetは最も遅い**：pure NumPy + im2col では 1epoch ≈ 2分（VGGの3倍）

### 実験05〜07（64px, 削減クラス）の狙い

1. 解像度を64×64に倍増して細粒度特徴を学習できる余地を作る
2. クラス数を削減してクラスあたりのサンプル数を確保
3. augmentationを軽量化（flip + 小crop のみ）

## ファイル構成

```
experiments/
  01-vgglike-baseline/    # 完了
  02-vggbn-batchnorm/     # 完了
  03-resnet-skipconn/     # 完了
  04-vggnoaug-overfit/    # 完了
  05-vggbn-catdog-binary/ # 予定
  06-vggbn-top10-64px/    # 実行中
  07-resnet-top10-64px/   # 予定
  comparison.png          # 01〜04比較グラフ
```
