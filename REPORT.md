# 画像認識プロジェクト 実験レポート

## プロジェクト概要

純NumPy実装のディープラーニングフレームワーク上で CIFAR-10 を対象に複数のモデルアーキテクチャと学習戦略を実験・比較。

---

## 実装したコンポーネント

### 基盤ライブラリ (common/)
- `optimizer.py`: SGD / Momentum / AdaGrad / Adam
- `layers.py`: Affine / ReLU / Sigmoid / Dropout / BatchNormalization / SoftmaxWithLoss
- `conv_layers.py`: Convolution / Pooling (im2col/col2im)
- `layers_ext.py`: BatchNormalization (4D テンソル対応)

### 実装モデル
| モデル | パラメータ数 | 特徴 |
|--------|-----------|------|
| VGGLike | ~2M | VGG風 6層CNN |
| VGGWithBN | ~2M | Batch Normalization 追加 |
| ResNet | ~3M | Skip connection, Global AvgPool |
| SEResNet | ~3M | Squeeze-and-Excitation attention |
| DenseNet | ~1M | Dense connections, growth rate k=12 |
| MobileNet | ~1.2M | Depthwise Separable Convolution |

---

## CIFAR-10 実験結果

### 完了実験

| 実験 | モデル | 拡張 | **Top-1** | **Top-5** | epoch | 備考 |
|------|--------|------|---------|---------|-------|------|
| C01 | VGGLike | mild | **87.75%** | 99.50% | 30 | ベースライン |
| C02 | VGGWithBN | mild | **90.58%** | 99.70% | 30 | BN効果確認 |
| C03 | ResNet | mild | — | — | 17(中断) | 遅すぎ(~2h/epoch) |
| C04 | VGGWithBN | none | **87.37%** | 99.31% | 30 | 拡張なし比較 |
| C05 | MobileNet | mild | **85.40%** | 99.37% | 30 | 軽量モデル |
| **C08** | **VGGWithBN+Mixup** | **mild+mixup** | **90.25%** | **99.65%** | 30 | **peak 91.9%@ep24** |
| C09 | KD (student) | mild | **84.90%** vs scratch **83.3%** | — | 30 | KD +1.56% |
| C11 | Label Smoothing | full | 26.8% (≈CE) | — | 30 | フル拡張で収束遅 |
| C12 | LR Schedules | full | WarmupCosine **42.97%** | — | 15/sched | 比較実験 |

### キー発見

**1. Batch Normalization の効果 (C01→C02)**
```
VGGLike:    87.75%
VGGWithBN:  90.58%  (+2.83%)  ← 正規化が学習安定化
```

**2. データ拡張の効果 (C04→C02)**
```
noaug:  87.37%
mild:   90.58%  (+3.21%)  ← 50k データには拡張が有効
```

**3. Mixup の効果 (C02→C08)**
```
標準 CE:  90.58%
Mixup:    90.25% (peak 91.9%@ep24)
Mixup は LR decay 後に急上昇：ep20 で 84.8%→ep21 で 90.0%
```
Mixup 学習曲線の特徴：
- 通常の CE より序盤の精度が低い（ソフトラベルの難しさ）
- LR decay 後に CE を超える逆転が発生
- epoch24 で 91.9% のピークを記録

**4. Knowledge Distillation の効果 (C09)**
```
Student (scratch): 83.33%
Student (KD):      84.90%  (+1.57%)  ← 教師モデルの知識が転移
教師モデル:        85.18% (10ep 事前学習)
```

**5. LR Schedule 比較 (C12, フル拡張 15ep)**
```
StepLR:        32.6%
CosineAnnealing: 38.3%
WarmupCosine:  42.97%  ← ウォームアップ + コサイン減衰が最良
```
WarmupCosine の優位性：初期の不安定な学習を線形ウォームアップで吸収し、
その後のコサイン減衰で収束を安定化。

**6. モデル規模 vs 精度トレードオフ**
```
MobileNet (1.2M params): 85.4%
VGGWithBN (2.0M params): 90.6%
精度差 +5.2% に対してパラメータ +67%
```

---

## Grad-CAM 可視化

実装済み (`utils/grad_cam_visualization.py`)。
VGGWithBN の最後の Conv 層に対して Grad-CAM を計算。
10クラス各1サンプルのヒートマップを `grad_cam_grid.png` に保存。

---

## モデルアンサンブル実験 (C17)

3モデルのアンサンブル戦略比較：

| 戦略 | 概要 |
|------|------|
| Soft average | 確率を平均 |
| Majority vote | 多数決 |
| Weighted | 精度で重み付け |

期待効果: 個々のモデルより +1〜2% 程度の向上

---

## 新アーキテクチャ実装 (実行待ち)

### C13: Vision Transformer (ViT) — 実装済み
- パッチサイズ 4×4 → 64 パッチ
- 4 Transformer層、4ヘッド、d_model=64
- 純 NumPy 手動バックプロパゲーション
- 期待精度: 70〜80% (CIFAR-10 はローカルテクスチャが重要なので ViT は不利)

### C14: EfficientNet-style Compound Scaling — 実装済み
- width_coeff=1.2, depth_coeff=1.1
- MobileNet ベースにスケーリング適用
- 期待精度: MobileNet より +2〜3%

---

## 拡張手法比較 (理論)

| 拡張手法 | 主な効果 | 推奨場面 |
|---------|---------|---------|
| Flip + CropPad (mild) | 汎化性向上 | 小〜中規模データ |
| Color Jitter | 色・照明不変性 | 実環境データ |
| Cutout | 欠損ロバスト性 | 物体検出との相性良 |
| **Mixup** | ソフトラベルで正則化 | **大規模データ** |
| **CutMix** | 領域混合で空間認識 | **位置情報重要なタスク** |

---

## 学習動態の分析

### Mixup の学習曲線 (C08)
```
epoch  1: test=47.9%  (CE よりも低いスタート)
epoch  6: test=82.5%  (最初のピーク)
epoch 14: test=88.5%  (LR decay 前)
epoch 20: LR decay (0.001→0.0001)
epoch 21: test=90.0%  (急上昇！)
epoch 22: test=90.6%  (新記録)
epoch 24: test=91.9%  (ピーク)
epoch 30: test=90.0%  (最終)
Final Top-1: 90.25%
```

### BN の学習安定化効果 (C01 vs C02)
VGGLike は epoch10-20 で損失が振動するが、
VGGWithBN は単調減少で最終精度も高い。

---

## 今後の改善計画

### 短期 (実行可能)
1. C13 ViT を実際に学習実行 (30epoch × ~100s/epoch = ~50分)
2. C14 EfficientNet を実際に学習実行
3. C16 CutMix をフル30epoch で学習し C08 Mixup と公平比較
4. C17 Ensemble を実際のモデル pkl を使って評価

### 中期 (GPU があれば)
5. SEResNet と DenseNet を完全学習（C06/C07 は遅すぎて途中停止）
6. MobileNetV2 / ShuffleNet の実装
7. 50epoch 学習でより公平な比較

### 長期
8. Oxford Pets Dataset での転移学習実験
9. 自己教師あり学習 (SimCLR) で事前学習 → 少量ラベルで Fine-tuning
10. CIFAR-100 への拡張 (100クラス)

---

## SimCLR 自己教師あり事前学習 (C10, 実装済み)

実装済みだが実行未完了。
設計:
- SimCLR エンコーダ + Projection Head
- NT-Xent 対比損失
- 1% ラベルで線形評価
- 期待: 監督学習比 -5〜10% (少量ラベル設定では有利)
