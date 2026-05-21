# 06 - VGGWithBN / Top10クラス / 64px / Mild Augmentation

## 概要

解像度倍増・クラス削減・augmentation軽量化の3点改善を同時投入。

## 構成

- モデル: VGGWithBN
- 入力サイズ: **64×64**（32×32の4倍の情報量）
- クラス数: **10**（全37→上位10犬猫種）
- Augmentation: **mild**（flip + 小crop pad=2 のみ、color jitter/cutout なし）
- Optimizer: Adam (lr=0.001)
- LR decay: epoch 30 → 1e-4, epoch 40 → 1e-5
- Epochs: 50
- Batch: 32
- Train/Test: 1000/1000サンプル

## 対象クラス（top10）

Abyssinian, american bulldog, american pit bull terrier, basset hound, beagle,
Bengal, Birman, boxer, British Shorthair, chihuahua

## 改善の狙い

| 問題 | 対策 |
|------|------|
| 32×32では細粒度特徴が失われる | 64×64に解像度倍増 |
| 37クラスはデータ少なすぎ（~100枚/class） | top10で~100枚→絞り込み |
| color jitterが有害（実験04で判明） | mild aug（flip+小crop）のみ |

## 結果

| 指標 | 値 |
|------|----|
| Best test acc (sampling) | 13.54% |
| Final Top-1 (full test) | 9.27% |
| Final Top-5 (full test) | 54.54% |

## 所見

Top-1が10クラスランダム精度（10%）とほぼ同等で期待を下回る。

考察：
- 100枚/クラスはVGGWithBNが安定収束するには少なすぎる
- BNのrunning statsが少ないデータで不安定になった可能性
- Top-5=54.54%はランダム(50%)より改善されており、学習は起きている
- → **実験05（catdog 2クラス/3680枚全データ）でデータ量の重要性を検証**
