# C12 - LR Schedule Comparison / CIFAR-10

## 概要

同じVGGWithBNモデルを3種類の学習率スケジュールで訓練し、収束速度と最終精度を比較する。各スケジュール15エポック。

## スケジュール

### 1. StepLR

エポック10で学習率を1/10に下げる。実装が単純で広く使われる。

```
lr(t) = lr_max     (t < 10)
      = lr_max * 0.1  (t >= 10)
```

### 2. CosineAnnealing

コサイン曲線に沿ってlr_maxからlr_minまで滑らかに減衰させる。バッチサイズ・エポック数に依存しにくく、シャープなミニマムを避ける効果がある。

```
lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
```

### 3. WarmupCosine

最初のN_warmupエポックで線形ウォームアップしてからコサイン減衰に入る。Transformerやバッチサイズが大きい設定で安定化に寄与することが知られている。

```
lr(t) = lr_max * (t+1) / N_warmup          (t < N_warmup)
      = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(t-N_warmup)/(T-N_warmup)))
                                            (t >= N_warmup)
```

## 構成

- モデル: VGGWithBN（32×32入力、10クラス）
- lr_max: 1e-3 / lr_min: 1e-5
- Warmup: 3 epochs
- Augmentation: mild（flip + crop pad=2）
- Optimizer: Adam（スケジュールに従いlrを毎エポック更新）
- Epochs: 15 / Batch: 128

## 実行方法

```bash
cd <repo_root>
python animal_classifier/experiments/cifar10/C12-lr-schedule/compare_schedules.py
```

出力: `lr_schedules.png`（LR曲線・損失・train精度・test精度の2×2グリッド）

## 結果

| スケジュール | Final Test Acc | Best Test Acc |
|-------------|---------------|--------------|
| StepLR | 実行後に記入 | 実行後に記入 |
| CosineAnnealing | 実行後に記入 | 実行後に記入 |
| WarmupCosine | 実行後に記入 | 実行後に記入 |
