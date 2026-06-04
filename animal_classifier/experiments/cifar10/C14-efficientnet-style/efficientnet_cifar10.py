"""
C14: EfficientNet-style Compound Scaling / CIFAR-10
Pure NumPy — consistent with the project's from-scratch style.

EfficientNet idea (Tan & Le, 2019)
────────────────────────────────────
Rather than scaling only width, depth, or resolution independently,
compound scaling co-scales all three dimensions with a fixed ratio:

    width_coeff   φ^α    (α = 1.2)
    depth_coeff   φ^β    (β = 1.1)
    resolution            resolution=32 (CIFAR fixed)

Base model (φ=1) ≈ MobileNet-style DS-Conv stack.
We apply φ=1.0 so the scaled model has:
    width_coeff  = 1.2^1 = 1.2
    depth_coeff  = 1.1^1 = 1.1  → ceil(n_blocks * 1.1) per stage

This experiment demonstrates the compound scaling concept in pure NumPy,
using the same DepthwiseConvolution + BatchNormalization infrastructure
already present in the project's MobileNet (C05).

Architecture (scaled)
─────────────────────
Base channels:  [32, 64, 128, 128, 256]
Scaled (×1.2):  [38, 77, 154, 154, 307] → rounded to even: [38,78,154,154,308]
Base depths:    [ 1,  1,   2,   2,   2] blocks per stage
Scaled (×1.1): ceil→ [ 2,  2,   3,   3,   3]

Stem: Conv(3→38, 3×3, s=1, p=1) → BN → ReLU
Stage1: DS(38→78,   s=1) × 2
Stage2: DS(78→154,  s=2) × 2 → DS(154→154, s=1) × 1
Stage3: DS(154→308, s=2) × 2 → DS(308→308, s=1) × 1
GlobalAvgPool → FC(308→10)

Training: Adam, cosine LR, 30 epochs, batch=128, mild augmentation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import math
import time
import pickle
import numpy as np

from animal_classifier.dataset.cifar10 import load_cifar10, normalize
from animal_classifier.dataset.augmentation import batch_mild_augment
from animal_classifier.models.mobilenet import DepthwiseConvolution, DSConvBlock
from common.conv_layers import Convolution
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization
from common.optimizer import Adam


# ─────────────────────────────────────────────────────────────────────────────
#  Compound Scaling Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Base model configuration
BASE_CHANNELS = [32, 64, 128, 128, 256]
BASE_DEPTHS   = [1,   1,   2,   2,   2]   # DS-blocks per stage

# Compound scaling coefficients (φ = 1.0)
PHI           = 1.0
WIDTH_ALPHA   = 1.2
DEPTH_BETA    = 1.1

WIDTH_COEFF   = WIDTH_ALPHA ** PHI    # 1.2
DEPTH_COEFF   = DEPTH_BETA  ** PHI    # 1.1


def scale_channels(base: int, coeff: float = WIDTH_COEFF) -> int:
    """Round to nearest even for hardware efficiency."""
    scaled = base * coeff
    return max(2, 2 * round(scaled / 2))


def scale_depth(base: int, coeff: float = DEPTH_COEFF) -> int:
    return max(1, math.ceil(base * coeff))


SCALED_CHANNELS = [scale_channels(c) for c in BASE_CHANNELS]
SCALED_DEPTHS   = [scale_depth(d)   for d in BASE_DEPTHS]

# ─────────────────────────────────────────────────────────────────────────────
#  EfficientNet-style Model
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS      = 30
BATCH_SIZE  = 128
LR          = 1e-3
LR_MIN      = 1e-5
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))


class EfficientNetStyleCIFAR:
    """
    Compound-scaled MobileNet-style network for CIFAR-10.

    Demonstrates the EfficientNet compound scaling concept:
    - Width and depth co-scaled with fixed coefficients α, β
    - Same DSConvBlock building block as C05-MobileNet

    Input : (N, 3, 32, 32)
    Output: (N, 10) logits
    """

    def __init__(self, input_channels: int = 3, output_size: int = 10):
        sc = SCALED_CHANNELS     # [38, 78, 154, 154, 308]
        sd = SCALED_DEPTHS       # [2, 2, 3, 3, 3]

        print(f"  Compound Scaling:")
        print(f"    width_coeff  = {WIDTH_COEFF:.2f}  (α={WIDTH_ALPHA}, φ={PHI})")
        print(f"    depth_coeff  = {DEPTH_COEFF:.2f}  (β={DEPTH_BETA}, φ={PHI})")
        print(f"    base channels: {BASE_CHANNELS}")
        print(f"    scaled channels: {sc}")
        print(f"    base depths: {BASE_DEPTHS}")
        print(f"    scaled depths: {sd}")

        # Stem
        self.stem = Convolution(
            np.sqrt(2.0 / (input_channels * 9)) * np.random.randn(sc[0], input_channels, 3, 3),
            np.zeros(sc[0]), stride=1, pad=1
        )
        self.stem_bn  = BatchNormalization(np.ones(sc[0]), np.zeros(sc[0]))
        self.stem_relu = Relu()

        # Stage 1: DS(sc[0]→sc[1]) × sd[0], then DS(sc[1]→sc[1]) × (sd[1]-1)
        self.stage1 = []
        self.stage1.append(DSConvBlock(sc[0], sc[1], stride=1))
        for _ in range(sd[0] - 1):
            self.stage1.append(DSConvBlock(sc[1], sc[1], stride=1))

        # Stage 2: DS(sc[1]→sc[2], s=2) then DS(sc[2]→sc[2]) × (sd[2]-1)
        self.stage2 = []
        self.stage2.append(DSConvBlock(sc[1], sc[2], stride=2))
        for _ in range(sd[1] - 1):
            self.stage2.append(DSConvBlock(sc[2], sc[2], stride=1))

        # Stage 3: DS(sc[2]→sc[3], s=1) then DS(sc[3]→sc[4], s=2) then repeat
        self.stage3 = []
        self.stage3.append(DSConvBlock(sc[2], sc[3], stride=1))
        self.stage3.append(DSConvBlock(sc[3], sc[4], stride=2))
        for _ in range(sd[2] - 1):
            self.stage3.append(DSConvBlock(sc[4], sc[4], stride=1))

        # Classification head
        self.fc = Affine(
            np.sqrt(2.0 / sc[4]) * np.random.randn(sc[4], output_size),
            np.zeros(output_size)
        )
        self.last_layer = SoftmaxWithLoss()

        self._all_blocks = self.stage1 + self.stage2 + self.stage3

        n_params = self._count_params()
        print(f"    total DS-blocks: {len(self._all_blocks)}")
        print(f"    (approx params counted via weight shapes)")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, x, train_flg=False):
        x = self.stem_relu.forward(
            self.stem_bn.forward(self.stem.forward(x), train_flg)
        )
        for block in self._all_blocks:
            x = block.forward(x, train_flg)
        # Global Average Pooling
        x = x.mean(axis=(2, 3))   # (N, C)
        return self.fc.forward(x)

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=128):
        t_lbl = t if t.ndim == 1 else np.argmax(t, axis=1)
        n = (x.shape[0] // batch_size) * batch_size
        correct = 0
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i+batch_size], train_flg=False)
            correct += np.sum(np.argmax(y, axis=1) == t_lbl[i:i+batch_size])
        return correct / n if n > 0 else 0.0

    # ── Backward ─────────────────────────────────────────────────────────────

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        dout = self.fc.backward(dout)

        # Expand GAP gradient back to spatial
        last_block = self._all_blocks[-1]
        H, W = last_block.pw.x.shape[2:]
        dout = dout[:, :, None, None] * np.ones((1, 1, H, W)) / (H * W)

        for block in reversed(self._all_blocks):
            dout = block.backward(dout)

        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        self.stem.backward(dout)
        return {}

    # ── Parameter collection ─────────────────────────────────────────────────

    def get_params_and_grads(self):
        params, grads = {}, {}
        idx = [0]

        def reg_conv(layer):
            k = idx[0]
            params[f"p{k}_W"] = layer.W
            params[f"p{k}_b"] = layer.b
            grads[f"p{k}_W"] = layer.dW
            grads[f"p{k}_b"] = layer.db
            idx[0] += 1

        def reg_bn(layer):
            k = idx[0]
            params[f"p{k}_g"]  = layer.gamma
            params[f"p{k}_bt"] = layer.beta
            grads[f"p{k}_g"]   = layer.dgamma
            grads[f"p{k}_bt"]  = layer.dbeta
            idx[0] += 1

        reg_conv(self.stem)
        reg_bn(self.stem_bn)

        for block in self._all_blocks:
            reg_conv(block.dw)
            reg_bn(block.dw_bn)
            reg_conv(block.pw)
            reg_bn(block.pw_bn)

        reg_conv(self.fc)
        return params, grads

    def update(self, optimizer):
        params, grads = self.get_params_and_grads()
        optimizer.update(params, grads)

    def save(self, path):
        params, _ = self.get_params_and_grads()
        try:
            with open(path, "wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            print(f"  [warn] save failed: {e}")

    def _count_params(self):
        params, _ = self.get_params_and_grads()
        return sum(v.size for v in params.values())


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(epoch, T=EPOCHS, lr_max=LR, lr_min=LR_MIN):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))


def _accuracy_quick(model, x, t, n=500):
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return model.accuracy(x[idx], t[idx])


def main():
    log_path  = os.path.join(OUTPUT_DIR, "train.log")
    best_path = os.path.join(OUTPUT_DIR, "EfficientNetStyle_cifar10_best.pkl")

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    print("\nBuilding EfficientNet-style model ...")
    model     = EfficientNetStyleCIFAR()
    optimizer = Adam(lr=cosine_lr(0))

    train_size     = len(x_train)
    iter_per_epoch = max(train_size // BATCH_SIZE, 1)
    best_test_acc  = 0.0

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"C14 - EfficientNet-style Compound Scaling / CIFAR-10")
    log(f"width_coeff={WIDTH_COEFF:.2f} (α={WIDTH_ALPHA}), "
        f"depth_coeff={DEPTH_COEFF:.2f} (β={DEPTH_BETA}), φ={PHI}")
    log(f"channels: {SCALED_CHANNELS}   depths: {SCALED_DEPTHS}")
    log(f"epochs={EPOCHS}, batch={BATCH_SIZE}, lr_max={LR}, lr_min={LR_MIN}")
    log("")

    for epoch in range(EPOCHS):
        lr = cosine_lr(epoch)
        optimizer.lr = lr

        epoch_loss = 0.0
        t0   = time.time()
        perm = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            bi   = perm[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x_b  = batch_mild_augment(x_train[bi], train_flg=True)
            t_b  = t_train[bi]
            model.gradient(x_b, t_b)
            model.update(optimizer)
            epoch_loss += model.last_layer.loss

        epoch_loss /= iter_per_epoch
        tr_acc  = _accuracy_quick(model, x_train, t_train)
        te_acc  = _accuracy_quick(model, x_test,  t_test)
        elapsed = time.time() - t0

        msg = (f"epoch {epoch+1:3d}/{EPOCHS}: "
               f"loss={epoch_loss:.4f}  "
               f"train={tr_acc:.4f}  "
               f"test={te_acc:.4f}  "
               f"({elapsed:.0f}s)")
        log(msg)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            model.save(best_path)

    log(f"\nBest test accuracy: {best_test_acc:.4f}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nLog saved: {log_path}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
