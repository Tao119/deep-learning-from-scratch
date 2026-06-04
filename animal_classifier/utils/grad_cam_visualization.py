"""
Grad-CAM Visualization for VGGWithBN on CIFAR-10
=================================================
Complete Grad-CAM pipeline:

1. Load VGGWithBN (C02 pkl) — falls back to random weights if pkl absent.
2. Hook the last conv layer activation (Relu6, index 19 in layer list).
3. Compute:
     weights_k = global_avg_pool(d_loss / d_feature_map_k)
     CAM = ReLU(sum_k  weights_k * feature_map_k)
4. Resize CAM to 32×32 (nearest-neighbour upsampling).
5. Overlay as jet-colourmap heatmap on the original image (alpha blend).
6. Pick one test image per class (10 classes total).
7. Save a 2×10 grid PNG:
     Row 0: original images (with true class label)
     Row 1: Grad-CAM overlays (with predicted class label)
   Output: utils/grad_cam_grid.png

Usage (run from animal_classifier/ directory):
    python utils/grad_cam_visualization.py
"""

import sys
import os

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_AC_ROOT   = os.path.abspath(os.path.join(_UTILS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_AC_ROOT, ".."))

for p in [_AC_ROOT, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def _build_vggbn_layers(p: dict) -> OrderedDict:
    from common.conv_layers import Convolution, Pooling
    from common.layers import Affine, Relu
    from common.layers_ext import BatchNormalization, Dropout

    def make_bn(i):
        return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

    return OrderedDict([
        ("Conv1",   Convolution(p["W1"], p["b1"], stride=1, pad=1)),
        ("BN1",     make_bn(1)),
        ("Relu1",   Relu()),
        ("Conv2",   Convolution(p["W2"], p["b2"], stride=1, pad=1)),
        ("BN2",     make_bn(2)),
        ("Relu2",   Relu()),
        ("Pool1",   Pooling(2, 2, stride=2)),
        ("Conv3",   Convolution(p["W3"], p["b3"], stride=1, pad=1)),
        ("BN3",     make_bn(3)),
        ("Relu3",   Relu()),
        ("Conv4",   Convolution(p["W4"], p["b4"], stride=1, pad=1)),
        ("BN4",     make_bn(4)),
        ("Relu4",   Relu()),
        ("Pool2",   Pooling(2, 2, stride=2)),
        ("Conv5",   Convolution(p["W5"], p["b5"], stride=1, pad=1)),
        ("BN5",     make_bn(5)),
        ("Relu5",   Relu()),
        ("Conv6",   Convolution(p["W6"], p["b6"], stride=1, pad=1)),
        ("BN6",     make_bn(6)),
        ("Relu6",   Relu()),           # ← index 19: target for Grad-CAM
        ("Pool3",   Pooling(2, 2, stride=2)),
        ("Affine1", Affine(p["W7"], p["b7"])),
        ("Relu7",   Relu()),
        ("Drop1",   Dropout(0.5)),
        ("Affine2", Affine(p["W8"], p["b8"])),
    ])


def load_vggbn(pkl_path: str, output_size: int = 10):
    """Load VGGWithBN from pkl; use random weights if pkl absent."""
    from models.vgg_bn import VGGWithBN

    model = VGGWithBN(input_channels=3, input_size=32, output_size=output_size)
    loaded = False

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
        model.params = params
        model.layers = _build_vggbn_layers(params)
        loaded = True
        print(f"Loaded weights from {pkl_path}")
    else:
        print(f"PKL not found ({pkl_path}) — using random weights (CAMs will be uninformative).")

    return model, loaded


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

def compute_grad_cam(model, x: np.ndarray, target_layer_idx: int = 19,
                     class_idx=None):
    """
    Compute Grad-CAM for a batch of images using pure NumPy.

    Parameters
    ----------
    model           : VGGWithBN instance
    x               : np.ndarray  (N, C, H, W)  normalised
    target_layer_idx: int  — index of the hook layer (19 = Relu6)
    class_idx       : int or None  — if None, use predicted class per sample

    Returns
    -------
    cams      : np.ndarray  (N, H, W)  values in [0, 1]
    pred_cls  : np.ndarray  (N,)       predicted class indices
    """
    from common.activation import softmax
    from common.layers_ext import BatchNormalization, Dropout

    layers = list(model.layers.values())
    feature_map = [None]
    grad_map    = [None]

    # ── Forward pass with activation capture ──────────────────────────────
    out = x
    for i, layer in enumerate(layers):
        if isinstance(layer, (BatchNormalization, Dropout)):
            out = layer.forward(out, train_flg=True)
        else:
            out = layer.forward(out)
        if i == target_layer_idx:
            feature_map[0] = out.copy()   # (N, K, fH, fW)

    probs    = softmax(out)
    pred_cls = np.argmax(probs, axis=1)
    targets  = np.full(x.shape[0], class_idx, dtype=int) if class_idx is not None else pred_cls

    # ── Backward pass with gradient capture ───────────────────────────────
    model.last_layer.y = probs
    model.last_layer.t = targets

    # One-hot gradient for target class scores
    N = probs.shape[0]
    dout = np.zeros_like(probs)
    dout[np.arange(N), targets] = 1.0

    for i in range(len(layers) - 1, -1, -1):
        dout = layers[i].backward(dout)
        if i == target_layer_idx:
            grad_map[0] = dout.copy()     # (N, K, fH, fW)

    feat = feature_map[0]   # (N, K, fH, fW)
    grad = grad_map[0]      # (N, K, fH, fW)

    # ── GAP of gradients → channel weights ────────────────────────────────
    weights = grad.mean(axis=(2, 3), keepdims=True)   # (N, K, 1, 1)
    cam_raw = np.sum(weights * feat, axis=1)           # (N, fH, fW)
    cam_raw = np.maximum(cam_raw, 0)                   # ReLU

    # ── Nearest-neighbour resize to input spatial size ─────────────────────
    H_in, W_in = x.shape[2], x.shape[3]
    fH, fW = cam_raw.shape[1], cam_raw.shape[2]
    if fH > 0 and fW > 0:
        sh = max(1, H_in // fH)
        sw = max(1, W_in // fW)
        cam_up = cam_raw.repeat(sh, axis=1).repeat(sw, axis=2)
    else:
        cam_up = np.ones((cam_raw.shape[0], H_in, W_in))

    # Crop / pad to exact target size
    cam_up = cam_up[:, :H_in, :W_in]
    if cam_up.shape[1] < H_in or cam_up.shape[2] < W_in:
        ph = H_in - cam_up.shape[1]
        pw = W_in - cam_up.shape[2]
        cam_up = np.pad(cam_up, ((0, 0), (0, ph), (0, pw)))

    # ── Normalise each CAM to [0, 1] ──────────────────────────────────────
    mins = cam_up.min(axis=(1, 2), keepdims=True)
    maxs = cam_up.max(axis=(1, 2), keepdims=True)
    denom = np.where((maxs - mins) > 1e-8, maxs - mins, 1.0)
    cams = (cam_up - mins) / denom

    return cams, pred_cls


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """Approximate jet colourmap for t in [0, 1]. Returns (H, W, 3) RGB."""
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(img_chw: np.ndarray, cam_hw: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Blend image (C,H,W) in [0,1] with jet-coloured CAM heatmap.

    Returns
    -------
    np.ndarray (H, W, 3) blended image in [0, 1]
    """
    img_hwc = np.clip(img_chw.transpose(1, 2, 0), 0, 1)
    heat    = _jet_colormap(cam_hw)
    return np.clip(alpha * heat + (1 - alpha) * img_hwc, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grad_cam_visualization(
    pkl_path: str | None = None,
    n_samples_per_class: int = 1,
    output_path: str | None = None,
    target_layer_idx: int = 19,
):
    """
    Run Grad-CAM on one test image per CIFAR-10 class and save a 2×10 grid.

    Parameters
    ----------
    pkl_path          : path to VGGWithBN weights pkl (None → auto-detect C02)
    n_samples_per_class: number of samples per class (default 1)
    output_path       : where to save the grid PNG (None → grad_cam_grid.png)
    target_layer_idx  : layer index for hook (19 = Relu6)
    """
    from dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES

    # --- PKL resolution ---
    if pkl_path is None:
        pkl_path = os.path.join(
            _UTILS_DIR, "../experiments/cifar10/C02-vggbn-batchnorm",
            "VGGWithBN_cifar10_mild_best.pkl"
        )

    if output_path is None:
        output_path = os.path.join(_UTILS_DIR, "grad_cam_grid.png")

    # --- Load data ---
    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train_n, x_test_n, _, _ = normalize(x_train, x_test)

    # --- Pick one image per class ---
    n_classes = 10
    samples_x_norm, samples_x_raw, samples_t = [], [], []

    for cls in range(n_classes):
        idxs = np.where(t_test == cls)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        samples_x_norm.append(x_test_n[idx])
        samples_x_raw.append(x_test[idx])
        samples_t.append(t_test[idx])

    samples_x_norm = np.stack(samples_x_norm)   # (10, 3, 32, 32)
    samples_t      = np.array(samples_t)
    n_total        = samples_x_norm.shape[0]

    # --- Load model ---
    model, loaded = load_vggbn(pkl_path, output_size=10)

    # --- Compute Grad-CAM ---
    print(f"Computing Grad-CAM (target_layer_idx={target_layer_idx}) ...")
    cams, pred_classes = compute_grad_cam(
        model, samples_x_norm, target_layer_idx=target_layer_idx
    )

    # --- Build 2×10 figure ---
    fig, axes = plt.subplots(2, n_total, figsize=(n_total * 2, 4.5))
    fig.patch.set_facecolor("#1a1a1a")

    for i in range(n_total):
        cls_name  = CLASS_NAMES[samples_t[i]]
        pred_name = CLASS_NAMES[pred_classes[i]]
        correct   = pred_classes[i] == samples_t[i]
        border_col = "#00ff88" if correct else "#ff4444"

        # Row 0: original
        raw_img = np.clip(samples_x_raw[i].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        axes[0, i].imshow(raw_img)
        axes[0, i].set_title(cls_name, fontsize=7, color="white", pad=2)
        axes[0, i].axis("off")
        for spine in axes[0, i].spines.values():
            spine.set_visible(True)
            spine.set_color(border_col)
            spine.set_linewidth(1.5)

        # Row 1: Grad-CAM overlay
        overlaid = overlay_heatmap(samples_x_norm[i], cams[i], alpha=0.55)
        axes[1, i].imshow(overlaid)
        tick_col = border_col
        axes[1, i].set_title(f"pred: {pred_name}", fontsize=6,
                              color=tick_col, pad=2)
        axes[1, i].axis("off")
        for spine in axes[1, i].spines.values():
            spine.set_visible(True)
            spine.set_color(border_col)
            spine.set_linewidth(1.5)

    axes[0, 0].set_ylabel("Original", fontsize=8, color="white")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8, color="white")

    weight_tag = "trained weights" if loaded else "random weights"
    plt.suptitle(
        f"Grad-CAM — VGGWithBN on CIFAR-10 test images ({weight_tag})\n"
        f"Green border = correct | Red = incorrect",
        fontsize=9, color="white", y=1.02
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Grad-CAM grid → {output_path}")

    # --- Text summary ---
    n_correct = int(np.sum(pred_classes == samples_t))
    print(f"\nPer-class predictions ({n_correct}/{n_total} correct on these samples):")
    for i in range(n_total):
        mark = "✓" if pred_classes[i] == samples_t[i] else "✗"
        print(f"  [{mark}] True: {CLASS_NAMES[samples_t[i]]:<12}  Pred: {CLASS_NAMES[pred_classes[i]]}")

    return cams, pred_classes


if __name__ == "__main__":
    run_grad_cam_visualization()
