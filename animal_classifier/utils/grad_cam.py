import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM for VGGWithBN (pure NumPy).

    Hooks into one convolutional layer, records its output (feature map)
    and the gradient of the loss w.r.t. that feature map during a backward
    pass, then computes a class-discriminative localization map.

    Algorithm (Selvaraju et al., 2017):
        weights_k = (1/Z) * sum_{i,j} d_score_c / d_A^k_{ij}   (global avg pool of gradients)
        CAM = ReLU( sum_k  weights_k * A^k )
    """

    def __init__(self, model, target_layer_idx):
        """
        Parameters
        ----------
        model : VGGWithBN instance
        target_layer_idx : int
            0-based index into list(model.layers.values()).
            For VGGWithBN, Conv6/BN6/Relu6 (indices 17-19) sit before Pool3.
            Relu6 (index 19) is a convenient target: it stores ReLU activations
            of the last conv block.
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        self._feature_map = None
        self._grad = None

    def _forward_with_hook(self, x):
        """
        Run forward pass with train_flg=True so every layer (including
        Dropout and BN) stores its internal state for backward.
        Captures the feature map at target_layer_idx.
        """
        layers = list(self.model.layers.values())
        from common.layers_ext import BatchNormalization, Dropout

        out = x
        for i, layer in enumerate(layers):
            if isinstance(layer, (BatchNormalization, Dropout)):
                out = layer.forward(out, train_flg=True)
            else:
                out = layer.forward(out)
            if i == self.target_layer_idx:
                self._feature_map = out.copy()
        return out

    def _backward_with_hook(self, target_classes):
        """
        Run backward pass driven by a one-hot gradient for target_classes
        and capture the gradient at target_layer_idx.
        """
        layers = list(self.model.layers.values())

        logits = self.model.last_layer.y   # already softmax probs from generate()
        batch_size = logits.shape[0]

        # One-hot gradient: d(score_c)/d(logit) = delta_{j,c} before softmax,
        # but here we treat the pre-softmax score as what we differentiate.
        # Standard Grad-CAM uses the class score (logit) directly.
        dout = np.zeros_like(logits)
        dout[np.arange(batch_size), target_classes] = 1.0

        grad = None
        for i in range(len(layers) - 1, -1, -1):
            dout = layers[i].backward(dout)
            if i == self.target_layer_idx:
                grad = dout.copy()
        self._grad = grad

    def generate(self, x, class_idx=None):
        """
        Compute Grad-CAM heatmap for a batch of images.

        Parameters
        ----------
        x : np.ndarray  shape (N, C, H, W)
        class_idx : int or None
            Target class. None → use the predicted class per sample.

        Returns
        -------
        cams : np.ndarray  shape (N, H_in, W_in)  values in [0, 1]
        pred_classes : np.ndarray  shape (N,)
        """
        from common.activation import softmax

        logits = self._forward_with_hook(x)
        probs = softmax(logits)
        pred_classes = np.argmax(probs, axis=1)

        if class_idx is None:
            target_classes = pred_classes
        else:
            target_classes = np.full(x.shape[0], class_idx, dtype=int)

        # Store probs so _backward_with_hook can read self.model.last_layer.y
        self.model.last_layer.y = probs
        self.model.last_layer.t = target_classes

        self._backward_with_hook(target_classes)

        feat = self._feature_map    # (N, K, fH, fW)
        grad = self._grad           # (N, K, fH, fW)

        # Global average pool of gradients → channel weights
        weights = grad.mean(axis=(2, 3), keepdims=True)  # (N, K, 1, 1)

        cam_raw = np.sum(weights * feat, axis=1)          # (N, fH, fW)
        cam_raw = np.maximum(cam_raw, 0)                  # ReLU

        # Resize to input spatial size (32×32) via nearest-neighbour repeat
        H_in, W_in = x.shape[2], x.shape[3]
        fH, fW = cam_raw.shape[1], cam_raw.shape[2]
        scale_h = H_in // fH if fH <= H_in else 1
        scale_w = W_in // fW if fW <= W_in else 1
        cam_up = cam_raw.repeat(scale_h, axis=1).repeat(scale_w, axis=2)
        # Crop or pad to exact target size
        cam_up = cam_up[:, :H_in, :W_in]
        if cam_up.shape[1] < H_in or cam_up.shape[2] < W_in:
            pad_h = H_in - cam_up.shape[1]
            pad_w = W_in - cam_up.shape[2]
            cam_up = np.pad(cam_up, ((0, 0), (0, pad_h), (0, pad_w)))

        # Normalize each map to [0, 1]
        mins = cam_up.min(axis=(1, 2), keepdims=True)
        maxs = cam_up.max(axis=(1, 2), keepdims=True)
        denom = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
        cams = (cam_up - mins) / denom

        return cams, pred_classes


def _overlay_heatmap(img_chw, cam_hw, alpha=0.5):
    """Blend a single image (C,H,W) in [0,1] with a heatmap (H,W) in [0,1]."""
    img_hwc = np.clip(img_chw.transpose(1, 2, 0), 0, 1)

    # Jet colormap approximation via NumPy
    r = np.clip(1.5 - np.abs(4 * cam_hw - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_hw - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_hw - 1), 0, 1)
    heat_hwc = np.stack([r, g, b], axis=-1)

    blended = alpha * heat_hwc + (1 - alpha) * img_hwc
    return np.clip(blended, 0, 1)


def _demo():
    import pickle

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
    from models.vgg_bn import VGGWithBN

    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train_n, x_test_n, _, _ = normalize(x_train, x_test)

    model = VGGWithBN(input_channels=3, input_size=32, output_size=10)

    pkl_candidates = [
        os.path.join(os.path.dirname(__file__),
                     "../experiments/cifar10/C02-vggbn-batchnorm/VGGWithBN_cifar10_mild_best.pkl"),
        os.path.join(os.path.dirname(__file__),
                     "../experiments/cifar10/C02-vggbn-batchnorm/VGGWithBN_cifar10_mild.pkl"),
    ]
    loaded = False
    for pkl_path in pkl_candidates:
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                params = pickle.load(f)
            model.params = params
            # Sync layer weights from loaded params
            from collections import OrderedDict
            from common.conv_layers import Convolution, Pooling
            from common.layers import Affine, Relu, SoftmaxWithLoss
            from common.layers_ext import BatchNormalization, Dropout
            p = model.params

            def make_bn(i):
                return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

            model.layers = OrderedDict([
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
                ("Relu6",   Relu()),
                ("Pool3",   Pooling(2, 2, stride=2)),
                ("Affine1", Affine(p["W7"], p["b7"])),
                ("Relu7",   Relu()),
                ("Drop1",   Dropout(0.5)),
                ("Affine2", Affine(p["W8"], p["b8"])),
            ])
            print(f"Loaded weights from {pkl_path}")
            loaded = True
            break

    if not loaded:
        print("No pkl found — using random weights (CAMs will be uninformative).")

    # Layer index of Relu6 in VGGWithBN:
    # 0:Conv1 1:BN1 2:Relu1 3:Conv2 4:BN2 5:Relu2 6:Pool1
    # 7:Conv3 8:BN3 9:Relu3 10:Conv4 11:BN4 12:Relu4 13:Pool2
    # 14:Conv5 15:BN5 16:Relu5 17:Conv6 18:BN6 19:Relu6
    # 20:Pool3 21:Affine1 22:Relu7 23:Drop1 24:Affine2
    target_idx = 19  # Relu6 — last activation before global pool

    cam = GradCAM(model, target_layer_idx=target_idx)

    # Pick one image per class from test set
    samples_x, samples_t = [], []
    for cls in range(10):
        idx = np.where(t_test == cls)[0][0]
        samples_x.append(x_test_n[idx])
        samples_t.append(t_test[idx])
    samples_x = np.stack(samples_x)   # (10, 3, 32, 32)
    samples_t = np.array(samples_t)

    cams, preds = cam.generate(samples_x)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        cls_name = CLASS_NAMES[samples_t[i]]
        pred_name = CLASS_NAMES[preds[i]]

        img_raw = np.clip(x_test[np.where(t_test == i)[0][0]].transpose(1, 2, 0), 0, 1)
        overlaid = _overlay_heatmap(samples_x[i], cams[i])

        axes[0, i].imshow(img_raw)
        axes[0, i].set_title(f"{cls_name}", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlaid)
        axes[1, i].set_title(f"pred:{pred_name}", fontsize=7)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)

    plt.suptitle("Grad-CAM on CIFAR-10 test images (one per class)", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "grad_cam_results.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    _demo()
