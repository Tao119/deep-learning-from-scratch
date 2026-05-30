import sys
sys.path.append("../../../..")
import os
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from animal_classifier.dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization, Dropout
from common.activation import softmax
from common.optimizer import Adam

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# SimCLR augmentation (pure NumPy)
# ---------------------------------------------------------------------------

def simclr_augment(x):
    C, H, W = x.shape

    pad = 4
    padded = np.pad(x, [(0, 0), (pad, pad), (pad, pad)], mode="reflect")
    top = np.random.randint(0, 2 * pad)
    left = np.random.randint(0, 2 * pad)
    x = padded[:, top:top + H, left:left + W]

    if np.random.rand() > 0.5:
        x = x[:, :, ::-1].copy()

    if np.random.rand() > 0.2:
        b = 1.0 + np.random.uniform(-0.4, 0.4)
        c = 1.0 + np.random.uniform(-0.4, 0.4)
        s = 1.0 + np.random.uniform(-0.2, 0.2)
        x = x * b
        mean = x.mean(axis=(1, 2), keepdims=True)
        x = (x - mean) * c + mean
        gray = x.mean(axis=0, keepdims=True)
        x = (1.0 - s) * gray + s * x
        x = np.clip(x, 0, 1)

    if np.random.rand() > 0.5:
        sigma = np.random.uniform(0.1, 1.0)
        ksize = 3
        ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=np.float32)
        kern1d = np.exp(-0.5 * (ax / sigma) ** 2)
        kern1d /= kern1d.sum()
        kernel = kern1d[:, None] * kern1d[None, :]
        for c in range(C):
            padded_c = np.pad(x[c], ksize // 2, mode="reflect")
            out = np.zeros_like(x[c])
            for i in range(H):
                for j in range(W):
                    out[i, j] = np.sum(padded_c[i:i + ksize, j:j + ksize] * kernel)
            x[c] = out

    return x


def batch_simclr_augment(x_batch):
    return np.array([simclr_augment(x) for x in x_batch])


# ---------------------------------------------------------------------------
# Backbone encoder (VGG-like, smaller)
# ---------------------------------------------------------------------------

class SimCLREncoder:
    def __init__(self, input_channels=3, proj_dim=128):
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)

        channel_cfg = [(input_channels, 64), (64, 64), (64, 128), (128, 128)]
        self.params = {}
        for i, (cin, cout) in enumerate(channel_cfg, 1):
            self.params[f"W{i}"] = scale(cin * 9) * np.random.randn(cout, cin, 3, 3)
            self.params[f"b{i}"] = np.zeros(cout)
            self.params[f"gamma{i}"] = np.ones(cout)
            self.params[f"beta{i}"] = np.zeros(cout)

        flat = 128 * 8 * 8
        self.params["W5"] = scale(flat) * np.random.randn(flat, 512)
        self.params["b5"] = np.zeros(512)
        self.params["W6"] = scale(512) * np.random.randn(512, proj_dim)
        self.params["b6"] = np.zeros(proj_dim)

        p = self.params

        def make_bn(i):
            return BatchNormalization(p[f"gamma{i}"], p[f"beta{i}"])

        self.enc_layers = OrderedDict([
            ("Conv1",  Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("BN1",    make_bn(1)),
            ("Relu1",  Relu()),
            ("Conv2",  Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("BN2",    make_bn(2)),
            ("Relu2",  Relu()),
            ("Pool1",  Pooling(2, 2, stride=2)),
            ("Conv3",  Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("BN3",    make_bn(3)),
            ("Relu3",  Relu()),
            ("Conv4",  Convolution(p["W4"], p["b4"], stride=1, pad=1)),
            ("BN4",    make_bn(4)),
            ("Relu4",  Relu()),
            ("Pool2",  Pooling(2, 2, stride=2)),
        ])

        self.proj_layers = OrderedDict([
            ("Affine1", Affine(p["W5"], p["b5"])),
            ("Relu5",   Relu()),
            ("Affine2", Affine(p["W6"], p["b6"])),
        ])

    def encode(self, x, train_flg=False):
        for name, layer in self.enc_layers.items():
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def project(self, h, train_flg=False):
        x = h
        for name, layer in self.proj_layers.items():
            x = layer.forward(x)
        return x

    def forward(self, x, train_flg=False):
        h = self.encode(x, train_flg)
        z = self.project(h, train_flg)
        return h, z

    def _backward_proj(self, dz):
        dout = dz
        for layer in reversed(list(self.proj_layers.values())):
            dout = layer.backward(dout)
        return dout

    def _backward_enc(self, dh):
        dout = dh
        for layer in reversed(list(self.enc_layers.values())):
            dout = layer.backward(dout)
        return dout

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.params = pickle.load(f)


# ---------------------------------------------------------------------------
# NT-Xent loss (normalized temperature-scaled cross entropy)
# ---------------------------------------------------------------------------

def nt_xent_loss_and_grad(z, temperature=0.5):
    N = z.shape[0]
    assert N % 2 == 0
    n = N // 2

    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(z_norm, z_norm.T) / temperature

    sim_flat = sim.copy()
    np.fill_diagonal(sim_flat, -1e9)

    labels = np.concatenate([np.arange(n, N), np.arange(0, n)])

    log_softmax = sim_flat - np.log(np.sum(np.exp(sim_flat - sim_flat.max(axis=1, keepdims=True)), axis=1, keepdims=True)) - sim_flat.max(axis=1, keepdims=True)
    loss = -np.mean(log_softmax[np.arange(N), labels])

    sm = softmax(sim_flat)
    dsim = sm.copy()
    dsim[np.arange(N), labels] -= 1.0
    dsim /= (N * temperature)

    dz_norm = 2.0 * np.dot(dsim + dsim.T, z_norm)
    norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    dz = (dz_norm - z_norm * np.sum(dz_norm * z_norm, axis=1, keepdims=True)) / norm

    return loss, dz


# ---------------------------------------------------------------------------
# Pretrain (contrastive)
# ---------------------------------------------------------------------------

def pretrain_simclr(encoder, x_train, epochs=20, batch_size=256,
                    temperature=0.5, lr=0.001, output_dir="."):
    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    loss_hist = []

    print(f"\n{'='*50}")
    print(f"SimCLR Pretraining  epochs={epochs}  batch={batch_size}  T={temperature}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            x_batch = x_train[batch_idx]

            x_i = batch_simclr_augment(x_batch)
            x_j = batch_simclr_augment(x_batch)
            x_pair = np.concatenate([x_i, x_j], axis=0)

            _, z = encoder.forward(x_pair, train_flg=True)

            loss, dz = nt_xent_loss_and_grad(z, temperature)
            epoch_loss += loss

            dh = encoder._backward_proj(dz)
            encoder._backward_enc(dh)

            grads = {}
            conv_names = ["Conv1", "Conv2", "Conv3", "Conv4"]
            for k, name in enumerate(conv_names, 1):
                grads[f"W{k}"] = encoder.enc_layers[name].dW
                grads[f"b{k}"] = encoder.enc_layers[name].db
                grads[f"gamma{k}"] = encoder.enc_layers[f"BN{k}"].dgamma
                grads[f"beta{k}"] = encoder.enc_layers[f"BN{k}"].dbeta
            grads["W5"] = encoder.proj_layers["Affine1"].dW
            grads["b5"] = encoder.proj_layers["Affine1"].db
            grads["W6"] = encoder.proj_layers["Affine2"].dW
            grads["b6"] = encoder.proj_layers["Affine2"].db

            optimizer.update(encoder.params, grads)

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}/{epochs}: contrastive_loss={epoch_loss:.4f}  ({elapsed:.0f}s)")

    encoder.save(os.path.join(output_dir, "simclr_encoder.pkl"))
    return loss_hist


# ---------------------------------------------------------------------------
# Linear evaluation on top of frozen encoder
# ---------------------------------------------------------------------------

class LinearHead:
    def __init__(self, in_dim, num_classes=10):
        scale = np.sqrt(2.0 / in_dim)
        self.W = scale * np.random.randn(in_dim, num_classes)
        self.b = np.zeros(num_classes)
        self.x = None
        self.last_layer = SoftmaxWithLoss()
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def loss(self, x, t):
        logits = self.forward(x)
        return self.last_layer.forward(logits, t)

    def backward(self):
        dlogits = self.last_layer.backward(1)
        self.dW = np.dot(self.x.T, dlogits)
        self.db = np.sum(dlogits, axis=0)

    def accuracy(self, x, t):
        logits = self.forward(x)
        pred = np.argmax(logits, axis=1)
        label = np.argmax(t, axis=1) if t.ndim != 1 else t
        return np.mean(pred == label)


def get_representations(encoder, x, batch_size=128):
    reps = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        h = encoder.encode(x[i:i + batch_size], train_flg=False)
        h_flat = h.reshape(h.shape[0], -1)
        reps.append(h_flat)
    return np.vstack(reps)


def linear_eval(encoder, x_train_sub, t_train_sub, x_test, t_test,
                epochs=30, batch_size=256, lr=0.01, output_dir="."):
    print(f"\n{'='*50}")
    print(f"Linear Evaluation on {len(x_train_sub)} labeled samples (1%)")
    print(f"{'='*50}")

    print("Extracting representations ...")
    h_train = get_representations(encoder, x_train_sub)
    h_test = get_representations(encoder, x_test)

    h_mean = h_train.mean(axis=0)
    h_std = h_train.std(axis=0) + 1e-8
    h_train = (h_train - h_mean) / h_std
    h_test = (h_test - h_mean) / h_std

    in_dim = h_train.shape[1]
    head = LinearHead(in_dim, num_classes=10)
    optimizer = Adam(lr=lr)
    train_size = h_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    acc_hist = []
    best_acc = 0.0

    for epoch in range(epochs):
        idx = np.random.permutation(train_size)
        for i in range(iter_per_epoch):
            bi = idx[i * batch_size:(i + 1) * batch_size]
            loss = head.loss(h_train[bi], t_train_sub[bi])
            head.backward()
            optimizer.update({"W": head.W, "b": head.b},
                             {"W": head.dW, "b": head.db})

        test_acc = head.accuracy(h_test, t_test)
        acc_hist.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0:
            train_acc = head.accuracy(h_train, t_train_sub)
            print(f"epoch {epoch+1:3d}/{epochs}: train={train_acc:.4f}  test={test_acc:.4f}")

    return best_acc, acc_hist


def supervised_baseline(x_sub, t_sub, x_test, t_test,
                        epochs=30, batch_size=128, lr=0.001, output_dir="."):
    from animal_classifier.models.vgg_bn import VGGWithBN
    from animal_classifier.dataset.augmentation import batch_mild_augment

    print(f"\n{'='*50}")
    print(f"Supervised baseline on {len(x_sub)} samples (1%)")
    print(f"{'='*50}")

    model = VGGWithBN(3, 32, 10)
    optimizer = Adam(lr=lr)
    train_size = x_sub.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    best_acc = 0.0
    acc_hist = []

    for epoch in range(epochs):
        idx = np.random.permutation(train_size)
        for i in range(iter_per_epoch):
            bi = idx[i * batch_size:(i + 1) * batch_size]
            xb = batch_mild_augment(x_sub[bi])
            grads = model.gradient(xb, t_sub[bi])
            optimizer.update(model.params, grads)

        test_acc = model.accuracy(x_test, t_test)
        acc_hist.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:3d}/{epochs}: test={test_acc:.4f}")

    return best_acc, acc_hist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--eval_epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--label_fraction", type=float, default=0.01,
                        help="fraction of training labels to use for linear eval")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="load existing encoder checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, mean, std = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    encoder = SimCLREncoder(input_channels=3, proj_dim=args.proj_dim)

    ckpt = os.path.join(args.output_dir, "simclr_encoder.pkl")
    if args.skip_pretrain and os.path.exists(ckpt):
        print(f"Loading encoder from {ckpt}")
        encoder.load(ckpt)
        loss_hist = []
    else:
        loss_hist = pretrain_simclr(
            encoder, x_train,
            epochs=args.pretrain_epochs,
            batch_size=args.batch,
            temperature=args.temperature,
            lr=0.001,
            output_dir=args.output_dir,
        )

    n_labeled = int(len(x_train) * args.label_fraction)
    labeled_idx = np.random.choice(len(x_train), n_labeled, replace=False)
    x_sub = x_train[labeled_idx]
    t_sub = t_train[labeled_idx]
    print(f"\nUsing {n_labeled} labeled samples ({args.label_fraction*100:.1f}%) for evaluation")

    simclr_acc, simclr_hist = linear_eval(
        encoder, x_sub, t_sub, x_test, t_test,
        epochs=args.eval_epochs, batch_size=min(256, n_labeled),
        lr=0.01, output_dir=args.output_dir,
    )

    sup_acc, sup_hist = supervised_baseline(
        x_sub, t_sub, x_test, t_test,
        epochs=args.eval_epochs, batch_size=min(128, n_labeled),
        lr=0.001, output_dir=args.output_dir,
    )

    print(f"\n{'='*50}")
    print(f"SimCLR linear eval best acc  ({args.label_fraction*100:.1f}% labels): {simclr_acc:.4f}")
    print(f"Supervised baseline best acc ({args.label_fraction*100:.1f}% labels): {sup_acc:.4f}")
    print(f"SimCLR advantage: {simclr_acc - sup_acc:+.4f}")
    print(f"{'='*50}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if loss_hist:
        axes[0].plot(range(1, len(loss_hist) + 1), loss_hist, color="purple")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("NT-Xent loss")
        axes[0].set_title("SimCLR Pretraining Loss")
    else:
        axes[0].text(0.5, 0.5, "Loaded from checkpoint", ha="center", va="center")

    axes[1].plot(range(1, len(simclr_hist) + 1), simclr_hist, label=f"SimCLR linear ({args.label_fraction*100:.0f}%)")
    axes[1].plot(range(1, len(sup_hist) + 1), sup_hist, linestyle="--", label=f"Supervised ({args.label_fraction*100:.0f}%)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Linear Evaluation vs Supervised Baseline")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "simclr_results.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved {fig_path}")

    summary_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Label fraction: {args.label_fraction*100:.1f}%  ({n_labeled} samples)\n")
        f.write(f"SimCLR linear eval best acc: {simclr_acc:.4f}\n")
        f.write(f"Supervised baseline best acc: {sup_acc:.4f}\n")
        f.write(f"SimCLR advantage: {simclr_acc - sup_acc:+.4f}\n")
    print(f"Saved summary to {summary_path}")
