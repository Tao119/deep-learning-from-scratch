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
from animal_classifier.dataset.augmentation import batch_mild_augment
from animal_classifier.models.vgg_bn import VGGWithBN
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization, Dropout
from common.activation import softmax
from common.optimizer import Adam

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHER_PKL = os.path.join(
    os.path.dirname(__file__),
    "../../C02-vggbn-batchnorm/VGGWithBN_cifar10_mild_best.pkl"
)


class StudentNet:
    def __init__(self, input_channels=3, input_size=32, output_size=10):
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)

        self.params = {
            "W1": scale(input_channels * 9) * np.random.randn(32, input_channels, 3, 3),
            "b1": np.zeros(32),
            "W2": scale(32 * 9) * np.random.randn(64, 32, 3, 3),
            "b2": np.zeros(64),
            "W3": scale(64 * 9) * np.random.randn(128, 64, 3, 3),
            "b3": np.zeros(128),
        }
        flat = 128 * (input_size // 8) ** 2
        self.params["W4"] = scale(flat) * np.random.randn(flat, 256)
        self.params["b4"] = np.zeros(256)
        self.params["W5"] = scale(256) * np.random.randn(256, output_size)
        self.params["b5"] = np.zeros(output_size)

        p = self.params
        self.layers = OrderedDict([
            ("Conv1",  Convolution(p["W1"], p["b1"], stride=1, pad=1)),
            ("Relu1",  Relu()),
            ("Pool1",  Pooling(2, 2, stride=2)),
            ("Conv2",  Convolution(p["W2"], p["b2"], stride=1, pad=1)),
            ("Relu2",  Relu()),
            ("Pool2",  Pooling(2, 2, stride=2)),
            ("Conv3",  Convolution(p["W3"], p["b3"], stride=1, pad=1)),
            ("Relu3",  Relu()),
            ("Pool3",  Pooling(2, 2, stride=2)),
            ("Affine1", Affine(p["W4"], p["b4"])),
            ("Relu4",  Relu()),
            ("Drop1",  Dropout(0.4)),
            ("Affine2", Affine(p["W5"], p["b5"])),
        ])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=128):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc, n = 0, (x.shape[0] // batch_size) * batch_size
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i + batch_size], train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == t_label[i:i + batch_size])
        return acc / n if n > 0 else 0.0

    def gradient_kd(self, x, soft_targets, hard_targets, alpha=0.7, T=4.0):
        logits = self.predict(x, train_flg=True)

        soft_pred = softmax(logits / T)
        kd_loss = -np.mean(np.sum(soft_targets * np.log(soft_pred + 1e-7), axis=1))

        hard_pred = softmax(logits)
        ce_loss = self.last_layer.forward(logits, hard_targets)

        total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
        self.last_layer.loss = total_loss

        d_soft = (soft_pred - soft_targets) / (x.shape[0] * T)
        d_hard = self.last_layer.backward(1)
        dlogits = alpha * d_soft + (1.0 - alpha) * d_hard

        dout = dlogits
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        layer_names = ["Conv1", "Conv2", "Conv3", "Affine1", "Affine2"]
        for i, name in enumerate(layer_names, 1):
            grads[f"W{i}"] = self.layers[name].dW
            grads[f"b{i}"] = self.layers[name].db
        return grads, total_loss

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)


def load_teacher(path):
    teacher = VGGWithBN(3, 32, 10)
    with open(path, "rb") as f:
        teacher.params = pickle.load(f)
    for i, (name, layer) in enumerate(teacher.layers.items()):
        if hasattr(layer, "W"):
            key = name.replace("Conv", "W").replace("Affine", "W")
    for name, layer in teacher.layers.items():
        if isinstance(layer, Convolution):
            pass
    return teacher


def get_teacher_soft_labels(teacher, x, T=4.0, batch_size=128):
    n = x.shape[0]
    soft = []
    for i in range(0, n, batch_size):
        logits = teacher.predict(x[i:i + batch_size], train_flg=False)
        soft.append(softmax(logits / T))
    return np.vstack(soft)


def train_student(student, x_train, t_train, soft_train,
                  x_test, t_test,
                  epochs=30, batch_size=256, lr=0.001,
                  lr_decay_at=(20, 25), lr_decay_factor=0.1,
                  alpha=0.7, T=4.0, output_dir="."):

    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    current_lr = lr

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0

    print(f"\n{'='*50}")
    print(f"Training StudentNet with KD  alpha={alpha}  T={T}")
    print(f"  train={train_size}, batch={batch_size}, epochs={epochs}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        if epoch in lr_decay_at:
            current_lr *= lr_decay_factor
            optimizer.lr = current_lr
            print(f"  LR decayed to {current_lr:.2e}")

        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            x_b = x_train[batch_idx]
            t_b = t_train[batch_idx]
            soft_b = soft_train[batch_idx]

            x_b = batch_mild_augment(x_b, train_flg=True)
            grads, loss_val = student.gradient_kd(x_b, soft_b, t_b, alpha=alpha, T=T)
            optimizer.update(student.params, grads)
            epoch_loss += loss_val

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)

        eval_n = min(500, len(x_train))
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test), eval_n, replace=False)
        train_acc = student.accuracy(x_train[tr_idx], t_train[tr_idx])
        test_acc = student.accuracy(x_test[te_idx], t_test[te_idx])
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  ({elapsed:.0f}s)")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            student.save(os.path.join(output_dir, "StudentNet_kd_best.pkl"))

    return train_acc_hist, test_acc_hist, loss_hist, best_test_acc


def train_scratch(student, x_train, t_train, x_test, t_test,
                  epochs=30, batch_size=256, lr=0.001,
                  lr_decay_at=(20, 25), lr_decay_factor=0.1,
                  output_dir="."):

    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    current_lr = lr

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0

    print(f"\n{'='*50}")
    print(f"Training StudentNet from scratch (no KD)")
    print(f"{'='*50}")

    for epoch in range(epochs):
        if epoch in lr_decay_at:
            current_lr *= lr_decay_factor
            optimizer.lr = current_lr
            print(f"  LR decayed to {current_lr:.2e}")

        epoch_loss = 0.0
        t0 = time.time()
        idx = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            x_b = x_train[batch_idx]
            t_b = t_train[batch_idx]
            x_b = batch_mild_augment(x_b, train_flg=True)

            loss_val = student.loss(x_b, t_b)
            dout = student.last_layer.backward(1)
            for layer in reversed(list(student.layers.values())):
                dout = layer.backward(dout)

            grads = {}
            layer_names = ["Conv1", "Conv2", "Conv3", "Affine1", "Affine2"]
            for j, name in enumerate(layer_names, 1):
                grads[f"W{j}"] = student.layers[name].dW
                grads[f"b{j}"] = student.layers[name].db

            optimizer.update(student.params, grads)
            epoch_loss += loss_val

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)

        eval_n = min(500, len(x_train))
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test), eval_n, replace=False)
        train_acc = student.accuracy(x_train[tr_idx], t_train[tr_idx])
        test_acc = student.accuracy(x_test[te_idx], t_test[te_idx])
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  ({elapsed:.0f}s)")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            student.save(os.path.join(output_dir, "StudentNet_scratch_best.pkl"))

    return train_acc_hist, test_acc_hist, loss_hist, best_test_acc


def plot_comparison(kd_hist, scratch_hist, output_dir):
    epochs = range(1, len(kd_hist[0]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, kd_hist[1], label="KD student test")
    ax.plot(epochs, scratch_hist[1], linestyle="--", label="scratch student test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Student Test Accuracy: KD vs Scratch")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, kd_hist[2], label="KD loss")
    ax.plot(epochs, scratch_hist[2], linestyle="--", label="scratch loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "kd_vs_scratch.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="KD loss weight: alpha*soft + (1-alpha)*hard")
    parser.add_argument("--T", type=float, default=4.0, help="distillation temperature")
    parser.add_argument("--teacher_pkl", type=str, default=TEACHER_PKL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, mean, std = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    teacher_pkl = os.path.abspath(args.teacher_pkl)
    if os.path.exists(teacher_pkl):
        print(f"Loading teacher from {teacher_pkl}")
        teacher = load_teacher(teacher_pkl)
        teacher_acc = teacher.accuracy(x_test, t_test, batch_size=128)
        print(f"Teacher test accuracy: {teacher_acc:.4f}")
    else:
        print("Teacher checkpoint not found. Training teacher briefly (10 epochs) ...")
        teacher = VGGWithBN(3, 32, 10)
        opt_t = Adam(lr=0.001)
        for ep in range(10):
            idx = np.random.permutation(len(x_train))
            for i in range(len(x_train) // 256):
                bi = idx[i * 256:(i + 1) * 256]
                xb = batch_mild_augment(x_train[bi])
                grads = teacher.gradient(xb, t_train[bi])
                opt_t.update(teacher.params, grads)
            acc = teacher.accuracy(x_test[:1000], t_test[:1000])
            print(f"  teacher epoch {ep+1}: test={acc:.4f}")
        teacher.save(os.path.join(args.output_dir, "teacher_trained.pkl"))

    print("\nGenerating soft labels from teacher (T={})...".format(args.T))
    soft_train = get_teacher_soft_labels(teacher, x_train, T=args.T, batch_size=128)
    print(f"soft_train shape: {soft_train.shape}")

    student_kd = StudentNet(3, 32, 10)
    kd_hist = train_student(
        student_kd, x_train, t_train, soft_train, x_test, t_test,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        lr_decay_at=(20, 25), lr_decay_factor=0.1,
        alpha=args.alpha, T=args.T, output_dir=args.output_dir,
    )

    student_scratch = StudentNet(3, 32, 10)
    scratch_hist = train_scratch(
        student_scratch, x_train, t_train, x_test, t_test,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        lr_decay_at=(20, 25), lr_decay_factor=0.1,
        output_dir=args.output_dir,
    )

    kd_best = kd_hist[3]
    scratch_best = scratch_hist[3]
    print(f"\n{'='*50}")
    print(f"Teacher accuracy:            {teacher_acc:.4f}" if os.path.exists(teacher_pkl) else "")
    print(f"Student (KD) best test acc:  {kd_best:.4f}")
    print(f"Student (scratch) best acc:  {scratch_best:.4f}")
    print(f"KD gain over scratch:        {kd_best - scratch_best:+.4f}")
    print(f"{'='*50}")

    plot_comparison(kd_hist, scratch_hist, args.output_dir)

    summary_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"KD student best test acc:    {kd_best:.4f}\n")
        f.write(f"Scratch student best acc:    {scratch_best:.4f}\n")
        f.write(f"KD gain:                     {kd_best - scratch_best:+.4f}\n")
    print(f"Saved summary to {summary_path}")
