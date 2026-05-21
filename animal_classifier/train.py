import sys
sys.path.append("..")
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset.oxford_pets import load_oxford_pets, normalize
from dataset.augmentation import batch_augment
from common.optimizer import Adam


def train(model, model_name, x_train, t_train, x_test, t_test,
          epochs=30, batch_size=32, lr=0.001, use_augment=True,
          lr_decay_at=(20, 25), lr_decay_factor=0.1, output_dir=".",
          aug_fn=None):

    optimizer = Adam(lr=lr)
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    best_test_acc = 0.0
    current_lr = lr

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"  train={train_size}, batch={batch_size}, epochs={epochs}, lr={lr}")
    print(f"  augmentation={'ON' if use_augment else 'OFF'}")
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
            batch_idx = idx[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[batch_idx]
            t_batch = t_train[batch_idx]

            if use_augment:
                fn = aug_fn if aug_fn is not None else batch_augment
                x_batch = fn(x_batch, train_flg=True)

            if hasattr(model, "gradient"):
                grads = model.gradient(x_batch, t_batch)
                if grads:
                    optimizer.update(model.params, grads)
                else:
                    model.update(optimizer)
            else:
                model.loss(x_batch, t_batch)
                model.update(optimizer)

            epoch_loss += model.last_layer.loss

        epoch_loss /= iter_per_epoch
        loss_hist.append(epoch_loss)

        eval_n = min(500, len(x_train))
        tr_idx = np.random.choice(len(x_train), eval_n, replace=False)
        te_idx = np.random.choice(len(x_test), eval_n, replace=False)
        train_acc = model.accuracy(x_train[tr_idx], t_train[tr_idx])
        test_acc = model.accuracy(x_test[te_idx], t_test[te_idx])
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  ({elapsed:.0f}s)")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if hasattr(model, "save"):
                model.save(os.path.join(output_dir, f"{model_name}_best.pkl"))

    os.makedirs(output_dir, exist_ok=True)
    _plot_history(train_acc_hist, test_acc_hist, loss_hist, model_name, output_dir)
    print(f"\nBest test acc: {best_test_acc:.4f}")
    return train_acc_hist, test_acc_hist, loss_hist


def _plot_history(train_acc, test_acc, loss, name, output_dir):
    epochs = range(1, len(train_acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_acc, label="train")
    ax1.plot(epochs, test_acc, linestyle="--", label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{name} - Accuracy")
    ax1.legend()
    ax2.plot(epochs, loss, color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"{name} - Loss")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name}_history.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg", "vgg_bn", "resnet"], default="vgg")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no_aug", action="store_true")
    args = parser.parse_args()

    print("Loading Oxford-IIIT Pets ...")
    (x_train, t_train), (x_test, t_test) = load_oxford_pets(image_size=64)
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"train: {x_train.shape}, test: {x_test.shape}")
    print(f"classes: {t_train.max() + 1}")

    if args.model == "vgg":
        from models.vgg_like import VGGLike
        model = VGGLike(input_channels=3, input_size=64, output_size=37)
        name = "VGGLike"
    elif args.model == "vgg_bn":
        from models.vgg_bn import VGGWithBN
        model = VGGWithBN(input_channels=3, input_size=64, output_size=37)
        name = "VGGWithBN"
    else:
        from models.resnet import ResNet
        model = ResNet(input_channels=3, output_size=37)
        name = "ResNet"

    train(model, name, x_train, t_train, x_test, t_test,
          epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, use_augment=not args.no_aug,
          output_dir="results")
