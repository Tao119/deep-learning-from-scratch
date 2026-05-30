import sys
sys.path.append("..")
import os
import argparse
import numpy as np
from dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from dataset.augmentation import batch_augment, batch_mild_augment
from models.vgg_like import VGGLike
from models.vgg_bn import VGGWithBN
from models.resnet import ResNet
from models.se_resnet import SEResNet
from models.densenet import DenseNet
from models.mobilenet import MobileNet
from train import train
from utils.visualization import plot_confusion_matrix, plot_sample_predictions, plot_top5_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg", "vgg_bn", "resnet", "se_resnet", "densenet", "mobilenet"], default="vgg_bn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--aug", choices=["full", "mild", "none"], default="mild")
    parser.add_argument("--decay_at", type=int, nargs=2, default=[30, 40])
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    model_key = args.model
    OUTPUT_DIR = args.output_dir if args.output_dir else f"results_cifar10_{model_key}_{args.aug}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, mean, std = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    if model_key == "vgg":
        model = VGGLike(3, 32, 10)
        name = f"VGGLike_cifar10_{args.aug}"
    elif model_key == "vgg_bn":
        model = VGGWithBN(3, 32, 10)
        name = f"VGGWithBN_cifar10_{args.aug}"
    elif model_key == "resnet":
        model = ResNet(3, 10)
        name = f"ResNet_cifar10_{args.aug}"
    elif model_key == "se_resnet":
        model = SEResNet(3, 10)
        name = f"SEResNet_cifar10_{args.aug}"
    elif model_key == "densenet":
        model = DenseNet(3, 10)
        name = f"DenseNet_cifar10_{args.aug}"
    else:
        model = MobileNet(3, 10)
        name = f"MobileNet_cifar10_{args.aug}"

    aug_fn = batch_mild_augment if args.aug == "mild" else batch_augment
    use_augment = args.aug != "none"

    train_acc, test_acc, loss = train(
        model, name, x_train, t_train, x_test, t_test,
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, use_augment=use_augment,
        lr_decay_at=tuple(args.decay_at), lr_decay_factor=0.1,
        output_dir=OUTPUT_DIR,
        aug_fn=aug_fn,
    )

    n = (len(x_test) // args.batch) * args.batch
    all_probs = np.vstack([
        model.predict(x_test[i:i+args.batch], train_flg=False)
        for i in range(0, n, args.batch)
    ])
    y_pred = np.argmax(all_probs, axis=1)
    y_true = t_test[:n]

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_confusion.png"), title=name)
    plot_sample_predictions(x_test[:n], y_true, y_pred, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_samples.png"),
        denorm_mean=mean, denorm_std=std)
    top1, top5 = plot_top5_accuracy(model, x_test, t_test, CLASS_NAMES,
        os.path.join(OUTPUT_DIR, f"{name}_per_class.png"))
    print(f"\nFinal  Top-1: {top1:.4f}  Top-5: {top5:.4f}")
    model.save(os.path.join(OUTPUT_DIR, f"{name}.pkl"))
