import sys
sys.path.append("..")
import os
import argparse
import numpy as np
from dataset.oxford_pets import load_oxford_pets, normalize, get_class_names
from dataset.augmentation import batch_augment, batch_mild_augment
from models.vgg_bn import VGGWithBN
from models.resnet import ResNet
from common.optimizer import Adam
from train import train
from utils.visualization import plot_confusion_matrix, plot_sample_predictions, plot_top5_accuracy


def filter_to_cat_dog(x, t):
    cat_ids = {0,5,6,7,9,11,20,23,26,27,32,33}
    t2 = np.array([0 if ti in cat_ids else 1 for ti in t])
    return x, t2


def filter_top_n(x, t, n=10):
    from collections import Counter
    counts = Counter(t.tolist())
    top_ids = sorted([c for c, _ in counts.most_common(n)])
    id_map = {old: new for new, old in enumerate(top_ids)}
    mask = np.array([ti in id_map for ti in t])
    return x[mask], np.array([id_map[ti] for ti in t[mask]]), top_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg_bn", "resnet"], default="vgg_bn")
    parser.add_argument("--mode", choices=["full37", "cat_dog", "top10"], default="top10")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--aug", choices=["full", "mild", "none"], default="full")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_at", type=int, nargs=2, default=[30, 40])
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir if args.output_dir else f"results_v2_{args.model}_{args.mode}_{args.aug}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Oxford-IIIT Pets ({args.size}x{args.size}) ...")
    (x_train, t_train), (x_test, t_test) = load_oxford_pets(image_size=args.size)
    x_train, x_test, mean, std = normalize(x_train, x_test)
    class_names = get_class_names()

    if args.mode == "cat_dog":
        x_train, t_train = filter_to_cat_dog(x_train, t_train)
        x_test, t_test = filter_to_cat_dog(x_test, t_test)
        class_names = ["Cat", "Dog"]
        n_classes = 2
    elif args.mode == "top10":
        x_train, t_train, top_ids = filter_top_n(x_train, t_train, 10)
        x_test, t_test, _ = filter_top_n(x_test, t_test, 10)
        class_names = [class_names[i] for i in top_ids]
        n_classes = 10
    else:
        n_classes = 37

    print(f"mode={args.mode}, classes={n_classes}, train={len(x_train)}, test={len(x_test)}")
    print(f"classes: {class_names}")

    if args.model == "vgg_bn":
        model = VGGWithBN(3, args.size, n_classes)
        name = f"VGGWithBN_{args.mode}_{args.aug}"
    else:
        model = ResNet(3, n_classes)
        name = f"ResNet_{args.mode}_{args.aug}"

    use_augment = args.aug != "none"
    aug_fn = batch_mild_augment if args.aug == "mild" else batch_augment

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

    plot_confusion_matrix(y_true, y_pred, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_confusion.png"), title=f"{name}")
    plot_sample_predictions(x_test[:n], y_true, y_pred, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_samples.png"),
        denorm_mean=mean, denorm_std=std)
    top1, top5 = plot_top5_accuracy(model, x_test, t_test, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_per_class.png"))
    print(f"\nFinal  Top-1: {top1:.4f}  Top-5: {top5:.4f}")
    model.save(os.path.join(OUTPUT_DIR, f"{name}.pkl"))
