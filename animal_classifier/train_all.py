import sys
sys.path.append("..")
import os
import numpy as np
from dataset.oxford_pets import load_oxford_pets, normalize, get_class_names
from dataset.augmentation import batch_augment
from models.vgg_like import VGGLike
from models.vgg_bn import VGGWithBN
from models.resnet import ResNet
from common.optimizer import Adam
from train import train
from utils.visualization import plot_model_comparison, plot_confusion_matrix, plot_sample_predictions, plot_top5_accuracy

IMAGE_SIZE = 32
EPOCHS = 30
BATCH = 32
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Oxford-IIIT Pets (32x32) ...")
(x_train, t_train), (x_test, t_test) = load_oxford_pets(image_size=IMAGE_SIZE)
x_train, x_test, mean, std = normalize(x_train, x_test)
print(f"train: {x_train.shape}, test: {x_test.shape}")

class_names = get_class_names()
print(f"class_names sample: {class_names[:5]}")
print(f"classes: {len(class_names)} -> {class_names[:5]} ...")

configs = [
    ("VGGLike",   VGGLike(3, IMAGE_SIZE, 37),   True,  "VGG baseline + augmentation"),
    ("VGGWithBN", VGGWithBN(3, IMAGE_SIZE, 37),  True,  "VGG + BatchNorm + augmentation"),
    ("ResNet",    ResNet(3, 37),                  True,  "ResNet skip connections"),
    ("VGG_NoAug", VGGLike(3, IMAGE_SIZE, 37),    False, "VGG baseline, no augmentation"),
]

all_results = {}

for name, model, use_aug, desc in configs:
    print(f"\n[{desc}]")
    train_acc, test_acc, loss = train(
        model, name, x_train, t_train, x_test, t_test,
        epochs=EPOCHS, batch_size=BATCH, lr=0.001,
        use_augment=use_aug,
        lr_decay_at=(20, 25), lr_decay_factor=0.1,
        output_dir=OUTPUT_DIR
    )
    all_results[name] = (train_acc, test_acc, loss)

    n = (len(x_test) // BATCH) * BATCH
    all_probs = np.vstack([
        model.predict(x_test[i:i+BATCH], train_flg=False)
        for i in range(0, n, BATCH)
    ])
    y_pred = np.argmax(all_probs, axis=1)
    y_true = t_test[:n]

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_confusion.png"),
        title=f"{name} Confusion Matrix"
    )
    plot_sample_predictions(
        x_test[:n], y_true, y_pred, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_samples.png"),
        denorm_mean=mean, denorm_std=std
    )
    top1, top5 = plot_top5_accuracy(
        model, x_test, t_test, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_per_class.png")
    )
    print(f"  Final Top-1: {top1:.4f}, Top-5: {top5:.4f}")

plot_model_comparison(all_results, os.path.join(OUTPUT_DIR, "comparison.png"))
print(f"\nAll done. Results in {OUTPUT_DIR}/")
