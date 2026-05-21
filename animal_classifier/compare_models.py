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

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Oxford-IIIT Pets ...")
(x_train, t_train), (x_test, t_test) = load_oxford_pets(image_size=64)
x_train, x_test, mean, std = normalize(x_train, x_test)
print(f"train: {x_train.shape}, test: {x_test.shape}")

annots_dir = os.path.join("dataset", "oxford_pets_data", "annotations")
class_names = get_class_names(annots_dir)
if len(class_names) < 37:
    class_names = [f"class_{i}" for i in range(37)]
print(f"classes: {len(class_names)}")

EPOCHS = 30
BATCH = 32
LR = 0.001

configs = [
    ("VGGLike",    VGGLike(3, 64, 37),    False),
    ("VGGWithBN",  VGGWithBN(3, 64, 37),  True),
    ("ResNet",     ResNet(3, 37),          True),
    ("VGG_NoAug",  VGGLike(3, 64, 37),    False),
]

all_results = {}

for name, model, use_aug in configs:
    train_acc, test_acc, loss = train(
        model, name, x_train, t_train, x_test, t_test,
        epochs=EPOCHS, batch_size=BATCH, lr=LR,
        use_augment=use_aug, output_dir=OUTPUT_DIR
    )
    all_results[name] = (train_acc, test_acc, loss)

    n = (x_test.shape[0] // BATCH) * BATCH
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
    plot_top5_accuracy(
        model, x_test, t_test, class_names,
        os.path.join(OUTPUT_DIR, f"{name}_per_class.png")
    )

plot_model_comparison(all_results, os.path.join(OUTPUT_DIR, "comparison.png"))
print("\nAll done. Results saved to", OUTPUT_DIR)
