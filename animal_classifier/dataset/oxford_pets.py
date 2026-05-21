import os
import tarfile
import urllib.request
import numpy as np
from PIL import Image

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oxford_pets_data")
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

IMAGE_SIZE = 64


def _download(url, dest_dir):
    fname = os.path.basename(url)
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        return dest
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {fname} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Extracting {fname} ...")
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(dest_dir)
    return dest


def _parse_split(split_file, images_dir):
    entries = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = parts[0]
            class_id = int(parts[1]) - 1
            img_path = os.path.join(images_dir, name + ".jpg")
            if os.path.exists(img_path):
                entries.append((img_path, class_id))
    return entries


def _load_entries(entries, image_size):
    xs, ts = [], []
    for img_path, class_id in entries:
        try:
            img = Image.open(img_path).convert("RGB").resize(
                (image_size, image_size), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0
            xs.append(arr.transpose(2, 0, 1))
            ts.append(class_id)
        except Exception:
            continue
    return np.array(xs, dtype=np.float32), np.array(ts, dtype=np.int64)


def get_class_names(annots_dir=None):
    if annots_dir is None:
        annots_dir = os.path.join(DATASET_DIR, "annotations")
    list_file = os.path.join(annots_dir, "list.txt")
    if not os.path.exists(list_file):
        return [str(i) for i in range(37)]

    class_map = {}
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name = parts[0]
            class_id = int(parts[1]) - 1
            breed = img_name.rsplit("_", 1)[0].replace("_", " ")
            if class_id not in class_map:
                class_map[class_id] = breed

    names = [class_map.get(i, f"class_{i}") for i in range(37)]
    return names


def load_oxford_pets(image_size=IMAGE_SIZE, cache=True):
    cache_path = os.path.join(DATASET_DIR, f"pets_{image_size}.npz")
    if cache and os.path.exists(cache_path):
        print("Loading from cache ...")
        data = np.load(cache_path)
        return (data["x_train"], data["t_train"]), (data["x_test"], data["t_test"])

    _download(IMAGES_URL, DATASET_DIR)
    _download(ANNOTS_URL, DATASET_DIR)

    images_dir = os.path.join(DATASET_DIR, "images")
    annots_dir = os.path.join(DATASET_DIR, "annotations")
    train_file = os.path.join(annots_dir, "trainval.txt")
    test_file = os.path.join(annots_dir, "test.txt")

    print("Loading train data ...")
    train_entries = _parse_split(train_file, images_dir)
    x_train, t_train = _load_entries(train_entries, image_size)

    print("Loading test data ...")
    test_entries = _parse_split(test_file, images_dir)
    x_test, t_test = _load_entries(test_entries, image_size)

    if cache:
        np.savez(cache_path, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
        print(f"Cached to {cache_path}")

    return (x_train, t_train), (x_test, t_test)


def normalize(x_train, x_test):
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_test - mean) / std, mean, std
