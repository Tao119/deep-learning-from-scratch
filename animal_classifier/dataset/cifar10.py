import os
import pickle
import tarfile
import urllib.request
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_data")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def _download():
    os.makedirs(DATASET_DIR, exist_ok=True)
    archive = os.path.join(DATASET_DIR, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print("Downloading CIFAR-10 ...")
        urllib.request.urlretrieve(URL, archive)
    extracted = os.path.join(DATASET_DIR, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        print("Extracting ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(DATASET_DIR)
    return extracted


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    t = np.array(d[b"labels"], dtype=np.int64)
    return x, t


def load_cifar10(cache=True):
    cache_path = os.path.join(DATASET_DIR, "cifar10.npz")
    if cache and os.path.exists(cache_path):
        print("Loading from cache ...")
        data = np.load(cache_path)
        return (data["x_train"], data["t_train"]), (data["x_test"], data["t_test"])

    batch_dir = _download()

    xs, ts = [], []
    for i in range(1, 6):
        x, t = _load_batch(os.path.join(batch_dir, f"data_batch_{i}"))
        xs.append(x)
        ts.append(t)
    x_train = np.concatenate(xs)
    t_train = np.concatenate(ts)

    x_test, t_test = _load_batch(os.path.join(batch_dir, "test_batch"))

    if cache:
        np.savez(cache_path, x_train=x_train, t_train=t_train,
                 x_test=x_test, t_test=t_test)
        print(f"Cached to {cache_path}")

    return (x_train, t_train), (x_test, t_test)


def normalize(x_train, x_test):
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_test - mean) / std, mean, std
