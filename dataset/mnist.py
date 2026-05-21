import os
import gzip
import pickle
import numpy as np
import urllib.request

url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir, "mnist.pkl")

train_num = 60000
test_num = 10000
img_size = 784


def _download(file_name):
    path = os.path.join(dataset_dir, file_name)
    if os.path.exists(path):
        return
    print(f"Downloading {file_name} ...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url_base + file_name, headers=headers)
    with urllib.request.urlopen(req) as response, open(path, "wb") as f:
        f.write(response.read())


def _load_label(file_name):
    path = os.path.join(dataset_dir, file_name)
    with gzip.open(path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


def _load_img(file_name):
    path = os.path.join(dataset_dir, file_name)
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, img_size)


def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(key_file["train_img"])
    dataset["train_label"] = _load_label(key_file["train_label"])
    dataset["test_img"] = _load_img(key_file["test_img"])
    dataset["test_label"] = _load_label(key_file["test_label"])
    return dataset


def init_mnist():
    for v in key_file.values():
        _download(v)
    dataset = _convert_numpy()
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f)


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (
        (dataset["train_img"], dataset["train_label"]),
        (dataset["test_img"], dataset["test_label"]),
    )
