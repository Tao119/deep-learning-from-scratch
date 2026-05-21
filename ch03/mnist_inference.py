import sys
sys.path.append("..")
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.activation import sigmoid, softmax


def get_data():
    (_, _), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


if __name__ == "__main__":
    x, t = get_data()
    print(f"test data shape: {x.shape}")
    print(f"test label shape: {t.shape}")
    print("MNIST loaded successfully")
    print(f"pixel range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"label examples: {t[:10]}")
