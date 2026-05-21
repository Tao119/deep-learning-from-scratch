import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
