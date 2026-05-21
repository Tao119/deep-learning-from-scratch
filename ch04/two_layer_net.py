import sys
sys.path.append("..")
import numpy as np
from common.activation import sigmoid, softmax
from common.loss import cross_entropy_error
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        z1 = sigmoid(np.dot(x, W1) + b1)
        return softmax(np.dot(z1, W2) + b2)

    def loss(self, x, t):
        return cross_entropy_error(self.predict(x), t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_label = np.argmax(y, axis=1)
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        return np.sum(y_label == t_label) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_fn = lambda W: self.loss(x, t)
        return {
            "W1": numerical_gradient(loss_fn, self.params["W1"]),
            "b1": numerical_gradient(loss_fn, self.params["b1"]),
            "W2": numerical_gradient(loss_fn, self.params["W2"]),
            "b2": numerical_gradient(loss_fn, self.params["b2"]),
        }

    def gradient(self, x, t):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        batch_size = x.shape[0]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        dy = y.copy()
        dy[np.arange(batch_size), t_label] -= 1
        dy /= batch_size

        dW2 = np.dot(z1.T, dy)
        db2 = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)
        da1 = dz1 * z1 * (1 - z1)
        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
