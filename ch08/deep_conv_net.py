import sys
sys.path.append("..")
import numpy as np
from collections import OrderedDict
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import Dropout


class DeepConvNet:
    """
    Conv-ReLU-Conv-ReLU-Pool x3
    Affine-ReLU-Dropout
    Affine-Dropout
    Softmax

    フィルター数: 16, 16, 32, 32, 64, 64
    フィルターサイズ: 3x3, He初期値, Adam最適化
    """

    def __init__(self, input_dim=(1, 28, 28), output_size=10):
        pre_node_nums = np.array([1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 3 * 3, 50])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, (out_channel, filter_size, pad) in enumerate(
            [(16, 3, 1), (16, 3, 1), (32, 3, 1), (32, 3, 1), (64, 3, 1), (64, 3, 1)]
        ):
            self.params[f"W{idx+1}"] = weight_init_scales[idx] * np.random.randn(
                out_channel, pre_channel_num, filter_size, filter_size
            )
            self.params[f"b{idx+1}"] = np.zeros(out_channel)
            pre_channel_num = out_channel

        self.params["W7"] = weight_init_scales[6] * np.random.randn(64 * 3 * 3, 50)
        self.params["b7"] = np.zeros(50)
        self.params["W8"] = weight_init_scales[7] * np.random.randn(50, output_size)
        self.params["b8"] = np.zeros(output_size)

        p = self.params
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(p["W1"], p["b1"], stride=1, pad=1)
        self.layers["Relu1"] = Relu()
        self.layers["Conv2"] = Convolution(p["W2"], p["b2"], stride=1, pad=1)
        self.layers["Relu2"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Conv3"] = Convolution(p["W3"], p["b3"], stride=1, pad=1)
        self.layers["Relu3"] = Relu()
        self.layers["Conv4"] = Convolution(p["W4"], p["b4"], stride=1, pad=1)
        self.layers["Relu4"] = Relu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Conv5"] = Convolution(p["W5"], p["b5"], stride=1, pad=1)
        self.layers["Relu5"] = Relu()
        self.layers["Conv6"] = Convolution(p["W6"], p["b6"], stride=1, pad=1)
        self.layers["Relu6"] = Relu()
        self.layers["Pool3"] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers["Affine1"] = Affine(p["W7"], p["b7"])
        self.layers["Relu7"] = Relu()
        self.layers["Dropout1"] = Dropout(dropout_ratio=0.5)
        self.layers["Affine2"] = Affine(p["W8"], p["b8"])
        self.layers["Dropout2"] = Dropout(dropout_ratio=0.5)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=100):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc = 0.0
        for i in range(x.shape[0] // batch_size):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t_label[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        for i, layer_name in enumerate(
            ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "Conv6", "Affine1", "Affine2"]
        ):
            grads[f"W{i+1}"] = self.layers[layer_name].dW
            grads[f"b{i+1}"] = self.layers[layer_name].db

        return grads

    def save_params(self, path="deep_conv_net_params.pkl"):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, path="deep_conv_net_params.pkl"):
        import pickle
        with open(path, "rb") as f:
            self.params = pickle.load(f)
        for i, layer_name in enumerate(
            ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "Conv6", "Affine1", "Affine2"]
        ):
            self.layers[layer_name].W = self.params[f"W{i+1}"]
            self.layers[layer_name].b = self.params[f"b{i+1}"]
