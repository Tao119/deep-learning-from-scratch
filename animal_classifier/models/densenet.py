import sys
sys.path.append("../..")
import numpy as np
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization


class DenseLayer:
    """BN-ReLU-Conv(3×3, k out channels) — pre-activation style"""

    def __init__(self, in_ch, growth_rate):
        self.bn = BatchNormalization(np.ones(in_ch), np.zeros(in_ch))
        self.relu = Relu()
        self.conv = Convolution(
            np.sqrt(2.0/(in_ch*9)) * np.random.randn(growth_rate, in_ch, 3, 3),
            np.zeros(growth_rate), stride=1, pad=1)
        self._in_ch = in_ch

    def forward(self, x, train_flg=False):
        out = self.bn.forward(x, train_flg)
        out = self.relu.forward(out)
        return self.conv.forward(out)

    def backward(self, dout):
        dout = self.conv.backward(dout)
        dout = self.relu.backward(dout)
        return self.bn.backward(dout)

    def params_and_grads(self):
        return [self.bn, self.conv]


class DenseBlock:
    """Stack of DenseLayers with dense connections"""

    def __init__(self, in_ch, num_layers, growth_rate):
        self.layers = []
        ch = in_ch
        for _ in range(num_layers):
            self.layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch
        self._all_outputs = None

    def forward(self, x, train_flg=False):
        outputs = [x]
        for layer in self.layers:
            inp = np.concatenate(outputs, axis=1)
            out = layer.forward(inp, train_flg)
            outputs.append(out)
        self._all_outputs = outputs
        return np.concatenate(outputs, axis=1)

    def backward(self, dout):
        outputs = self._all_outputs
        # split dout by channel counts of each saved output
        splits = [o.shape[1] for o in outputs]
        # dgrad[i] = accumulated gradient for outputs[i]
        dgrad = []
        idx = 0
        for s in splits:
            dgrad.append(dout[:, idx:idx+s, :, :].copy())
            idx += s

        # propagate backwards through layers (reverse order)
        for i in range(len(self.layers)-1, -1, -1):
            d = self.layers[i].backward(dgrad[i+1])
            # d is gradient for the concatenated input [outputs[0..i]]
            ch_so_far = [o.shape[1] for o in outputs[:i+1]]
            pos = 0
            for j, cs in enumerate(ch_so_far):
                dgrad[j] += d[:, pos:pos+cs, :, :]
                pos += cs

        return dgrad[0]

    def params_and_grads(self):
        result = []
        for layer in self.layers:
            result.extend(layer.params_and_grads())
        return result


class TransitionLayer:
    """BN-ReLU-Conv(1×1)-AvgPool(2×2)"""

    def __init__(self, in_ch, out_ch):
        self.bn = BatchNormalization(np.ones(in_ch), np.zeros(in_ch))
        self.relu = Relu()
        self.conv = Convolution(
            np.sqrt(2.0/in_ch) * np.random.randn(out_ch, in_ch, 1, 1),
            np.zeros(out_ch), stride=1, pad=0)
        self.pool = Pooling(2, 2, stride=2)

    def forward(self, x, train_flg=False):
        out = self.bn.forward(x, train_flg)
        out = self.relu.forward(out)
        out = self.conv.forward(out)
        return self.pool.forward(out)

    def backward(self, dout):
        dout = self.pool.backward(dout)
        dout = self.conv.backward(dout)
        dout = self.relu.backward(dout)
        return self.bn.backward(dout)

    def params_and_grads(self):
        return [self.bn, self.conv]


class DenseNet:
    """
    DenseNet-40 variant for CIFAR-10 (32×32):
      Init Conv(16) → Block(6, k=12) → Trans → Block(6, k=12) → Trans → Block(6, k=12)
      → BN-ReLU → GlobalAvgPool → FC(10)
    """

    def __init__(self, input_channels=3, output_size=10, growth_rate=12, num_layers=6):
        k = growth_rate
        init_ch = 16
        self.init_conv = Convolution(
            np.sqrt(2.0/(input_channels*9))*np.random.randn(init_ch, input_channels, 3, 3),
            np.zeros(init_ch), stride=1, pad=1)

        self.block1 = DenseBlock(init_ch, num_layers, k)
        t1_in = self.block1.out_channels
        t1_out = t1_in // 2
        self.trans1 = TransitionLayer(t1_in, t1_out)

        self.block2 = DenseBlock(t1_out, num_layers, k)
        t2_in = self.block2.out_channels
        t2_out = t2_in // 2
        self.trans2 = TransitionLayer(t2_in, t2_out)

        self.block3 = DenseBlock(t2_out, num_layers, k)
        final_ch = self.block3.out_channels

        self.final_bn = BatchNormalization(np.ones(final_ch), np.zeros(final_ch))
        self.final_relu = Relu()
        self.fc = Affine(
            np.sqrt(2.0/final_ch)*np.random.randn(final_ch, output_size),
            np.zeros(output_size))
        self.last_layer = SoftmaxWithLoss()
        self._final_ch = final_ch

    def predict(self, x, train_flg=False):
        x = self.init_conv.forward(x)
        x = self.block1.forward(x, train_flg)
        x = self.trans1.forward(x, train_flg)
        x = self.block2.forward(x, train_flg)
        x = self.trans2.forward(x, train_flg)
        x = self.block3.forward(x, train_flg)
        self._last_feat_shape = x.shape[2:]
        x = self.final_relu.forward(self.final_bn.forward(x, train_flg))
        return self.fc.forward(x.mean(axis=(2, 3)))

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x, train_flg=True), t)

    def accuracy(self, x, t, batch_size=32):
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t
        acc, n = 0, (x.shape[0]//batch_size)*batch_size
        for i in range(0, n, batch_size):
            y = self.predict(x[i:i+batch_size], train_flg=False)
            acc += np.sum(np.argmax(y, axis=1) == t_label[i:i+batch_size])
        return acc/n if n > 0 else 0.0

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        dout = self.fc.backward(dout)
        H, W = self._last_feat_shape
        dout = dout[:, :, None, None] * np.ones((1, 1, H, W)) / (H * W)
        dout = self.final_bn.backward(self.final_relu.backward(dout))
        dout = self.block3.backward(dout)
        dout = self.trans2.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.trans1.backward(dout)
        dout = self.block1.backward(dout)
        self.init_conv.backward(dout)
        return {}

    def _get_feat_shape_from_block3(self):
        return self.block3._all_outputs[-1].shape[2:]

    def get_params_and_grads(self):
        params, grads = {}, {}
        idx = 0

        def reg(layer):
            nonlocal idx
            kw, kb = f"p{idx}_W", f"p{idx}_b"
            params[kw] = layer.W if hasattr(layer, "W") else layer.gamma
            params[kb] = layer.b if hasattr(layer, "b") else layer.beta
            grads[kw] = layer.dW if hasattr(layer, "dW") else layer.dgamma
            grads[kb] = layer.db if hasattr(layer, "db") else layer.dbeta
            idx += 1

        reg(self.init_conv)
        for block in [self.block1, self.block2, self.block3]:
            for l in block.params_and_grads():
                reg(l)
        for trans in [self.trans1, self.trans2]:
            for l in trans.params_and_grads():
                reg(l)
        for l in [self.final_bn, self.fc]:
            reg(l)
        return params, grads

    def update(self, optimizer):
        params, grads = self.get_params_and_grads()
        optimizer.update(params, grads)

    def save(self, path):
        import pickle
        params, _ = self.get_params_and_grads()
        try:
            with open(path, "wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            print(f"  [warn] save failed: {e}")
