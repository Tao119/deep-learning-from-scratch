import sys
sys.path.append("../..")
import numpy as np
from common.conv_layers import Convolution, Pooling
from common.layers import Affine, Relu, SoftmaxWithLoss
from common.layers_ext import BatchNormalization


class SEBlock:
    """Squeeze-and-Excitation: channel-wise attention"""

    def __init__(self, channels, reduction=16):
        r = max(channels // reduction, 1)
        self.fc1 = Affine(
            np.sqrt(2.0 / channels) * np.random.randn(channels, r),
            np.zeros(r)
        )
        self.fc2 = Affine(
            np.sqrt(2.0 / r) * np.random.randn(r, channels),
            np.zeros(channels)
        )
        self.relu = Relu()
        self._shape = None

    def forward(self, x):
        self._shape = x.shape
        N, C, H, W = x.shape
        z = x.mean(axis=(2, 3))           # squeeze: (N, C)
        z = self.relu.forward(self.fc1.forward(z))
        s = _sigmoid(self.fc2.forward(z)) # excitation: (N, C)
        self._s = s
        return x * s[:, :, None, None]

    def backward(self, dout):
        N, C, H, W = self._shape
        ds_scaled = (dout * self._s[:, :, None, None]).sum(axis=(2, 3))  # NOT right - need chain rule
        # gradient w.r.t. x: s * dout
        dx = dout * self._s[:, :, None, None]
        # gradient w.r.t. s: dout * x_orig
        # we need x before scaling - store it
        ds = (dout * self._x_before_scale).sum(axis=(2, 3))  # (N, C)
        # sigmoid backward
        ds = ds * self._s * (1 - self._s)
        ds = self.fc2.backward(ds)
        ds = self.relu.backward(ds)
        ds = self.fc1.backward(ds)
        return dx

    def params_and_grads(self):
        return [self.fc1, self.fc2]


def _sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SEResidualBlock:
    """ResidualBlock + SE attention"""

    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        scale = np.sqrt(2.0 / (in_ch * 9))
        self.conv1 = Convolution(
            scale * np.random.randn(out_ch, in_ch, 3, 3), np.zeros(out_ch), stride=stride, pad=1)
        self.bn1 = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))
        self.relu1 = Relu()
        self.conv2 = Convolution(
            np.sqrt(2.0 / (out_ch * 9)) * np.random.randn(out_ch, out_ch, 3, 3),
            np.zeros(out_ch), stride=1, pad=1)
        self.bn2 = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))
        self.relu2 = Relu()

        r = max(out_ch // reduction, 1)
        self.se_fc1 = Affine(np.sqrt(2.0/out_ch)*np.random.randn(out_ch, r), np.zeros(r))
        self.se_fc2 = Affine(np.sqrt(2.0/r)*np.random.randn(r, out_ch), np.zeros(out_ch))
        self.se_relu = Relu()

        self.use_projection = (in_ch != out_ch or stride != 1)
        if self.use_projection:
            self.proj = Convolution(
                np.sqrt(2.0/in_ch)*np.random.randn(out_ch, in_ch, 1, 1),
                np.zeros(out_ch), stride=stride, pad=0)
            self.proj_bn = BatchNormalization(np.ones(out_ch), np.zeros(out_ch))

    def _se_forward(self, x):
        N, C, H, W = x.shape
        z = x.mean(axis=(2, 3))
        z = self.se_relu.forward(self.se_fc1.forward(z))
        s = _sigmoid(self.se_fc2.forward(z))
        self._se_s = s
        self._se_x = x
        return x * s[:, :, None, None]

    def _se_backward(self, dout):
        s = self._se_s
        x = self._se_x
        N, C, H, W = x.shape
        dx = dout * s[:, :, None, None]
        ds = (dout * x).sum(axis=(2, 3)) * s * (1 - s)
        ds = self.se_fc2.backward(ds)
        ds = self.se_relu.backward(ds)
        self.se_fc1.backward(ds)
        return dx

    def forward(self, x, train_flg=False):
        self._x = x
        residual = x
        if self.use_projection:
            residual = self.proj_bn.forward(self.proj.forward(x), train_flg)
        self._residual = residual

        out = self.conv1.forward(x)
        out = self.bn1.forward(out, train_flg)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, train_flg)
        out = self._se_forward(out)
        out += residual
        return self.relu2.forward(out)

    def backward(self, dout):
        dout = self.relu2.backward(dout)
        dresidual = dout.copy()
        dout = self._se_backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        if self.use_projection:
            dresidual = self.proj_bn.backward(dresidual)
            dresidual = self.proj.backward(dresidual)
        return dout + dresidual

    def params_and_grads(self):
        layers = [self.conv1, self.bn1, self.conv2, self.bn2, self.se_fc1, self.se_fc2]
        if self.use_projection:
            layers += [self.proj, self.proj_bn]
        return layers


class SEResNet:
    """ResNet + Squeeze-and-Excitation blocks"""

    def __init__(self, input_channels=3, output_size=10):
        self.stem_conv = Convolution(
            np.sqrt(2.0/(input_channels*9))*np.random.randn(64, input_channels, 3, 3),
            np.zeros(64), stride=1, pad=1)
        self.stem_bn = BatchNormalization(np.ones(64), np.zeros(64))
        self.stem_relu = Relu()

        self.blocks = [
            SEResidualBlock(64, 64),
            SEResidualBlock(64, 64),
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128),
            SEResidualBlock(128, 256, stride=2),
            SEResidualBlock(256, 256),
        ]

        self.fc = Affine(np.sqrt(2.0/256)*np.random.randn(256, output_size), np.zeros(output_size))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        x = self.stem_relu.forward(self.stem_bn.forward(self.stem_conv.forward(x), train_flg))
        for block in self.blocks:
            x = block.forward(x, train_flg)
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
        H, W = self.blocks[-1]._x.shape[2:]
        dout = dout[:, :, None, None] * np.ones((1, 1, H, W)) / (H * W)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        self.stem_conv.backward(dout)
        return {}

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

        for l in [self.stem_conv, self.stem_bn]:
            reg(l)
        for block in self.blocks:
            for l in block.params_and_grads():
                reg(l)
        reg(self.fc)
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
