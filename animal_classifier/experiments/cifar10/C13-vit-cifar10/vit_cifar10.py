"""
C13: Vision Transformer (ViT) for CIFAR-10
Pure NumPy implementation — consistent with the project's from-scratch style.

Architecture
────────────
Input : (N, 3, 32, 32)

1. Patch Embedding
   patch_size = 4  →  (32/4)² = 64 patches
   Each patch: 4×4×3 = 48 pixels → Linear(48→64=d_model)
   Result: (N, 64, 64)

2. CLS token prepend → (N, 65, 64)

3. Learnable positional embedding: (65, 64)

4. 4× TransformerEncoderBlock
   - Pre-LN MultiHeadSelfAttention (4 heads, head_dim=16)
   - Pre-LN FFN: Linear(64→256) → GELU → Linear(256→64)
   - Residual connections throughout

5. CLS token → Linear(64→10) → SoftmaxWithLoss

Training
────────
  Adam, lr=1e-3, 30 epochs, batch=128
  Mild augmentation (flip + crop pad=2)
  Log format: "epoch X/30: loss=Y.YYYY  train=Z.ZZZZ  test=W.WWWW  (Ns)"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animal_classifier.dataset.cifar10 import load_cifar10, normalize, CLASS_NAMES
from animal_classifier.dataset.augmentation import batch_mild_augment
from common.optimizer import Adam


# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE   = 4
IMG_SIZE     = 32
NUM_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2     # 64
PATCH_DIM    = PATCH_SIZE * PATCH_SIZE * 3        # 48
D_MODEL      = 64
N_HEADS      = 4
HEAD_DIM     = D_MODEL // N_HEADS                 # 16
N_LAYERS     = 4
D_FF         = 256
NUM_CLASSES  = 10
EPOCHS       = 30
BATCH_SIZE   = 128
LR           = 1e-3
LR_MIN       = 1e-5
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Activation
# ─────────────────────────────────────────────────────────────────────────────

def gelu(x):
    """GELU activation: x * Φ(x) approximated via tanh."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(tanh_arg)
    dtanh = (1.0 - tanh_val ** 2) * np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, t):
    """logits: (N, C), t: (N,) or (N, C)."""
    N = logits.shape[0]
    y = softmax(logits)
    if t.ndim == 1:
        loss = -np.log(y[np.arange(N), t] + 1e-7).mean()
        dy = y.copy()
        dy[np.arange(N), t] -= 1.0
        return loss, dy / N
    else:
        loss = -(t * np.log(y + 1e-7)).sum(axis=1).mean()
        dy = (y - t) / N
        return loss, dy


# ─────────────────────────────────────────────────────────────────────────────
#  Layer-Norm (pure NumPy, backprop included)
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm:
    def __init__(self, d: int):
        self.gamma = np.ones(d, dtype=np.float32)
        self.beta  = np.zeros(d, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self._cache = None

    def forward(self, x):
        mu  = x.mean(axis=-1, keepdims=True)
        xc  = x - mu
        var = (xc ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-6)
        xn  = xc / std
        self._cache = (xc, std, xn, x.shape)
        return self.gamma * xn + self.beta

    def backward(self, dout):
        xc, std, xn, shape = self._cache
        D = shape[-1]
        self.dgamma = (dout * xn).reshape(-1, D).sum(axis=0)
        self.dbeta  = dout.reshape(-1, D).sum(axis=0)

        dxn   = dout * self.gamma
        dvar  = (-0.5 * dxn * xc / std ** 3).sum(axis=-1, keepdims=True)
        dmu   = (-dxn / std).sum(axis=-1, keepdims=True) + dvar * (-2 * xc).mean(axis=-1, keepdims=True)
        dx    = dxn / std + 2 * dvar * xc / D + dmu / D
        return dx


# ─────────────────────────────────────────────────────────────────────────────
#  Linear layer
# ─────────────────────────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        scale = np.sqrt(2.0 / in_dim)
        self.W = scale * np.random.randn(in_dim, out_dim).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, dout):
        x = self._x
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        d_2d = dout.reshape(-1, dout.shape[-1])
        self.dW = x_2d.T @ d_2d
        if self.b is not None:
            self.db = d_2d.sum(axis=0)
        dx = (d_2d @ self.W.T).reshape(orig_shape)
        return dx


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Head Self-Attention (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention:
    """
    Multi-head self-attention.
    Input  : (N, T, d_model)
    Output : (N, T, d_model)

    Implemented as a single large projection then reshaped to heads.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

        # Caches for backward
        self._q = self._k = self._v = None
        self._attn = None
        self._N = self._T = None

    def forward(self, x):
        N, T, D = x.shape
        h = self.n_heads
        hd = self.head_dim

        q = self.Wq.forward(x).reshape(N, T, h, hd).transpose(0, 2, 1, 3)  # (N,h,T,hd)
        k = self.Wk.forward(x).reshape(N, T, h, hd).transpose(0, 2, 1, 3)
        v = self.Wv.forward(x).reshape(N, T, h, hd).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale   # (N,h,T,T)
        attn   = softmax(scores)                                # (N,h,T,T)

        ctx = attn @ v                                          # (N,h,T,hd)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(N, T, D)       # (N,T,D)

        out = self.Wo.forward(ctx)

        self._q    = q
        self._k    = k
        self._v    = v
        self._attn = attn
        self._ctx  = ctx
        self._x    = x
        self._N    = N
        self._T    = T
        return out

    def backward(self, dout):
        N, T, D = dout.shape
        h, hd = self.n_heads, self.head_dim

        # Backward through Wo
        dctx = self.Wo.backward(dout)                          # (N,T,D)
        # reshape ctx → (N,h,T,hd)
        dctx_h = dctx.reshape(N, T, h, hd).transpose(0, 2, 1, 3)

        # Backward through attn @ v
        dattn = dctx_h @ self._v.transpose(0, 1, 3, 2)        # (N,h,T,T)
        dv    = self._attn.transpose(0, 1, 3, 2) @ dctx_h     # (N,h,T,hd)

        # Backward through softmax
        # d(softmax) via sum-trick
        ds = self._attn * (dattn - (dattn * self._attn).sum(axis=-1, keepdims=True))
        ds = ds * self.scale                                    # (N,h,T,T)

        # Backward through q @ k.T
        dq = ds @ self._k                                      # (N,h,T,hd)
        dk = ds.transpose(0, 1, 3, 2) @ self._q               # (N,h,T,hd)

        # Reshape back
        dq = dq.transpose(0, 2, 1, 3).reshape(N, T, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(N, T, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(N, T, D)

        dx  = self.Wq.backward(dq)
        dx += self.Wk.backward(dk)
        dx += self.Wv.backward(dv)
        return dx


# ─────────────────────────────────────────────────────────────────────────────
#  Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FFN:
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self._a1 = None   # pre-activation
        self._h1 = None   # post-activation

    def forward(self, x):
        a1 = self.fc1.forward(x)
        h1 = gelu(a1)
        self._a1 = a1
        self._h1 = h1
        return self.fc2.forward(h1)

    def backward(self, dout):
        dh1 = self.fc2.backward(dout)
        da1 = dh1 * gelu_grad(self._a1)
        return self.fc1.backward(da1)


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Encoder Block (pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock:
    def __init__(self):
        self.ln1  = LayerNorm(D_MODEL)
        self.attn = MultiHeadSelfAttention()
        self.ln2  = LayerNorm(D_MODEL)
        self.ffn  = FFN()
        self._x1 = None
        self._x2 = None

    def forward(self, x):
        # Attention sub-layer (pre-LN)
        self._x1 = x
        h1 = x + self.attn.forward(self.ln1.forward(x))

        # FFN sub-layer (pre-LN)
        self._x2 = h1
        out = h1 + self.ffn.forward(self.ln2.forward(h1))
        return out

    def backward(self, dout):
        # FFN sub-layer backward
        dln2 = self.ln2.backward(self.ffn.backward(dout))
        dh1  = dout + dln2

        # Attention sub-layer backward
        dln1 = self.ln1.backward(self.attn.backward(dh1))
        dx   = dh1 + dln1
        return dx


# ─────────────────────────────────────────────────────────────────────────────
#  Vision Transformer
# ─────────────────────────────────────────────────────────────────────────────

class ViT:
    """
    Vision Transformer for CIFAR-10.
    Input: (N, 3, 32, 32)  — NCHW format, values in [0,1]
    """

    def __init__(self):
        # Patch embedding: (N, 64, 48) → (N, 64, 64)
        self.patch_proj  = Linear(PATCH_DIM, D_MODEL)

        # Learnable [CLS] token and positional embedding
        self.cls_token = np.zeros((1, 1, D_MODEL), dtype=np.float32)
        self.pos_emb   = (np.random.randn(1, NUM_PATCHES + 1, D_MODEL) * 0.02).astype(np.float32)
        self.dcls      = np.zeros_like(self.cls_token)
        self.dpos      = np.zeros_like(self.pos_emb)

        # Transformer blocks
        self.blocks = [TransformerBlock() for _ in range(N_LAYERS)]

        # Final LayerNorm + head
        self.ln_final = LayerNorm(D_MODEL)
        self.head      = Linear(D_MODEL, NUM_CLASSES)

        # Last loss
        self.loss_val = 0.0
        self._dy      = None
        self._N       = 0

    # ── Patch extraction ─────────────────────────────────────────────────────

    @staticmethod
    def extract_patches(x):
        """
        x      : (N, 3, 32, 32)
        return : (N, 64, 48)  — 64 patches, 48 pixels each
        """
        N, C, H, W = x.shape
        ph = pw = PATCH_SIZE
        nh = H // ph
        nw = W // pw
        # (N, C, nh, ph, nw, pw)
        x_r = x.reshape(N, C, nh, ph, nw, pw)
        # (N, nh, nw, ph, pw, C)
        x_r = x_r.transpose(0, 2, 4, 3, 5, 1)
        # (N, nh*nw, ph*pw*C)
        patches = x_r.reshape(N, nh * nw, ph * pw * C)
        return patches.astype(np.float32)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x, train_flg=False):
        N = x.shape[0]
        self._N = N

        # Patches → embeddings: (N, 64, 64)
        patches = self.extract_patches(x)         # (N, 64, 48)
        emb     = self.patch_proj.forward(patches) # (N, 64, 64)

        # Prepend CLS token
        cls = np.broadcast_to(self.cls_token, (N, 1, D_MODEL)).copy()
        tokens = np.concatenate([cls, emb], axis=1)  # (N, 65, 64)
        self._cls_before = cls

        # Add positional embedding
        tokens = tokens + self.pos_emb
        self._tokens_before_blocks = tokens

        # Transformer blocks
        for block in self.blocks:
            tokens = block.forward(tokens)

        self._tokens_after_blocks = tokens

        # Final LN + extract CLS
        tokens_norm = self.ln_final.forward(tokens)
        cls_out     = tokens_norm[:, 0, :]       # (N, 64)
        self._cls_out = cls_out

        # Classification logits
        logits = self.head.forward(cls_out)      # (N, 10)
        return logits

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(self, x, t):
        logits = self.forward(x, train_flg=True)
        loss, dy = softmax_cross_entropy(logits, t)
        self.loss_val = loss
        self._dy = dy
        return loss

    # ── Accuracy ─────────────────────────────────────────────────────────────

    def accuracy(self, x, t, batch_size=128):
        t_lbl = t if t.ndim == 1 else np.argmax(t, axis=1)
        n = (x.shape[0] // batch_size) * batch_size
        correct = 0
        for i in range(0, n, batch_size):
            logits = self.forward(x[i:i+batch_size], train_flg=False)
            correct += np.sum(np.argmax(logits, axis=1) == t_lbl[i:i+batch_size])
        return correct / n if n > 0 else 0.0

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self):
        N = self._N
        dy = self._dy        # (N, 10)

        # Head backward
        d_cls = self.head.backward(dy)        # (N, 64)

        # Expand to token sequence: only CLS position gets gradient
        d_tokens = np.zeros((N, NUM_PATCHES + 1, D_MODEL), dtype=np.float32)
        d_tokens[:, 0, :] = d_cls

        # Final LN backward
        d_tokens = self.ln_final.backward(d_tokens)

        # Positional embedding gradient
        self.dpos = d_tokens.sum(axis=0, keepdims=True)    # (1, 65, 64)

        # Transformer blocks (reverse)
        for block in reversed(self.blocks):
            d_tokens = block.backward(d_tokens)

        # CLS token gradient
        self.dcls = d_tokens[:, 0:1, :].sum(axis=0, keepdims=True)  # (1,1,64)

        # Patch embedding backward
        d_emb    = d_tokens[:, 1:, :]          # (N, 64, 64)
        self.patch_proj.backward(d_emb)         # updates dW/db inside

    # ── Collect params/grads for Adam ────────────────────────────────────────

    def get_params_and_grads(self):
        params, grads = {}, {}
        idx = [0]

        def reg(arr, darr):
            k = f"p{idx[0]}"
            params[k] = arr
            grads[k]  = darr
            idx[0] += 1

        # Patch projection
        reg(self.patch_proj.W, self.patch_proj.dW)
        reg(self.patch_proj.b, self.patch_proj.db)

        # CLS token and positional embedding
        reg(self.cls_token, self.dcls)
        reg(self.pos_emb,   self.dpos)

        # Transformer blocks
        for block in self.blocks:
            # LN1
            reg(block.ln1.gamma, block.ln1.dgamma)
            reg(block.ln1.beta,  block.ln1.dbeta)
            # Attention projections
            for proj in [block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wo]:
                reg(proj.W, proj.dW)
                reg(proj.b, proj.db)
            # LN2
            reg(block.ln2.gamma, block.ln2.dgamma)
            reg(block.ln2.beta,  block.ln2.dbeta)
            # FFN
            reg(block.ffn.fc1.W, block.ffn.fc1.dW)
            reg(block.ffn.fc1.b, block.ffn.fc1.db)
            reg(block.ffn.fc2.W, block.ffn.fc2.dW)
            reg(block.ffn.fc2.b, block.ffn.fc2.db)

        # Final LN
        reg(self.ln_final.gamma, self.ln_final.dgamma)
        reg(self.ln_final.beta,  self.ln_final.dbeta)

        # Head
        reg(self.head.W, self.head.dW)
        reg(self.head.b, self.head.db)

        return params, grads

    def update(self, optimizer):
        params, grads = self.get_params_and_grads()
        optimizer.update(params, grads)

    def save(self, path):
        params, _ = self.get_params_and_grads()
        try:
            with open(path, "wb") as f:
                pickle.dump(params, f)
        except OSError as e:
            print(f"  [warn] save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(epoch, T=EPOCHS, lr_max=LR, lr_min=LR_MIN):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))


def _accuracy_quick(model, x, t, n=500):
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return model.accuracy(x[idx], t[idx])


def main():
    log_path = os.path.join(OUTPUT_DIR, "train.log")
    best_path = os.path.join(OUTPUT_DIR, "ViT_cifar10_best.pkl")

    print("Loading CIFAR-10 ...")
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    x_train, x_test, _, _ = normalize(x_train, x_test)
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    model     = ViT()
    optimizer = Adam(lr=cosine_lr(0))

    train_size     = len(x_train)
    iter_per_epoch = max(train_size // BATCH_SIZE, 1)
    best_test_acc  = 0.0

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"C13 - Vision Transformer / CIFAR-10")
    log(f"patches={NUM_PATCHES}, d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF}")
    log(f"epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}")
    log("")

    for epoch in range(EPOCHS):
        lr = cosine_lr(epoch)
        optimizer.lr = lr

        epoch_loss = 0.0
        t0 = time.time()
        perm = np.random.permutation(train_size)

        for i in range(iter_per_epoch):
            bi   = perm[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x_b  = batch_mild_augment(x_train[bi], train_flg=True)
            t_b  = t_train[bi]
            loss = model.loss(x_b, t_b)
            model.backward()
            model.update(optimizer)
            epoch_loss += loss

        epoch_loss /= iter_per_epoch
        tr_acc = _accuracy_quick(model, x_train, t_train)
        te_acc = _accuracy_quick(model, x_test,  t_test)
        elapsed = time.time() - t0

        msg = (f"epoch {epoch+1:3d}/{EPOCHS}: "
               f"loss={epoch_loss:.4f}  "
               f"train={tr_acc:.4f}  "
               f"test={te_acc:.4f}  "
               f"({elapsed:.0f}s)")
        log(msg)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            model.save(best_path)

    log(f"\nBest test accuracy: {best_test_acc:.4f}")

    # Write log
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nLog saved: {log_path}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
