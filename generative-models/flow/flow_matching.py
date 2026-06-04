"""
Flow Matching (Lipman et al. 2022) — Pure NumPy Implementation
==============================================================
Optimal Transport (OT) conditional flow matching on 2D 8-Gaussian data.

Key ideas:
- Define a probability path from noise p0 = N(0,I) to data p1
- OT conditional flow:  x(t) = (1-t)*x0 + t*x1
- The vector field:     v(x,t) = x1 - x0  (constant along straight paths)
- Train a neural net:   v_theta(x,t) ≈ v(x,t)
- Sampling:             Euler integration dx/dt = v_theta(x,t)

Reference: "Flow Matching for Generative Modeling" (Lipman et al. 2022)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Data: 8 Gaussians arranged on a circle
# ---------------------------------------------------------------------------

def make_8gaussians(n_samples: int, std: float = 0.15, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(42)
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ])
    idx = rng.integers(0, 8, size=n_samples)
    data = centers[idx] + rng.normal(0, std, size=(n_samples, 2))
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: np.ndarray, dim: int = 32) -> np.ndarray:
    """
    t : (B,) float in [0, 1]
    returns (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = np.exp(np.linspace(0, np.log(1000), half)).astype(np.float32)  # (half,)
    args = t[:, None] * freqs[None, :]  # (B, half)
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)  # (B, dim)
    return emb


# ---------------------------------------------------------------------------
# MLP velocity field  v_theta(x, t)
# Architecture: (2 + 32) -> 64 -> 128 -> 64 -> 2
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


class VelocityMLP:
    def __init__(self, input_dim: int = 2, time_emb_dim: int = 32,
                 hidden_dims=(64, 128, 64), output_dim: int = 2,
                 lr: float = 1e-3, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.lr = lr
        self.time_emb_dim = time_emb_dim
        in_dim = input_dim + time_emb_dim
        dims = [in_dim] + list(hidden_dims) + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            w = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray, t: np.ndarray):
        """
        x : (B, 2)
        t : (B,) in [0, 1]
        returns pred (B, 2), cache for backprop
        """
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, 32)
        h = np.concatenate([x, t_emb], axis=1)  # (B, 34)
        cache = {"input": h}
        activations = [h]
        pre_acts = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            pre = activations[-1] @ w + b
            pre_acts.append(pre)
            if i < len(self.weights) - 1:
                activations.append(relu(pre))
            else:
                activations.append(pre)  # linear output
        cache["activations"] = activations
        cache["pre_acts"] = pre_acts
        return activations[-1], cache

    def backward(self, cache: dict, target: np.ndarray):
        """
        MSE loss backward; returns grads and loss value.
        """
        activations = cache["activations"]
        pre_acts = cache["pre_acts"]
        pred = activations[-1]
        batch = pred.shape[0]
        loss = np.mean((pred - target) ** 2)

        # grad of MSE output
        delta = 2.0 * (pred - target) / batch  # (B, 2)

        dw_list = []
        db_list = []
        for i in range(len(self.weights) - 1, -1, -1):
            dw = activations[i].T @ delta
            db = delta.sum(axis=0)
            dw_list.insert(0, dw)
            db_list.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T * relu_grad(pre_acts[i - 1])
            # i==0: we don't need gradient w.r.t. input
        return dw_list, db_list, loss

    def step(self, dw_list, db_list):
        for i, (dw, db) in enumerate(zip(dw_list, db_list)):
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(x, t)
        return pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_flow_matching(n_epochs: int = 2000, batch_size: int = 512,
                        n_data: int = 4096, lr: float = 5e-3,
                        rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(42)

    data = make_8gaussians(n_data, rng=rng)  # (N, 2)
    model = VelocityMLP(lr=lr, rng=rng)
    losses = []

    print(f"Training Flow Matching for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # Sample mini-batch from data (x1) and noise (x0)
        idx = rng.integers(0, n_data, size=batch_size)
        x1 = data[idx]  # (B, 2)
        x0 = rng.normal(0, 1, size=(batch_size, 2)).astype(np.float32)  # (B, 2)
        t = rng.uniform(0, 1, size=batch_size).astype(np.float32)  # (B,)

        # OT conditional flow:  x_t = (1-t)*x0 + t*x1
        t_col = t[:, None]
        x_t = (1 - t_col) * x0 + t_col * x1  # (B, 2)

        # Target vector field: v = x1 - x0  (constant velocity)
        v_target = x1 - x0  # (B, 2)

        pred, cache = model.forward(x_t, t)
        dw, db, loss = model.backward(cache, v_target)
        model.step(dw, db)
        losses.append(loss)

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  loss={loss:.5f}")

    return model, losses


# ---------------------------------------------------------------------------
# Sampling via Euler integration
# ---------------------------------------------------------------------------

def sample_euler(model: VelocityMLP, n_samples: int, n_steps: int,
                 rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(99)
    x = rng.normal(0, 1, size=(n_samples, 2)).astype(np.float32)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step * dt
        t = np.full(n_samples, t_val, dtype=np.float32)
        v = model.predict(x, t)
        x = x + dt * v
    return x


def sample_intermediate(model: VelocityMLP, n_samples: int, n_steps: int,
                        save_times=(0.25, 0.5, 0.75, 1.0),
                        rng: np.random.Generator = None):
    """Return samples at specific integration times."""
    if rng is None:
        rng = np.random.default_rng(99)
    x = rng.normal(0, 1, size=(n_samples, 2)).astype(np.float32)
    dt = 1.0 / n_steps
    snapshots = {}
    for step in range(n_steps):
        t_val = step * dt
        t = np.full(n_samples, t_val, dtype=np.float32)
        v = model.predict(x, t)
        x = x + dt * v
        current_t = (step + 1) * dt
        for ts in save_times:
            if abs(current_t - ts) < dt / 2:
                snapshots[ts] = x.copy()
    return snapshots


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(model, losses, data_ref, save_path: str):
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Flow Matching Results", fontsize=16, fontweight="bold")

    # Row 0: Training loss
    ax = axes[0, 0]
    window = max(1, len(losses) // 100)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(smooth, color="steelblue", linewidth=1.5)
    ax.set_title("Training Loss (smoothed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)

    # Row 0: Reference data
    ax = axes[0, 1]
    ax.scatter(data_ref[:, 0], data_ref[:, 1], s=4, alpha=0.4, color="green")
    ax.set_title("Reference Data (8 Gaussians)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 20 steps vs 100 steps
    s20 = sample_euler(model, 1000, n_steps=20, rng=np.random.default_rng(11))
    s100 = sample_euler(model, 1000, n_steps=100, rng=np.random.default_rng(11))

    ax = axes[0, 2]
    ax.scatter(s20[:, 0], s20[:, 1], s=4, alpha=0.4, color="coral")
    ax.set_title("Euler 20 steps")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    ax.scatter(s100[:, 0], s100[:, 1], s=4, alpha=0.4, color="purple")
    ax.set_title("Euler 100 steps")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Rows 1 & 2: Intermediate states at t=0.25, 0.5, 0.75, 1.0 (two different runs)
    for run_idx in range(2):
        seed = 22 + run_idx * 10
        snaps = sample_intermediate(model, 800, n_steps=100,
                                    save_times=(0.25, 0.5, 0.75, 1.0),
                                    rng=np.random.default_rng(seed))
        for col_idx, ts in enumerate([0.25, 0.5, 0.75, 1.0]):
            ax = axes[1 + run_idx, col_idx]
            pts = snaps.get(ts, np.zeros((2, 2)))
            ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5,
                       color=plt.cm.plasma(ts))
            ax.set_title(f"t = {ts:.2f} (run {run_idx+1})")
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "flow_matching_results.png")

    rng = np.random.default_rng(42)
    model, losses = train_flow_matching(
        n_epochs=2000, batch_size=512, n_data=4096, lr=5e-3, rng=rng
    )

    data_ref = make_8gaussians(1000, rng=np.random.default_rng(1))
    plot_results(model, losses, data_ref, save_path)

    # Quick sanity-check: coverage of 8 modes
    samples = sample_euler(model, 2000, n_steps=100, rng=np.random.default_rng(55))
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ])
    covered = 0
    for c in centers:
        if np.min(np.linalg.norm(samples - c, axis=1)) < 0.5:
            covered += 1
    print(f"\nMode coverage: {covered}/8 Gaussian modes captured")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Results saved to {save_path}")
