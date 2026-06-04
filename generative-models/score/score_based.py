"""
Score-Based Generative Model (SMLD) — Pure NumPy Implementation
===============================================================
Implements the Score Matching with Langevin Dynamics (SMLD) framework
as described in Song & Ermon (2019) "Generative Modeling by Estimating
Gradients of the Data Distribution".

Algorithm:
1. Define a noise schedule σ_1 < σ_2 < ... < σ_L  (geometric)
2. Perturb data: x_tilde = x + sigma * noise
3. Score network s_θ(x_tilde, σ) learns ∇log p_σ(x_tilde) ≈ -noise/σ
4. Loss: Σ_l λ(σ_l) * E[||s_θ(x_tilde, σ_l) + noise/σ_l²||²]
5. Sampling: Annealed Langevin dynamics across σ schedule
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_8gaussians(n_samples: int, std: float = 0.15,
                    rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ], dtype=np.float32)
    idx = rng.integers(0, 8, size=n_samples)
    data = centers[idx] + rng.normal(0, std, size=(n_samples, 2)).astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# Noise schedule: geometric from σ_min to σ_max
# ---------------------------------------------------------------------------

def make_sigma_schedule(sigma_min: float = 0.01, sigma_max: float = 5.0,
                        L: int = 10) -> np.ndarray:
    """Geometric noise schedule: σ_l = σ_min * (σ_max/σ_min)^(l/(L-1))"""
    return np.geomspace(sigma_min, sigma_max, L).astype(np.float32)


# ---------------------------------------------------------------------------
# Score Network: MLP(3 → 64 → 128 → 128 → 64 → 2)
# Input: [x (2D), log(σ) (1D)] = 3D
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


class ScoreNetwork:
    def __init__(self, lr: float = 1e-3, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.lr = lr
        dims = [3, 64, 128, 128, 64, 2]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            w = rng.normal(0, np.sqrt(2.0 / fan_in),
                           (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x_noisy: np.ndarray, sigma: np.ndarray):
        """
        x_noisy : (B, 2)
        sigma   : (B,) noise levels
        returns : (B, 2) predicted score, cache
        """
        log_sigma = np.log(sigma[:, None])  # (B, 1)
        h = np.concatenate([x_noisy, log_sigma], axis=1)  # (B, 3)
        activations = [h]
        pre_acts = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            pre = activations[-1] @ w + b
            pre_acts.append(pre)
            if i < len(self.weights) - 1:
                activations.append(relu(pre))
            else:
                activations.append(pre)
        return activations[-1], {"activations": activations, "pre_acts": pre_acts}

    def backward(self, cache: dict, target: np.ndarray):
        activations = cache["activations"]
        pre_acts = cache["pre_acts"]
        pred = activations[-1]
        batch = pred.shape[0]
        loss = np.mean((pred - target) ** 2)
        delta = 2.0 * (pred - target) / batch
        dw_list, db_list = [], []
        for i in range(len(self.weights) - 1, -1, -1):
            dw = activations[i].T @ delta
            db = delta.sum(axis=0)
            dw_list.insert(0, dw)
            db_list.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T * relu_grad(pre_acts[i - 1])
        return dw_list, db_list, loss

    def step(self, dw_list, db_list):
        for i, (dw, db) in enumerate(zip(dw_list, db_list)):
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def predict(self, x_noisy: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x_noisy, sigma)
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_score_model(n_epochs: int = 3000, batch_size: int = 512,
                      n_data: int = 4096, lr: float = 3e-3,
                      L: int = 10, sigma_min: float = 0.01,
                      sigma_max: float = 5.0,
                      rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(42)

    data = make_8gaussians(n_data, rng=rng)
    sigmas = make_sigma_schedule(sigma_min, sigma_max, L)
    model = ScoreNetwork(lr=lr, rng=rng)
    losses = []

    print(f"Training Score-Based Model for {n_epochs} epochs, L={L} noise levels...")
    print(f"  σ_min={sigma_min}, σ_max={sigma_max}")

    for epoch in range(n_epochs):
        # Sample data
        idx = rng.integers(0, n_data, size=batch_size)
        x = data[idx]  # (B, 2)

        # Sample random noise level index for each sample
        level_idx = rng.integers(0, L, size=batch_size)
        sigma = sigmas[level_idx]  # (B,)

        # Add noise: x_tilde = x + sigma * eps
        eps = rng.normal(0, 1, size=(batch_size, 2)).astype(np.float32)
        x_noisy = x + sigma[:, None] * eps  # (B, 2)

        # Target score: ∇log p_σ(x_tilde | x) = -eps/σ
        target_score = -eps / sigma[:, None]  # (B, 2)

        # Loss weighting λ(σ) = σ² (as in original paper)
        # The loss term becomes: σ² * ||s_θ + eps/σ||² = ||σ*s_θ + eps||²
        # We weight the MSE by σ² implicitly via target scaling
        # Actually we use: loss = mean(σ²) * MSE(s_θ, -eps/σ)
        # Simpler: weight each sample by sigma_l^2
        pred, cache = model.forward(x_noisy, sigma)

        # Weighted residual for loss, but backprop on raw MSE for stability
        dw, db, raw_loss = model.backward(cache, target_score)

        # Compute proper weighted loss for logging
        residual = pred - target_score  # (B, 2)
        weighted_loss = float(np.mean(sigma[:, None] ** 2 * residual ** 2))

        model.step(dw, db)
        losses.append(weighted_loss)

        if (epoch + 1) % 300 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  weighted_loss={weighted_loss:.5f}")

    return model, sigmas, losses


# ---------------------------------------------------------------------------
# Annealed Langevin Dynamics Sampling
# ---------------------------------------------------------------------------

def annealed_langevin(model: ScoreNetwork, sigmas: np.ndarray,
                      n_samples: int = 500, n_steps_per_level: int = 100,
                      eps_anneal: float = 2e-4,
                      rng: np.random.Generator = None,
                      clip_radius: float = 6.0) -> np.ndarray:
    """
    Annealed Langevin dynamics across the noise schedule (coarse → fine).
    At each σ_l, run T steps of:
        x_{t+1} = x_t + α_l * s_θ(x_t, σ_l) + sqrt(2*α_l)*z_t
    where α_l = eps_anneal * (σ_l / σ_1)²

    clip_radius clips the score magnitude to prevent numerical overflow.
    """
    if rng is None:
        rng = np.random.default_rng(99)

    # Start from noise proportional to largest sigma
    x = rng.normal(0, sigmas[-1], size=(n_samples, 2)).astype(np.float32)

    # Anneal from large sigma to small
    for sigma in reversed(sigmas):
        alpha = float(eps_anneal * (sigma / sigmas[0]) ** 2)
        alpha = min(alpha, 1e-2)  # cap step size for stability
        sigma_arr = np.full(n_samples, sigma, dtype=np.float32)
        for _ in range(n_steps_per_level):
            score = model.predict(x, sigma_arr)  # (N, 2)
            # Clip score magnitude to prevent exploding steps
            score_norm = np.linalg.norm(score, axis=1, keepdims=True).clip(min=1e-8)
            score_clipped = np.where(
                score_norm > clip_radius,
                score / score_norm * clip_radius,
                score,
            )
            noise = rng.normal(0, 1, size=x.shape).astype(np.float32)
            x = x + alpha * score_clipped + np.sqrt(2 * alpha) * noise
            # Clip x to a reasonable range to prevent runaway
            x = np.clip(x, -10.0, 10.0)
    return x


# ---------------------------------------------------------------------------
# Score field visualization
# ---------------------------------------------------------------------------

def visualize_score_field(model: ScoreNetwork, sigma: float, ax, title: str,
                          grid_range=(-2.5, 2.5), n_grid: int = 20):
    xs = np.linspace(*grid_range, n_grid)
    ys = np.linspace(*grid_range, n_grid)
    xv, yv = np.meshgrid(xs, ys)
    grid = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)
    sigma_arr = np.full(grid.shape[0], sigma, dtype=np.float32)
    scores = model.predict(grid, sigma_arr)  # (G, 2)

    # Normalize arrows for visibility
    norms = np.linalg.norm(scores, axis=1, keepdims=True).clip(min=1e-6)
    scores_unit = scores / norms

    ax.quiver(grid[:, 0], grid[:, 1],
              scores_unit[:, 0], scores_unit[:, 1],
              norms.ravel(), cmap="viridis", alpha=0.7,
              scale=25, width=0.005)
    ax.set_xlim(*grid_range)
    ax.set_ylim(*grid_range)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(model, sigmas, losses, data_ref, save_path: str):
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle("Score-Based Generative Model (SMLD) Results", fontsize=16, fontweight="bold")

    # Training loss
    ax = axes[0, 0]
    window = max(1, len(losses) // 80)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(smooth, color="steelblue", linewidth=1.5)
    ax.set_title("Training Loss (weighted, smoothed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)

    # Reference data
    ax = axes[0, 1]
    ax.scatter(data_ref[:, 0], data_ref[:, 1], s=5, alpha=0.5, color="green")
    ax.set_title("Reference Data (8 Gaussians)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Generated samples
    samples = annealed_langevin(model, sigmas, n_samples=800,
                                n_steps_per_level=80, eps_anneal=1e-3,
                                rng=np.random.default_rng(7))
    ax = axes[0, 2]
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="coral")
    ax.set_title("Generated Samples (Annealed Langevin)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Noise schedule
    ax = axes[0, 3]
    ax.semilogy(range(len(sigmas)), sigmas, "o-", color="purple")
    ax.set_title("Noise Schedule σ")
    ax.set_xlabel("Level index")
    ax.set_ylabel("σ")
    ax.grid(True, alpha=0.4)

    # Score field at various sigma levels
    sigma_vis = [sigmas[-1], sigmas[len(sigmas) // 2], sigmas[1], sigmas[0]]
    sigma_labels = [f"σ={s:.3f}" for s in sigma_vis]
    positions = [(1, 0), (1, 1), (1, 2), (1, 3)]
    for (r, c), sigma_v, label in zip(positions, sigma_vis, sigma_labels):
        visualize_score_field(model, float(sigma_v), axes[r, c],
                              f"Score Field ({label})")
        # Overlay reference data lightly
        axes[r, c].scatter(data_ref[:200, 0], data_ref[:200, 1],
                           s=3, alpha=0.3, color="red", zorder=5)

    # Langevin trajectory snapshots at different sigma steps
    def langevin_snapshots(model, sigmas, n_samples, n_steps_per_level,
                           eps_anneal, rng, save_sigmas, clip_radius=6.0):
        rng_inner = np.random.default_rng(rng.integers(1 << 30))
        x = rng_inner.normal(0, sigmas[-1], size=(n_samples, 2)).astype(np.float32)
        snaps = {}
        sigma_reversed = list(reversed(sigmas))
        for si, sigma in enumerate(sigma_reversed):
            alpha = float(min(eps_anneal * (sigma / sigmas[0]) ** 2, 1e-2))
            sigma_arr = np.full(n_samples, sigma, dtype=np.float32)
            for _ in range(n_steps_per_level):
                score = model.predict(x, sigma_arr)
                score_norm = np.linalg.norm(score, axis=1, keepdims=True).clip(min=1e-8)
                score = np.where(score_norm > clip_radius, score / score_norm * clip_radius, score)
                noise = rng_inner.normal(0, 1, size=x.shape).astype(np.float32)
                x = x + alpha * score + np.sqrt(2 * alpha) * noise
                x = np.clip(x, -10.0, 10.0)
            for sv in save_sigmas:
                if abs(sigma - sv) < 1e-6:
                    snaps[sigma] = x.copy()
        return snaps

    snap_sigmas = [sigmas[-1], sigmas[len(sigmas) * 3 // 4],
                   sigmas[len(sigmas) // 4], sigmas[0]]
    snaps = langevin_snapshots(model, sigmas, 500, 80, 1e-3,
                               np.random.default_rng(33), snap_sigmas)
    snap_titles = ["Start (σ_max)", "After 25%", "After 75%", "Final (σ_min)"]
    for col, (sv, title) in enumerate(zip(snap_sigmas, snap_titles)):
        ax = axes[2, col]
        pts = snaps.get(sv, np.zeros((2, 2)))
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6,
                   color=plt.cm.coolwarm(col / 3.0))
        ax.set_title(f"Langevin: {title}")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
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
    save_path = os.path.join(base_dir, "score_based_results.png")

    rng = np.random.default_rng(42)
    model, sigmas, losses = train_score_model(
        n_epochs=3000, batch_size=512, n_data=4096,
        lr=3e-3, L=10, sigma_min=0.01, sigma_max=5.0, rng=rng
    )

    data_ref = make_8gaussians(800, rng=np.random.default_rng(1))
    plot_results(model, sigmas, losses, data_ref, save_path)

    # Coverage check
    samples = annealed_langevin(model, sigmas, n_samples=2000,
                                n_steps_per_level=100, eps_anneal=1e-3,
                                rng=np.random.default_rng(55))
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ])
    covered = sum(
        1 for c in centers if np.min(np.linalg.norm(samples - c, axis=1)) < 0.6
    )
    print(f"\nMode coverage: {covered}/8 Gaussian modes captured")
    print(f"Final training loss: {losses[-1]:.6f}")
    print(f"Noise schedule: σ_min={sigmas[0]:.4f}, σ_max={sigmas[-1]:.4f}")
    print(f"Results saved to {save_path}")
