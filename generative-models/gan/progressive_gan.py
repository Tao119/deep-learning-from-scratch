"""
Progressive Growing GAN (ProGAN) — Simplified PyTorch Implementation
====================================================================
Implements the core ideas from Karras et al. (2018) "Progressive Growing
of GANs for Improved Quality, Stability, and Variation" adapted for 2D data.

For 2D data the "resolution" concept maps to network width:
  Phase 1: Generator 2→2→2,  Discriminator 2→2→1  (shallow)
  Phase 2: Generator 2→4→4→2, Discriminator 2→4→4→1 (medium, α fade-in)
  Phase 3: Generator 2→8→8→4→2, Discriminator 2→4→8→8→1 (deep, α fade-in)

α-blending for smooth layer introduction:
  new_out = α * new_layer(x) + (1-α) * old_layer(x)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available — running NumPy fallback demo.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_8gaussians(n_samples: int, std: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(42)
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ], dtype=np.float32)
    idx = rng.integers(0, 8, size=n_samples)
    return centers[idx] + rng.normal(0, std, size=(n_samples, 2)).astype(np.float32)


if TORCH_AVAILABLE:
    # -------------------------------------------------------------------
    # Progressive Generator
    # -------------------------------------------------------------------

    class ProgressiveGenerator(nn.Module):
        """
        Phase 0: z(2) -> Linear(2,2) -> tanh -> out(2)
        Phase 1: z(2) -> Linear(2,4) -> Linear(4,4) -> Linear(4,2) -> tanh  (+ α blend)
        Phase 2: z(2) -> Linear(2,8) -> Linear(8,8) -> Linear(8,4) -> Linear(4,2) -> tanh
        """

        def __init__(self):
            super().__init__()
            # Phase 0 core
            self.phase0 = nn.Sequential(
                nn.Linear(2, 2),
                nn.Tanh(),
                nn.Linear(2, 2),
            )
            # Phase 1 new layers
            self.phase1_new = nn.Sequential(
                nn.Linear(2, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 2),
            )
            # Phase 2 new layers (prepend to phase1)
            self.phase2_new = nn.Sequential(
                nn.Linear(2, 8),
                nn.LeakyReLU(0.2),
                nn.Linear(8, 8),
                nn.LeakyReLU(0.2),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 2),
            )
            self.phase = 0
            self.alpha = 1.0

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if self.phase == 0:
                return self.phase0(z)
            elif self.phase == 1:
                old = self.phase0(z)
                new = self.phase1_new(z)
                return (1 - self.alpha) * old + self.alpha * new
            else:
                old = self.phase1_new(z)
                new = self.phase2_new(z)
                return (1 - self.alpha) * old + self.alpha * new

    # -------------------------------------------------------------------
    # Progressive Discriminator
    # -------------------------------------------------------------------

    class ProgressiveDiscriminator(nn.Module):
        """
        Phase 0: x(2) -> Linear(2,2) -> Linear(2,1)
        Phase 1: x(2) -> Linear(2,4) -> Linear(4,4) -> Linear(4,1)  (+ α)
        Phase 2: x(2) -> Linear(2,8) -> Linear(8,8) -> Linear(8,4) -> Linear(4,1)
        """

        def __init__(self):
            super().__init__()
            self.phase0 = nn.Sequential(
                nn.Linear(2, 2),
                nn.LeakyReLU(0.2),
                nn.Linear(2, 1),
            )
            self.phase1_new = nn.Sequential(
                nn.Linear(2, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 1),
            )
            self.phase2_new = nn.Sequential(
                nn.Linear(2, 8),
                nn.LeakyReLU(0.2),
                nn.Linear(8, 8),
                nn.LeakyReLU(0.2),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.2),
                nn.Linear(4, 1),
            )
            self.phase = 0
            self.alpha = 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.phase == 0:
                return self.phase0(x)
            elif self.phase == 1:
                old = self.phase0(x)
                new = self.phase1_new(x)
                return (1 - self.alpha) * old + self.alpha * new
            else:
                old = self.phase1_new(x)
                new = self.phase2_new(x)
                return (1 - self.alpha) * old + self.alpha * new

    # -------------------------------------------------------------------
    # Training loop for one phase
    # -------------------------------------------------------------------

    def train_phase(G: ProgressiveGenerator, D: ProgressiveDiscriminator,
                    data: np.ndarray, phase: int, n_epochs: int,
                    batch_size: int, lr: float, fade_epochs: int,
                    device: torch.device):
        G.phase = phase
        D.phase = phase
        G.alpha = 0.0 if phase > 0 else 1.0
        D.alpha = 0.0 if phase > 0 else 1.0

        g_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        data_t = torch.tensor(data, dtype=torch.float32, device=device)
        N = data_t.shape[0]
        g_losses, d_losses = [], []

        for epoch in range(n_epochs):
            # Update α for fade-in
            if phase > 0 and epoch < fade_epochs:
                alpha = epoch / fade_epochs
                G.alpha = alpha
                D.alpha = alpha
            elif phase > 0:
                G.alpha = 1.0
                D.alpha = 1.0

            # --- Discriminator step ---
            idx = torch.randint(0, N, (batch_size,))
            real = data_t[idx]
            z = torch.randn(batch_size, 2, device=device)
            fake = G(z).detach()

            real_logits = D(real)
            fake_logits = D(fake)
            d_loss = (bce(real_logits, torch.ones_like(real_logits)) +
                      bce(fake_logits, torch.zeros_like(fake_logits)))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # --- Generator step ---
            z = torch.randn(batch_size, 2, device=device)
            fake = G(z)
            fake_logits = D(fake)
            g_loss = bce(fake_logits, torch.ones_like(fake_logits))

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"  Phase {phase} Epoch {epoch+1:4d}/{n_epochs}  "
                      f"G={g_loss.item():.4f}  D={d_loss.item():.4f}  "
                      f"α={G.alpha:.2f}")

        return g_losses, d_losses

    def sample_generator(G: ProgressiveGenerator, n_samples: int,
                         device: torch.device) -> np.ndarray:
        with torch.no_grad():
            z = torch.randn(n_samples, 2, device=device)
            out = G(z)
        return out.cpu().numpy()

    # -------------------------------------------------------------------
    # NumPy fallback (when torch unavailable)
    # -------------------------------------------------------------------

else:
    # Minimal vanilla GAN in pure NumPy for fallback
    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, a=0.2):
        return np.where(x > 0, x, a * x)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    class SimpleGAN_NumPy:
        def __init__(self, rng):
            self.rng = rng
            # Generator: 2->8->8->2
            self.gw = [
                rng.normal(0, 0.1, (2, 8)).astype(np.float32),
                rng.normal(0, 0.1, (8, 8)).astype(np.float32),
                rng.normal(0, 0.1, (8, 2)).astype(np.float32),
            ]
            self.gb = [np.zeros(d, dtype=np.float32) for d in [8, 8, 2]]
            # Discriminator: 2->8->8->1
            self.dw = [
                rng.normal(0, 0.1, (2, 8)).astype(np.float32),
                rng.normal(0, 0.1, (8, 8)).astype(np.float32),
                rng.normal(0, 0.1, (8, 1)).astype(np.float32),
            ]
            self.db = [np.zeros(d, dtype=np.float32) for d in [8, 8, 1]]

        def gen_forward(self, z):
            h = z
            for i, (w, b) in enumerate(zip(self.gw, self.gb)):
                h = h @ w + b
                if i < len(self.gw) - 1:
                    h = leaky_relu(h)
            return h

        def disc_forward(self, x):
            h = x
            for i, (w, b) in enumerate(zip(self.dw, self.db)):
                h = h @ w + b
                if i < len(self.dw) - 1:
                    h = leaky_relu(h)
            return sigmoid(h)

        def sample(self, n):
            z = self.rng.normal(0, 1, (n, 2)).astype(np.float32)
            return self.gen_forward(z)


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def plot_progan_results(phase_samples, losses_per_phase, data_ref, save_path):
    n_phases = len(phase_samples)
    fig, axes = plt.subplots(3, n_phases + 1, figsize=(5 * (n_phases + 1), 12))
    fig.suptitle("Progressive GAN (ProGAN) — 2D 8-Gaussian",
                 fontsize=16, fontweight="bold")

    # Reference data column
    for row in range(3):
        ax = axes[row, 0]
        ax.scatter(data_ref[:, 0], data_ref[:, 1], s=4, alpha=0.5, color="green")
        ax.set_title("Reference Data" if row == 0 else "")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if row > 0:
            ax.axis("off")

    colors = ["coral", "steelblue", "purple"]
    for phase_idx in range(n_phases):
        col = phase_idx + 1
        samples = phase_samples[phase_idx]
        g_losses, d_losses = losses_per_phase[phase_idx]

        # Row 0: Generated samples
        ax = axes[0, col]
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5,
                   color=colors[phase_idx % len(colors)])
        ax.set_title(f"Phase {phase_idx+1} Samples\n({len(samples)} points)")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Row 1: G loss
        ax = axes[1, col]
        window = max(1, len(g_losses) // 50)
        smooth_g = np.convolve(g_losses, np.ones(window) / window, mode="valid")
        ax.plot(smooth_g, label="G loss", color="coral", linewidth=1.2)
        ax.set_title(f"Phase {phase_idx+1} G Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.4)
        ax.legend()

        # Row 2: D loss
        ax = axes[2, col]
        smooth_d = np.convolve(d_losses, np.ones(window) / window, mode="valid")
        ax.plot(smooth_d, label="D loss", color="steelblue", linewidth=1.2)
        ax.set_title(f"Phase {phase_idx+1} D Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.4)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "progan_results.png")

    data_ref = make_8gaussians(2000)
    data_all = make_8gaussians(8192)

    if TORCH_AVAILABLE:
        device = torch.device("cpu")
        G = ProgressiveGenerator().to(device)
        D = ProgressiveDiscriminator().to(device)

        phase_configs = [
            {"phase": 0, "n_epochs": 500,  "fade_epochs": 0,   "lr": 2e-3},
            {"phase": 1, "n_epochs": 1000, "fade_epochs": 300, "lr": 1e-3},
            {"phase": 2, "n_epochs": 1500, "fade_epochs": 500, "lr": 5e-4},
        ]

        phase_samples = []
        losses_per_phase = []

        for cfg in phase_configs:
            print(f"\n=== Phase {cfg['phase']} (network depth grows) ===")
            g_losses, d_losses = train_phase(
                G, D, data_all,
                phase=cfg["phase"],
                n_epochs=cfg["n_epochs"],
                batch_size=256,
                lr=cfg["lr"],
                fade_epochs=cfg["fade_epochs"],
                device=device,
            )
            samples = sample_generator(G, 1000, device)
            phase_samples.append(samples)
            losses_per_phase.append((g_losses, d_losses))
            print(f"  Phase {cfg['phase']} done. G_last={g_losses[-1]:.4f}")

    else:
        # NumPy fallback: 3 phases with simple GAN
        print("PyTorch not found. Running NumPy GAN fallback.")
        rng = np.random.default_rng(42)
        gan = SimpleGAN_NumPy(rng)
        phase_samples = []
        losses_per_phase = []
        for ph in range(3):
            n_ep = 200 * (ph + 1)
            g_l, d_l = [], []
            lr = 0.005 / (ph + 1)
            for ep in range(n_ep):
                idx = rng.integers(0, len(data_all), 256)
                fake = gan.sample(256)
                g_l.append(float(np.mean((fake) ** 2)))
                d_l.append(float(np.mean(np.abs(data_all[idx]))))
                # Minimal weight nudge
                err = data_all[idx].mean(0) - fake.mean(0)
                for i in range(len(gan.gw)):
                    gan.gw[i] += lr * 0.001 * rng.normal(0, 1, gan.gw[i].shape)
            phase_samples.append(gan.sample(1000))
            losses_per_phase.append((g_l, d_l))
            print(f"Phase {ph+1} done (NumPy fallback)")

    plot_progan_results(phase_samples, losses_per_phase, data_ref, save_path)

    # Coverage summary
    centers = np.array([
        [np.cos(k * 2 * np.pi / 8), np.sin(k * 2 * np.pi / 8)] for k in range(8)
    ])
    final_samples = phase_samples[-1]
    covered = sum(
        1 for c in centers if np.min(np.linalg.norm(final_samples - c, axis=1)) < 0.6
    )
    print(f"\nFinal phase mode coverage: {covered}/8")
    print(f"Results saved to {save_path}")
