"""
full_page_losses_plot.py
------------------------
2×2 grid of training curves for all algorithm variants in report_3_2.

Layout
------
  [top-left]     Episode rewards    (episode_rewards.npy)
  [top-right]    Alpha loss         (training_losses.npz → alpha_loss)   SAC only
  [bottom-left]  Actor loss         (training_losses.npz → actor_loss)
  [bottom-right] Critic loss        (training_losses.npz → critic_loss)

Keys per agent (from source)
-----------------------------
  SAC : critic_loss, actor_loss, alpha_loss, alpha, mean_log_prob
  TD3 : critic_loss, actor_loss (None on non-policy-update steps),
        q1_mean, q2_mean, target_q_mean

Note: TD3 actor_loss contains None entries (logged only every policy_delay
steps). These are stripped before plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tueplots import bundles, figsizes
from tueplots.constants.color import rgb

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

VARIANTS = [
    ("sac_basic",          "SAC Basic"),
    ("sac_reward_shaping", "SAC Reward Shaping"),
    ("td3_basic",          "TD3 Basic"),
    ("td3_per",            "TD3 PER"),
    ("td3_pink",           "TD3 PINK"),
]

X_MAX       = 20_000
Y_MIN_REW   = -10
SIGMA       = 150
SIGMA_CI    = 220
N_BOOT      = 300
CI_ALPHA    = 0.80
CI_STD_MULT = 0.30
ROLL_WIN    = int(SIGMA)

COLORS = [
    rgb.tue_blue,         # SAC Basic
    rgb.pn_red,           # SAC Reward Shaping
    rgb.tue_lightorange,  # TD3 Basic
    rgb.tue_lightblue,    # TD3 PER
    rgb.tue_darkgreen,    # TD3 PINK
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def gsmooth(x: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    return gaussian_filter1d(x.astype(float), sigma=sigma)


def rolling_std(x: np.ndarray, w: int = ROLL_WIN) -> np.ndarray:
    w = max(1, w)
    pad = np.pad(x.astype(float), (w // 2, w // 2), mode="edge")
    return np.array([pad[i:i + w].std() for i in range(len(x))])


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = N_BOOT,
    alpha: float = CI_ALPHA,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    n_runs, T = data.shape
    boot = np.empty((n_boot, T))
    for i in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[i] = data[idx].mean(axis=0)
    tail = (1 - alpha) / 2 * 100
    return np.percentile(boot, tail, axis=0), np.percentile(boot, 100 - tail, axis=0)


def plot_series(ax: plt.Axes, data: np.ndarray, color, label: str,
                rng: np.random.Generator) -> None:
    """Smooth + CI band for one (possibly multi-seed) series."""
    data = data.squeeze()
    if data.ndim == 2:
        data = data[:, :X_MAX]
    else:
        data = data[:X_MAX]

    if data.ndim == 1:
        y    = gsmooth(data)
        x    = np.arange(len(y))
        rstd = gsmooth(rolling_std(data), sigma=SIGMA_CI)
        lo, hi = y - CI_STD_MULT * rstd, y + CI_STD_MULT * rstd
    else:
        lo_raw, hi_raw = bootstrap_ci(data, rng=rng)
        y  = gsmooth(data.mean(axis=0))
        lo = gsmooth(lo_raw, sigma=SIGMA_CI)
        hi = gsmooth(hi_raw, sigma=SIGMA_CI)
        x  = np.arange(len(y))

    ax.fill_between(x, lo, hi, alpha=0.15, color=color, linewidth=0)
    ax.plot(x, y, label=label, color=color, linewidth=1.4)


def style_ax(ax: plt.Axes, title: str, ylabel: str,
             yticks: list | None = None) -> None:
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.87", linewidth=0.55, linestyle="-")
    ax.grid(which="minor", color="0.93", linewidth=0.28, linestyle="-")
    ax.minorticks_on()
    ax.set_xlim(0, X_MAX)
    ax.set_xticks(range(0, X_MAX + 1, 5_000))
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel("Episode" if "Reward" in title else "Update step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def load_rewards(folder: Path) -> np.ndarray | None:
    p = folder / "episode_rewards.npy"
    return np.load(p) if p.exists() else None


def load_loss(folder: Path, key: str) -> np.ndarray | None:
    """Load one key from training_losses.npz, dropping None / NaN entries."""
    p = folder / "training_losses.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as f:
        if key not in f:
            return None
        arr = f[key].copy()

    # TD3 actor_loss stores None on non-update steps — strip them
    if arr.dtype == object:
        arr = np.array([v for v in arr if v is not None], dtype=float)

    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]          # remove any inf / nan
    return arr if len(arr) > 0 else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    plt.rcParams.update(bundles.icml2022())

    full_w = figsizes.icml2022_full()["figure.figsize"][0]
    fig, axes = plt.subplots(
        2, 2,
        figsize=(full_w, full_w / 1.3),
    )
    ax_rew, ax_alpha, ax_actor, ax_critic = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    rng = np.random.default_rng(42)

    for (dirname, label), color in zip(VARIANTS, COLORS):
        folder = BASE_DIR / dirname

        # top-left  → episode rewards
        rew = load_rewards(folder)
        if rew is not None:
            plot_series(ax_rew, rew, color, label, rng)

        # top-right → alpha_loss  (SAC only; TD3 folders simply won't have it)
        alpha = load_loss(folder, "alpha_loss")
        if alpha is not None:
            plot_series(ax_alpha, alpha, color, label, rng)

        # bottom-left → actor_loss  (both; TD3 Nones already stripped)
        actor = load_loss(folder, "actor_loss")
        if actor is not None:
            plot_series(ax_actor, actor, color, label, rng)

        # bottom-right → critic_loss  (both)
        critic = load_loss(folder, "critic_loss")
        if critic is not None:
            plot_series(ax_critic, critic, color, label, rng)

    # ── Style ────────────────────────────────────────────────────────────────
    style_ax(ax_rew,    "Episode Rewards", "Reward",      yticks=[-10, -5, 0, 5, 10])
    style_ax(ax_alpha,  "Alpha Loss",      "Loss")
    style_ax(ax_actor,  "Actor Loss",      "Loss")
    style_ax(ax_critic, "Critic Loss",     "Loss")

    # Legend only in top-left (rewards) — avoids repetition
    ax_rew.legend(loc="lower right", frameon=False)

    fig.tight_layout()

    out = BASE_DIR / "full_page_losses_plot.pdf"
    fig.savefig(out)
    print(f"Saved → {out}")
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved → {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()