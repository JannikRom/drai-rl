"""
full_page_plot.py
-----------------
Plots episode rewards for all algorithm variants in report_3_2.
Uses tueplots for publication-quality formatting.

Colour palette : hand-picked from tueplots.constants.color.rgb
CI band        : 60 % bootstrap CI (tight)
                 For single-seed files: ±0.25 × rolling-std
Smoothing      : Gaussian  (sigma = SIGMA episodes)
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
Y_MIN       = -10
SIGMA       = 150          # Gaussian σ for line smoothing
SIGMA_CI    = 220          # σ for CI envelope smoothing
N_BOOT      = 500          # bootstrap resamples
CI_ALPHA    = 0.80         # 50 % CI → very tight band
CI_STD_MULT = 0.30         # single-seed: ± this × rolling-std (very narrow)
ROLL_WIN    = int(SIGMA)   # rolling-std window (single-seed)

# Hand-picked from tueplots.constants.color.rgb —
# a modern, high-contrast set with clear visual separation:
#   tue_blue       – vivid sky blue        (SAC Basic)
#   tue_red        – strong crimson        (SAC Reward Shaping)
#   tue_ocre       – warm amber/ocre       (TD3 Basic)
#   tue_lightblue  – bright cyan-blue      (TD3 PER)
#   tue_green      – deep forest green     (TD3 PINK)
COLORS = [
    rgb.tue_blue,        # SAC Basic
    rgb.pn_red,         # SAC Reward Shaping
    rgb.tue_lightorange,        # TD3 Basic
    rgb.tue_lightblue,   # TD3 PER
    rgb.tue_darkgreen,   # TD3 PINK
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def gsmooth(x: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    return gaussian_filter1d(x.astype(float), sigma=sigma)


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = N_BOOT,
    alpha: float = CI_ALPHA,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI for the mean. data: (runs, T) → lo, hi: (T,)"""
    rng = rng or np.random.default_rng(0)
    n_runs, T = data.shape
    boot = np.empty((n_boot, T))
    for i in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[i] = data[idx].mean(axis=0)
    tail = (1 - alpha) / 2 * 100
    return np.percentile(boot, tail, axis=0), np.percentile(boot, 100 - tail, axis=0)


def rolling_std(x: np.ndarray, w: int = ROLL_WIN) -> np.ndarray:
    w = max(1, w)
    pad = np.pad(x.astype(float), (w // 2, w // 2), mode="edge")
    return np.array([pad[i:i + w].std() for i in range(len(x))])


def load_rewards(folder: Path) -> np.ndarray | None:
    path = folder / "episode_rewards.npy"
    if not path.exists():
        print(f"[warn] {path} not found – skipping.")
        return None
    return np.load(path)


# ── Plot ─────────────────────────────────────────────────────────────────────

def main() -> None:
    plt.rcParams.update(bundles.icml2022())

    full_w = figsizes.icml2022_full()["figure.figsize"][0]
    fig, ax = plt.subplots(figsize=(full_w, full_w / 1.618))

    rng     = np.random.default_rng(42)
    plotted = 0

    for (dirname, label), color in zip(VARIANTS, COLORS):
        rewards = load_rewards(BASE_DIR / dirname)
        if rewards is None:
            continue

        rewards = rewards.squeeze()
        if rewards.ndim == 2:
            rewards = rewards[:, :X_MAX]
        else:
            rewards = rewards[:X_MAX]

        if rewards.ndim == 1:
            y    = gsmooth(rewards)
            x    = np.arange(len(y))
            rstd = gsmooth(rolling_std(rewards), sigma=SIGMA_CI)
            lo   = y - CI_STD_MULT * rstd
            hi   = y + CI_STD_MULT * rstd
            ax.fill_between(x, lo, hi, alpha=0.15, color=color, linewidth=0)
            ax.plot(x, y, label=label, color=color, linewidth=1.4)

        else:
            lo_raw, hi_raw = bootstrap_ci(rewards, rng=rng)
            mean = gsmooth(rewards.mean(axis=0))
            lo   = gsmooth(lo_raw,  sigma=SIGMA_CI)
            hi   = gsmooth(hi_raw,  sigma=SIGMA_CI)
            x    = np.arange(len(mean))
            ax.fill_between(x, lo, hi, alpha=0.15, color=color, linewidth=0)
            ax.plot(x, mean, label=label, color=color, linewidth=1.4)

        plotted += 1

    if plotted == 0:
        raise FileNotFoundError(
            f"No episode_rewards.npy files found. Check BASE_DIR = {BASE_DIR.resolve()}"
        )

    # ── Axes -----------------------------------------------------------------
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(bottom=Y_MIN)

    # ── Grid -----------------------------------------------------------------
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.87", linewidth=0.55, linestyle="-")
    ax.grid(which="minor", color="0.93", linewidth=0.28, linestyle="-")
    ax.minorticks_on()

    # ── Ticks ----------------------------------------------------------------
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xticks(range(0, X_MAX + 1, 5_000))

    # ── Labels ---------------------------------------------------------------
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards – all variants")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()

    out = BASE_DIR / "full_page_plot.pdf"
    fig.savefig(out)
    print(f"Saved → {out}")
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved → {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()