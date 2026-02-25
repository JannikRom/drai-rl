import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
from scipy.ndimage import gaussian_filter1d

BASE_DIR   = os.path.dirname(__file__)
TEXT_WIDTH = 6.063 
ASPECT     = 0.82    

REWARD_CFG = dict(smooth_sigma=1.5,  roll_window=40, band_alpha=0.15, line_width=1.0)
ALPHA_CFG  = dict(smooth_sigma=1.5,  roll_window=40, band_alpha=0.0,  line_width=1.0)
ACTOR_CFG  = dict(smooth_sigma=1.5,  roll_window=40, band_alpha=0.0,  line_width=1.0)
CRITIC_CFG = dict(smooth_sigma=10.0, roll_window=40, band_alpha=0.0,  line_width=1.0)

ALPHA_YMAX = 0.008

MARGIN_LEFT   = 0.09
MARGIN_RIGHT  = 0.99
MARGIN_TOP    = 0.97
MARGIN_BOTTOM = 0.20 
HSPACE        = 0.50   
WSPACE        = 0.38   

ALL_AGENTS = [
    ("sac_basic",          "SAC Basic"),
    ("sac_reward_shaping", "SAC Reward Shaping"),
    ("td3_basic",          "TD3 Basic"),
    ("td3_pink",           "TD3 Pink"),
    ("td3_per",            "TD3 PER"),
]
SAC_AGENTS  = ALL_AGENTS[:2]
TD3_AGENTS  = ALL_AGENTS[2:]
LOSS_AGENTS = ALL_AGENTS


def load_json(fpath: str) -> tuple[np.ndarray, np.ndarray]:
    with open(fpath) as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return np.array([]), np.array([])
        if isinstance(data[0], (list, tuple)):
            steps  = np.array([r[1] for r in data], dtype=float)
            values = np.array([r[2] for r in data], dtype=float)
        else:
            steps  = np.array([r["step"]  for r in data], dtype=float)
            values = np.array([r["value"] for r in data], dtype=float)
    elif isinstance(data, dict):
        if "steps" in data and "values" in data:
            steps  = np.array(data["steps"],  dtype=float)
            values = np.array(data["values"], dtype=float)
        else:
            inner  = next(iter(data.values()))
            steps  = np.array([r[1] for r in inner], dtype=float)
            values = np.array([r[2] for r in inner], dtype=float)
    else:
        return np.array([]), np.array([])
    idx = np.argsort(steps)
    return steps[idx], values[idx]


def load_runs(data_dir: str, agent_key: str,
              value_max: float | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    runs = []
    if not os.path.isdir(data_dir):
        return runs
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json") or agent_key.lower() not in fname.lower():
            continue
        steps, values = load_json(os.path.join(data_dir, fname))
        if not len(steps):
            continue
        if value_max is not None:
            mask = values <= value_max
            steps, values = steps[mask], values[mask]
        if len(steps):
            runs.append((steps, values))
    return runs


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=max(sigma, 1e-6))


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    n    = len(arr)
    out  = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i]  = arr[lo:hi].std()
    return out


def interpolate_to_common_grid(
    runs: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_min = max(r[0][0]  for r in runs)
    step_max = min(r[0][-1] for r in runs)
    grid     = np.linspace(step_min, step_max, 500)
    interp   = np.stack([np.interp(grid, s, v) for s, v in runs])
    return grid, interp.mean(0), interp.std(0)


def draw_curves(ax, data_dir: str, agents: list[tuple[str, str]],
                ylabel: str, title: str,
                smooth_sigma: float, roll_window: int,
                band_alpha: float, line_width: float,
                color_map: dict | None = None,
                value_max: float | None = None):
    if color_map is None:
        color_map = {}

    for agent_key, agent_label in agents:
        runs = load_runs(data_dir, agent_key, value_max=value_max)
        if not runs:
            print(f"[warn] No data for '{agent_key}' in {data_dir}")
            continue

        steps, raw = runs[0] if len(runs) == 1 else interpolate_to_common_grid(runs)[:2]

        mean_s = smooth(raw, smooth_sigma)
        std_s  = smooth(rolling_std(raw, roll_window), smooth_sigma)

        kwargs = {"label": agent_label, "linewidth": line_width}
        if agent_key in color_map:
            kwargs["color"] = color_map[agent_key]

        (line,) = ax.plot(steps, mean_s, **kwargs)
        color_map[agent_key] = line.get_color()

        ax.fill_between(steps, mean_s - std_s, mean_s + std_s,
                        alpha=band_alpha, color=line.get_color(), linewidth=0)

    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    return color_map

plt.rcParams.update(bundles.neurips2024())

fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * ASPECT))

ax_reward, ax_alpha = axes[0]
ax_actor,  ax_critic = axes[1]

color_map = draw_curves(
    ax_reward,
    data_dir = os.path.join(BASE_DIR, "rewards"),
    agents   = ALL_AGENTS,
    ylabel   = "Reward", title = "Reward",
    **REWARD_CFG,
)

draw_curves(
    ax_alpha,
    data_dir  = os.path.join(BASE_DIR, "alpha"),
    agents    = SAC_AGENTS,
    ylabel    = "Alpha", title = "Alpha",
    color_map = {k: color_map[k] for k in dict(SAC_AGENTS) if k in color_map},
    **ALPHA_CFG,
)
ax_alpha.set_ylim(bottom=0, top=ALPHA_YMAX)

draw_curves(
    ax_actor,
    data_dir  = os.path.join(BASE_DIR, "actor_loss"),
    agents    = LOSS_AGENTS,
    ylabel    = "Actor Loss", title = "Actor Loss",
    color_map = {k: color_map[k] for k in dict(LOSS_AGENTS) if k in color_map},
    **ACTOR_CFG,
)

draw_curves(
    ax_critic,
    data_dir  = os.path.join(BASE_DIR, "critic_loss"),
    agents    = LOSS_AGENTS,
    ylabel    = "Critic Loss", title = "Critic Loss",
    color_map = {k: color_map[k] for k in dict(LOSS_AGENTS) if k in color_map},
    **CRITIC_CFG,
)


fig.subplots_adjust(
    left   = MARGIN_LEFT,
    right  = MARGIN_RIGHT,
    top    = MARGIN_TOP,
    bottom = MARGIN_BOTTOM,
    hspace = HSPACE,
    wspace = WSPACE,
)


handles, labels = ax_reward.get_legend_handles_labels()
leg = fig.legend(
    handles, labels,
    loc            = "lower center",
    ncol           = len(ALL_AGENTS),
    frameon        = True,
    facecolor      = "white",
    edgecolor      = "#888888",
    framealpha     = 1.0,

    bbox_to_anchor = (0.5, 0.-0.05),
    bbox_transform = fig.transFigure,
)
leg.get_frame().set_linewidth(0.8)

out_path = os.path.join(BASE_DIR, "overview_plot.pdf")

fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
print(f"Saved → {out_path}")
plt.show()