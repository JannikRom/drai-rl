import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes
from scipy.ndimage import gaussian_filter1d

REWARDS_DIR = os.path.join(os.path.dirname(__file__), "rewards")

TEXT_WIDTH = 6.063 

AGENTS = [
    ("sac_basic",           "SAC Basic"),
    ("sac_reward_shaping",  "SAC Reward Shaping"),
    ("td3_basic",           "TD3 Basic"),
    ("td3_pink",            "TD3 Pink"),
    ("td3_per",             "TD3 PER"),
]

SMOOTH_SIGMA = 1.5    
BAND_ALPHA   = 0.15 
ROLL_WINDOW  = 40  
                

def load_runs(agent_key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (steps, values) arrays, one per matching JSON file."""
    runs = []
    for fname in sorted(os.listdir(REWARDS_DIR)):
        if not fname.endswith(".json"):
            continue
        if agent_key.lower() not in fname.lower():
            continue
        fpath = os.path.join(REWARDS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) == 0:
                continue
            if isinstance(data[0], (list, tuple)):
                steps  = np.array([row[1] for row in data], dtype=float)
                values = np.array([row[2] for row in data], dtype=float)
            else:
                steps  = np.array([row["step"]  for row in data], dtype=float)
                values = np.array([row["value"] for row in data], dtype=float)
        elif isinstance(data, dict):
            if "steps" in data and "values" in data:
                steps  = np.array(data["steps"],  dtype=float)
                values = np.array(data["values"], dtype=float)
            else:
                inner  = next(iter(data.values()))
                steps  = np.array([row[1] for row in inner], dtype=float)
                values = np.array([row[2] for row in inner], dtype=float)
        else:
            print(f"  [warn] Unrecognised format in {fname}, skipping.")
            continue

        sort_idx = np.argsort(steps)
        runs.append((steps[sort_idx], values[sort_idx]))
    return runs


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=sigma)


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred sliding-window standard deviation; edges shrink the window."""
    half = window // 2
    n    = len(arr)
    out  = np.empty(n)
    for i in range(n):
        lo     = max(0, i - half)
        hi     = min(n, i + half + 1)
        out[i] = arr[lo:hi].std()
    return out


def interpolate_to_common_grid(
    runs: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_min = max(r[0][0]  for r in runs)
    step_max = min(r[0][-1] for r in runs)
    grid = np.linspace(step_min, step_max, 500)
    interp = np.stack(
        [np.interp(grid, steps, vals) for steps, vals in runs], axis=0
    )
    return grid, interp.mean(axis=0), interp.std(axis=0)

plt.rcParams.update(bundles.neurips2024())
fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.6))


for agent_key, agent_label in AGENTS:
    runs = load_runs(agent_key)
    if not runs:
        print(f"[warn] No data found for '{agent_key}' in {REWARDS_DIR}/")
        conti

    if len(runs) == 1:
        steps, raw = runs[0]
    else:
        steps, raw, _ = interpolate_to_common_grid(runs)


    mean_s = smooth(raw, SMOOTH_SIGMA)


    std_r  = rolling_std(raw, ROLL_WINDOW)
    std_s  = smooth(std_r, SMOOTH_SIGMA)   

    (line,) = ax.plot(steps, mean_s, label=agent_label, linewidth=1.5)
    ax.fill_between(
        steps,
        mean_s - std_s,
        mean_s + std_s,
        alpha=BAND_ALPHA,
        color=line.get_color(),
        linewidth=0,
    )

ax.set_xlabel("Steps")
ax.set_ylabel("Avg. Episode Reward")
ax.set_title("Episode Rewards")
leg = ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="#888888",
    framealpha=1.0,
)
leg.get_frame().set_linewidth(0.8)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.5)
out_path = os.path.join(os.path.dirname(__file__), "reward_plot.pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
print(f"Saved plot to {out_path}")
plt.show()