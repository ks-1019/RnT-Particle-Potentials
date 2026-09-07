import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# ============================================================
# Configuration
# ============================================================

CSV_FILE = "heatmap_results/heatmap_results.csv"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_FILE)

df = df.sort_values("Pe").reset_index(drop=True)

Pe = df["Pe"].to_numpy()
current = df["current"].to_numpy()

accepted = (
    df["accepted"].astype(bool).to_numpy()
)

# ============================================================
# Heatmap values
# ============================================================

heatmap_values = current.copy()

# Mark rejected points separately
heatmap_values[~accepted] = np.nan

# 1-row image
heatmap_array = heatmap_values.reshape(1, -1)

# ============================================================
# Colormap
# ============================================================

cmap = plt.cm.RdYlGn.copy()

# rejected points shown as black
cmap.set_bad("black")

vmin = np.nanmin(heatmap_values)
vmax = np.nanmax(heatmap_values)

norm = Normalize(vmin=vmin, vmax=vmax)

# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(15, 6))
gs = GridSpec(
    2,
    1,
    height_ratios=[1, 3],
    hspace=0.35
)

# ============================================================
# Top: heatmap strip
# ============================================================

ax0 = fig.add_subplot(gs[0])

im = ax0.imshow(
    heatmap_array,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    interpolation="nearest"
)

ax0.set_yticks([])

tick_idx = np.arange(len(Pe))

ax0.set_xticks(tick_idx)
ax0.set_xticklabels(
    [f"{p:.2g}" for p in Pe],
    rotation=45,
    ha="right"
)

ax0.set_xlabel("Pe")
ax0.set_title(
    "Current optimisation results (Qe = 1)"
)

# ============================================================
# Colourbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax0,
    orientation="vertical",
    fraction=0.04,
    pad=0.02
)

cbar.set_label("Current")

# ============================================================
# Bottom: line plot
# ============================================================

ax1 = fig.add_subplot(gs[1])

# Accepted points
ax1.plot(
    Pe[accepted],
    current[accepted],
    marker="o",
    linewidth=2,
    label="Accepted"
)

# Rejected points
if np.any(~accepted):
    ax1.scatter(
        Pe[~accepted],
        current[~accepted],
        marker="x",
        s=80,
        linewidths=2,
        label="Rejected"
    )

ax1.set_xscale("log")

ax1.set_xlabel("Pe")
ax1.set_ylabel("Current")

ax1.set_title(
    "Optimised current vs Pe"
)

ax1.grid(True, which="both", alpha=0.3)

ax1.legend()

# ============================================================
# Summary statistics
# ============================================================

n_total = len(df)
n_accepted = np.sum(accepted)

fig.suptitle(
    f"Accepted points: {n_accepted}/{n_total}",
    fontsize=14
)

# ============================================================
# Save
# ============================================================

plt.savefig(
    "Pe_continuation_strip.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: Pe_continuation_strip.png")

plt.show()