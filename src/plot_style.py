from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE_PRIMARY = {
    "blue_light": "#4eb3d3",
    "blue_dark": "#2c7bb6",
    "orange": "orange",
    "red_dark": "#d73027",
    "grey": "#b0b0b0",
    "black": "#000000",
}

GRADIENT_BLUES = ["#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cde"]

# Okabe–Ito colorblind-safe palette (ICLR-friendly)
PALETTE_ICLR = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # purple/pink
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
]

MARKERS = ["o", "v", "x", "s", "^", "1"]


def apply_global_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Times",
            "font.size": 18,
            # Grid aesthetics
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        }
    )
    plt.rc("legend", fontsize=12)


def default_figsize() -> tuple[float, float]:
    return (6, 4)


def savefig(path: str) -> None:
    plt.savefig(path, dpi=600, bbox_inches="tight")


def enable_axes_grid(ax) -> None:
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.grid(axis="x", linestyle="--", zorder=0)


