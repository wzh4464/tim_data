from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE_PRIMARY = {
    "blue_light": "#4eb3d3",
    "blue_dark": "#2c7bb6",
    "orange": "orange",
    "red_dark": "#d73027",
    "grey": "grey",
    "black": "black",
}

GRADIENT_BLUES = ["#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cde"]

MARKERS = ["o", "^", "x", "s", "v", "1"]


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


