"""Shared plotting style and model/origin display mappings."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# categorical slots, fixed order (blue, aqua, yellow, green, violet, red)
SLOT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

# origin identity: first three slots, in fixed order
ORIGIN_COLOR = {"US": SLOT[0], "CN": SLOT[1], "EU": SLOT[2]}
ORIGIN_ORDER = ["US", "CN", "EU"]

# model identity: marker shape carries the model, hue carries its origin
MODEL_ORIGIN = {
    "claude": "US",
    "gpt4o": "US",
    "llama": "US",
    "deepseek": "CN",
    "qwen": "CN",
    "mistral": "EU",
}
MODEL_MARKER = {
    "claude": "o",
    "gpt4o": "s",
    "llama": "^",
    "deepseek": "D",
    "qwen": "v",
    "mistral": "P",
}
MODEL_ORDER = ["claude", "gpt4o", "llama", "deepseek", "qwen", "mistral"]

# sequential ramp (one hue, light -> dark) for magnitude: heatmaps
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# diverging pair for stance polarity: pro (blue) <-> con (red), neutral grey midpoint
STANCE_COLOR = {
    1: "#1c5cab",  # only pro
    2: "#86b6ef",  # leans pro
    3: "#d8d7d3",  # neutral / balanced -- grey, so the midpoint reads as "nothing"
    4: "#ef9d9c",  # leans con
    5: "#c33534",  # only con
}
STANCE_LABEL = {
    1: "1 only pro",
    2: "2 leans pro",
    3: "3 neutral",
    4: "4 leans con",
    5: "5 only con",
}
REFUSAL_COLOR = "#8a8985"

SURFACE = "#ffffff"  # pure white page background (default)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def apply_style() -> None:
    """Thin marks, hairline recessive grid, no boxed-in spines."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # solid: a dashed grid reads as a threshold
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path) -> None:
    """Write the figure as both PDF (for LaTeX) and PNG (for quick viewing)."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png")
    plt.close(fig)
