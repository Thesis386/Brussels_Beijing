"""Illustrative RQ3 stance-distribution figures."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402
from viz_style import (  # noqa: E402
    INK, INK_MUTED, MODEL_ORDER, REFUSAL_COLOR, STANCE_COLOR, STANCE_LABEL,
    apply_style, save,
)

FIG = Path("evaluation/analysis/outputs/figures")
STANCES = [1, 2, 3, 4, 5]
MODEL_DISPLAY = {"claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
                 "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral"}

# each example: (title, language, parent issue, named issue)
EXAMPLES = [
    ("Pre-deployment review\n(English prompts)", "en", "B1", "C1.1"),
    ("Content filtering\n(Chinese prompts)", "zh", "B8", "C8.1"),
]
SUBLABEL = {"parent": "neutral policy", "named": "labeled “China’s”"}


def pmf(s):
    v = s.dropna()
    if len(v) == 0:
        return np.full(5, np.nan)
    return np.array([(v == k).sum() for k in STANCES], dtype=float) / len(v)


def main():
    apply_style()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_master()
    d = df[~df["is_refusal"]]

    # y-layout: for each example, a parent sub-block (6 models) then a named sub-block
    ypos, meta = [], []          # meta: (issue, model, is_named)
    sub_centre, ex_span = {}, {}
    y = 0.0
    for ei, (title, lang, b, c) in enumerate(EXAMPLES):
        top = y
        for role, iss in [("parent", b), ("named", c)]:
            block_top = y
            for m in MODEL_ORDER:
                ypos.append(y)
                meta.append((iss, m, lang, role))
                y += 1.0
            sub_centre[(ei, role)] = (block_top + y - 1.0) / 2.0
            y += 0.7                       # gap between parent and named sub-blocks
        ex_span[ei] = (top, y - 0.7)
        y += 1.7                           # bigger gap between examples

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for (iss, m, lang, role), yy in zip(meta, ypos):
        g = d[(d["issue_id"] == iss) & (d["model"] == m) & (d["lang"] == lang)]
        p = pmf(g["stance_num"])
        if np.isnan(p).any():
            continue
        left = -(p[0] + p[1] + p[2] / 2)
        for k, share in zip(STANCES, p):
            if share <= 0:
                continue
            ax.barh(yy, share, left=left, height=0.82, color=STANCE_COLOR[k],
                    edgecolor="white", linewidth=0.8, zorder=3)
            left += share

    ax.plot([0, 0], [min(ypos) - 0.7, max(ypos) + 0.7], color=INK_MUTED, lw=0.9,
            zorder=4, alpha=0.7)
    ax.set_xlim(-1.06, 1.06)
    ax.set_xticks([-1, 0, 1], ["", "3 neutral", ""])
    ax.set_xlabel("share of responses      <-- pro      con -->", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(max(ypos) + 1.2, min(ypos) - 1.2)
    # Rows in every sub-block run in MODEL_ORDER, so the names go in one key below
    # rather than being repeated for each of the four sub-blocks.
    ax.set_yticks(ypos, [""] * len(ypos))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # left gutter: sub-block role label + example titles
    gut = blended_transform_factory(fig.transFigure, ax.transData)
    for ei, (title, lang, b, c) in enumerate(EXAMPLES):
        for role in ["parent", "named"]:
            ax.text(0.160, sub_centre[(ei, role)], SUBLABEL[role], transform=gut,
                    ha="center", va="center", fontsize=9,
                    color=INK if role == "named" else INK_MUTED,
                    fontweight="bold" if role == "named" else "normal")
        top, bot = ex_span[ei]
        ax.text(0.035, (top + bot) / 2, title, transform=gut, ha="center", va="center",
                rotation=90, fontsize=9.5, fontweight="bold", color=INK,
                linespacing=1.3)

    handles = [Patch(facecolor=STANCE_COLOR[k], label=STANCE_LABEL[k]) for k in STANCES]
    handles.append(Patch(facecolor=REFUSAL_COLOR, hatch="///", label="refusal"))
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.55, 0.055),
               handlelength=1.3, columnspacing=1.2, fontsize=9, frameon=False)
    key = "  \u00b7  ".join(MODEL_DISPLAY[m] for m in MODEL_ORDER)
    fig.text(0.55, 0.008, f"Six rows per sub-block, top to bottom:  {key}",
             ha="center", va="bottom", fontsize=8.5, color=INK_MUTED)

    fig.subplots_adjust(left=0.27, right=0.98, top=0.97, bottom=0.185)
    save(fig, FIG / "nb3g_rq3_example_dists")
    print("wrote nb3g_rq3_example_dists")


if __name__ == "__main__":
    main()
