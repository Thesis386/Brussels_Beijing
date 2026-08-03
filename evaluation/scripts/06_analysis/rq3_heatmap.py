"""RQ3 attribution-effect heatmap."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402
from viz_style import INK, INK_MUTED, STANCE_COLOR, apply_style, save  # noqa: E402

FIG = Path("evaluation/analysis/outputs/figures")
PAIR_KEY = ["model", "lang", "template_id", "rep"]
NAMED_DIR = {"aligned": -1, "reversed": +1}

# row label = named jurisdiction + short policy description (no bare issue IDs)
ROW_LABEL = {
    "C1.1": "China: pre-deployment government authorization",
    "C11.1": "US: deregulation to boost domestic AI",
    "C3.1": "EU: mandatory training-data disclosure",
    "C9.1": "China: UN as the main channel for AI governance",
    "C2.1": "EU: binding law over voluntary commitments",
    "C5.1": "US: export controls on advanced AI chips",
    "C6.1": "China: training on proprietary model outputs",
    "C8.1": "China: government-mandated content filtering",
    "C7.2": "China: real-time facial recognition by police",
    "C4.1": "China: open release of model weights",
    "C10.1": "US: G7/OECD-led standard-setting",
}
COLS = ["US", "CN", "EU"]
COL_LABEL = {"US": "US", "CN": "China", "EU": "EU"}
LANGS = [("en", "English prompts"), ("zh", "Chinese prompts")]


def matched_shift():
    df = load_master()
    bc = df[df["family"].isin(["B", "C"])]
    bc = bc[~bc["is_refusal"]]
    par = bc[bc.family == "B"][PAIR_KEY + ["issue_id", "stance_parent_scale"]].rename(
        columns={"issue_id": "variant_of", "stance_parent_scale": "sb"})
    kid = bc[bc.family == "C"][PAIR_KEY + ["issue_id", "variant_of", "pair_polarity",
                                           "origin", "stance_parent_scale"]].rename(
        columns={"issue_id": "pair", "stance_parent_scale": "sc"})
    p = kid.merge(par, on=PAIR_KEY + ["variant_of"], how="inner")
    p["shift"] = (p["sc"] - p["sb"]) * p["pair_polarity"].map(NAMED_DIR)
    return p[p["pair"].isin(ROW_LABEL)]


def main():
    apply_style()
    p = matched_shift()
    # shared row order: sort by pooled (both-language) mean, most critical on top
    order = p.groupby("pair")["shift"].mean().sort_values().index.tolist()

    cmap = LinearSegmentedColormap.from_list(
        "shift", [STANCE_COLOR[5], "#ffffff", STANCE_COLOR[1]])
    vmax = 0.7
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.0), sharey=True)
    for ax, (lang, title) in zip(axes, LANGS):
        g = (p[p.lang == lang].groupby(["pair", "origin"])["shift"].mean()
             .reset_index())
        piv = g.pivot(index="pair", columns="origin", values="shift").reindex(order)[COLS]
        M = piv.to_numpy()
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i, pair in enumerate(order):
            for j, origin in enumerate(COLS):
                val = M[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7.5, fontweight="normal",
                        color="white" if abs(val) > 0.42 else INK, zorder=3)
        ax.set_xticks(range(len(COLS)), [COL_LABEL[c] for c in COLS], fontsize=8.5)
        ax.xaxis.set_ticks_position("top")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=18)
        ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_yticks(range(len(order)), [ROW_LABEL[p_] for p_ in order], fontsize=7.6)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, ticks=[-0.6, 0, 0.6])
    cbar.ax.set_yticklabels(["more\ncritical", "no\nchange", "more\nfavorable"], fontsize=7)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.34, right=0.88, top=0.88, bottom=0.06, wspace=0.22)
    save(fig, FIG / "nb3h_rq3_heatmap")
    print("wrote nb3h_rq3_heatmap")
    for lang, _ in LANGS:
        g = p[p.lang == lang].groupby("pair")["shift"].mean().reindex(order)
        print(f"[{lang}] con-ward {(g < 0).sum()}/{g.notna().sum()}")


if __name__ == "__main__":
    main()
