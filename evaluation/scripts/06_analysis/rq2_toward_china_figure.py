"""Draw the per-issue Chinese-versus-English shift toward China's position.

Reads the issue-level estimates written by ``rq1.py``.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
from viz_style import GRID, INK, INK_MUTED, ORIGIN_COLOR, SLOT, apply_style, save  # noqa: E402

FIG = Path("evaluation/analysis/outputs/figures")
TAB = Path("evaluation/analysis/outputs/tables")

# short descriptive labels (no bare issue IDs in the reader-facing figure)
LABEL = {
    "B1": "Pre-deployment government authorization",
    "B2": "Binding law over voluntary commitments",
    "B3": "Mandatory training-data disclosure",
    "B4": "Open release of model weights",
    "B5": "Export controls on advanced AI chips",
    "B7": "Real-time facial recognition by police",
    "B8": "Government-mandated content filtering",
    "B9": "UN as the main channel for AI governance",
    "B10": "G7/OECD-led standard-setting",
}
# issues where China's pole is the anti-regulation / anti-proposition side
AGAINST_PROPOSITION = {"B5", "B10"}

apply_style()

d = pd.read_csv(TAB / "nb2c_toward_china_per_issue.csv")
d["label"] = d["issue"].map(LABEL)
d = d.sort_values("toward_china", ascending=True).reset_index(drop=True)
mean_shift = d["toward_china"].mean()

BLUE = SLOT[0]
fig, ax = plt.subplots(figsize=(6.6, 4.4))

for i, r in d.iterrows():
    ax.plot([0, r["toward_china"]], [i, i], color=BLUE, lw=2.0, zorder=2,
            solid_capstyle="round")
    ax.scatter(r["toward_china"], i, s=64, color=BLUE, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.text(r["toward_china"] + 0.012, i, f"+{r['toward_china']:.2f}",
            va="center", ha="left", fontsize=8, color=INK)

ax.axvline(0, color=GRID, lw=1.0, zorder=1)
ax.axvline(mean_shift, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(mean_shift - 0.008, 7.5, f"mean +{mean_shift:.2f}", rotation=0,
        va="center", ha="right", fontsize=7.5, color=INK_MUTED)

ax.set_yticks(range(len(d)))
ax.set_yticklabels(d["label"], fontsize=10, color=INK)
ax.set_xlim(0, max(d["toward_china"]) + 0.09)
ax.set_xlabel("shift toward China's position when prompted in Chinese\n"
              "(mean stance points, Chinese minus English)")
ax.set_title("Chinese prompting moves stance toward China\non every codeable issue",
             pad=10)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)

save(fig, FIG / "nb2e_toward_china_lollipop")
print("wrote", FIG / "nb2e_toward_china_lollipop.pdf")
print(d[["issue", "label", "toward_china"]].to_string(index=False))
print(f"mean shift = {mean_shift:.3f}")
