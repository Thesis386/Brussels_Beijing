"""Draw RQ3 jurisdiction-attribution figures from the paired estimates in ``rq3.py``."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
from viz_style import GRID, INK, INK_MUTED, ORIGIN_COLOR, apply_style, save  # noqa: E402

TAB = Path("evaluation/analysis/outputs/tables")
FIG = Path("evaluation/analysis/outputs/figures")

# descriptive pair labels (no bare issue IDs in reader-facing figures)
PAIR_LABEL = {
    "C1.1": "Pre-deployment government authorization",
    "C2.1": "Binding law over voluntary commitments",
    "C3.1": "Mandatory training-data disclosure",
    "C4.1": "Open release of model weights",
    "C5.1": "Export controls on advanced AI chips",
    "C6.1": "Training on proprietary model outputs",
    "C7.2": "Real-time facial recognition by police",
    "C8.1": "Government-mandated content filtering",
    "C9.1": "UN as the main channel for AI governance",
    "C10.1": "G7/OECD-led standard-setting",
    "C11.1": "Deregulation to boost domestic AI",
}
MODEL_LABEL = {
    "claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
    "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral",
}
ORIGIN_OF = {"claude": "US", "gpt4o": "US", "llama": "US",
             "deepseek": "CN", "qwen": "CN", "mistral": "EU"}
# jurisdiction named -> the origin whose models are "home" for it
HOME_ORIGIN = {"China": "CN", "EU": "EU", "US": "US"}
JUR_TITLE = {"China": "China-named policy", "EU": "EU-named policy", "US": "US-named policy"}

apply_style()

long = pd.read_csv(TAB / "nb3c_delta_by_pair_model_long.csv")
aligned = long[long["polarity"] == "aligned"].copy()

# ---------------------------------------------------------------------------
# FIG 1 — home vs rival, per model, grouped by named jurisdiction
# ---------------------------------------------------------------------------
per = (aligned.groupby(["named", "model"], as_index=False)["toward_named"].mean())
per["origin"] = per["model"].map(ORIGIN_OF)
per["is_home"] = per.apply(lambda r: r["origin"] == HOME_ORIGIN[r["named"]], axis=1)

jur_order = ["China", "EU", "US"]  # each row-block
# model order inside a block: home models first, then rivals, each in fixed order
FIXED = ["deepseek", "qwen", "claude", "gpt4o", "llama", "mistral"]

fig, ax = plt.subplots(figsize=(6.8, 6.2))
y = 0.0
yticks, ylabels, block_spans = [], [], []
for jur in jur_order:
    block = per[per["named"] == jur].copy()
    block["ord_home"] = ~block["is_home"]  # home (False) sorts first
    block["ord_fix"] = block["model"].map({m: i for i, m in enumerate(FIXED)})
    block = block.sort_values(["ord_home", "ord_fix"])
    top = y
    for _, r in block.iterrows():
        color = ORIGIN_COLOR[r["origin"]] if r["is_home"] else "#b9b8b4"
        ax.plot([0, r["toward_named"]], [y, y], color=color, lw=2.2, zorder=2,
                solid_capstyle="round")
        ax.scatter(r["toward_named"], y, s=78, color=color, zorder=3,
                   edgecolor="white", linewidth=1.0)
        face = INK if r["is_home"] else INK_MUTED
        weight = "bold" if r["is_home"] else "normal"
        ax.text(-0.02, y, MODEL_LABEL[r["model"]], ha="right", va="center",
                fontsize=8, color=face, fontweight=weight)
        yticks.append(y)
        ylabels.append("")
        y -= 1.0
    block_spans.append((jur, top, y + 1.0))
    y -= 0.9  # gap between blocks

ax.axvline(0, color=INK_MUTED, lw=1.0, zorder=1)
# block titles, to the far left
for jur, top, bot in block_spans:
    mid = (top + bot) / 2
    ax.text(-0.40, mid, JUR_TITLE[jur], ha="center", va="center", rotation=90,
            fontsize=9, fontweight="bold", color=INK)

ax.set_ylim(y + 0.6, 0.8)
ax.set_yticks([])
for s in ["left", "top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_axisbelow(True)
ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.set_xlabel("Shift toward the named jurisdiction's policy\n"
              "(left = the label made the model more critical of it)")
ax.margins(x=0.12)
fig.subplots_adjust(left=0.30, right=0.97, top=0.94, bottom=0.14)
save(fig, FIG / "nb3e_home_vs_rival")
print("wrote nb3e_home_vs_rival")

# ---------------------------------------------------------------------------
# FIG 2 — per-pair attribution effect (the penalty), appendix
# ---------------------------------------------------------------------------
pair = (aligned.groupby(["pair", "named"], as_index=False)["toward_named"].mean())
pair["label"] = pair["pair"].map(PAIR_LABEL)
pair["origin"] = pair["named"].map(HOME_ORIGIN)
pair = pair.sort_values("toward_named").reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
for i, r in pair.iterrows():
    color = ORIGIN_COLOR[r["origin"]]
    ax2.plot([0, r["toward_named"]], [i, i], color=color, lw=2.2, zorder=2,
             solid_capstyle="round")
    ax2.scatter(r["toward_named"], i, s=72, color=color, zorder=3,
                edgecolor="white", linewidth=0.9)
    # value label always just beyond the dot tip, away from zero
    if r["toward_named"] > 0:
        ax2.text(r["toward_named"] + 0.015, i, f"{r['toward_named']:+.2f}",
                 ha="left", va="center", fontsize=7.5, color=INK_MUTED)
    else:
        ax2.text(r["toward_named"] - 0.015, i, f"{r['toward_named']:+.2f}",
                 ha="right", va="center", fontsize=7.5, color=INK_MUTED)

ax2.axvline(0, color=INK_MUTED, lw=1.0, zorder=1)
ax2.set_ylim(-0.6, len(pair) - 0.4)
ax2.set_yticks(range(len(pair)))
ax2.set_yticklabels(pair["label"], fontsize=8.5, color=INK)
for s in ["left", "top", "right"]:
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="y", length=0)
ax2.set_axisbelow(True)
ax2.xaxis.grid(True)
ax2.yaxis.grid(False)
ax2.set_xlabel("Mean shift toward the named jurisdiction's policy\n"
               "(left = naming the jurisdiction made the model more critical)")
ax2.set_xlim(-0.68, 0.42)
# legend for the three jurisdiction colors, in the empty upper-left region
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ORIGIN_COLOR[o],
                      markersize=8, label=lab)
           for o, lab in [("CN", "China named"), ("EU", "EU named"), ("US", "US named")]]
ax2.legend(handles=handles, loc="upper left", fontsize=8, handletextpad=0.2,
           borderaxespad=0.4)
fig2.subplots_adjust(left=0.28, right=0.97, top=0.95, bottom=0.17)
save(fig2, FIG / "nb3e_attribution_penalty")
print("wrote nb3e_attribution_penalty")

# print the aggregates for the write-up
print("\nHome vs rival (mean toward_named, aligned pairs):")
agg = (per.groupby(["named", "is_home"])["toward_named"].mean().unstack())
print(agg.round(3))
