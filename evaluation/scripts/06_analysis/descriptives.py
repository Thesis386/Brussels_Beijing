"""Descriptive coverage, refusal, stance-distribution, and template-effect checks.

Reads ``master.csv`` and writes tables and figures to ``analysis/outputs``.
"""

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402
from viz_style import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    MODEL_MARKER,
    MODEL_ORDER,
    MODEL_ORIGIN,
    ORIGIN_COLOR,
    REFUSAL_COLOR,
    STANCE_COLOR,
    STANCE_LABEL,
    apply_style,
    save,
)

TABLES = Path("evaluation/analysis/outputs/tables")
FIG = Path("evaluation/analysis/outputs/figures")
ISSUES_CSV = Path("evaluation/prompt_ingredients/issues.csv")
TABLES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

STANCES = [1, 2, 3, 4, 5]

# Byte-identical across the AB/BC stance-distribution sections in the original
# scripts -- defined once here and shared by both.
MODEL_DISPLAY = {
    "claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
    "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral",
}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pmf(s: pd.Series) -> np.ndarray:
    v = s.dropna()
    if len(v) == 0:
        return np.full(5, np.nan)
    return np.array([(v == k).sum() for k in STANCES], dtype=float) / len(v)


def _apply_white_style() -> None:
    """apply_style() + the pure-white background override both AB and BC used."""
    apply_style()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# =====================================================================================
# 1. nb0 -- coverage, refusals, stance distribution, rep-to-rep reliability
# =====================================================================================
#
# This is the prerequisite for every RQ notebook. It establishes:
#   1. Coverage  — did we actually collect what the design says we collected?
#   2. Refusals  — the denominators. Stance means are conditioned on non-refusals, and
#                  refusal rates differ by condition, so this table has to be reported
#                  alongside every stance result (pre-specified in main.tex).
#   3. Stance distribution — how much of the corpus sits at each scale point.
#   4. Rep-to-rep reliability — the same prompt run 3x at temperature 1.0. This is the
#                  measurement-noise floor: no origin/language effect is interpretable
#                  if it is smaller than the run-to-run wobble.


def run_coverage_refusals() -> None:
    df = load_master()

    # ---------------------------------------------------------------- 1. coverage
    section("1. COVERAGE")

    print(f"rows: {len(df):,}")
    expected = 100 * 30 * 2 * 6 * 3  # templates x issues x langs x models x reps
    print(f"expected by design: {expected:,}   ({'MATCH' if len(df) == expected else 'MISMATCH'})")

    print("\nrows per rep x model:")
    cov = df.pivot_table(index="model", columns="rep", values="response_id", aggfunc="count")
    print(cov)

    print("\nissues per family (should be A=6, B=11, C=13):")
    print(df.groupby("family")["issue_id"].nunique())

    # a stance is missing if the judge returned nothing usable
    missing = df["stance"].isna().sum()
    print(f"\nrows with no judge label at all: {missing:,}")

    # non-'stop' finish reasons: truncation would bias stance (a cut-off response reads
    # more one-sided than it is), so confirm these are negligible
    fr = df["finish_reason"].value_counts(dropna=False)
    print(f"\nfinish_reason:\n{fr}")

    # ---------------------------------------------------------------- 2. refusals
    section("2. REFUSAL RATES (%)  — the denominators for every stance mean")

    df["refusal_pct"] = df["is_refusal"].astype(float) * 100

    ref_ml = df.pivot_table(index="model", columns="lang", values="refusal_pct", aggfunc="mean").round(2)
    ref_ml["all"] = df.groupby("model")["refusal_pct"].mean().round(2)
    ref_ml = ref_ml.join(df.groupby("model")["origin"].first())
    ref_ml = ref_ml.sort_values(["origin", "all"])
    print("\nby model x language:")
    print(ref_ml)
    ref_ml.to_csv(TABLES / "nb0_refusal_by_model_lang.csv")

    print("\nby origin x language:")
    ref_ol = df.pivot_table(index="origin", columns="lang", values="refusal_pct", aggfunc="mean").round(2)
    print(ref_ol)
    ref_ol.to_csv(TABLES / "nb0_refusal_by_origin_lang.csv")

    print("\nby family x language (are refusals concentrated in the contested B-series?):")
    print(df.pivot_table(index="family", columns="lang", values="refusal_pct", aggfunc="mean").round(2))

    print("\ntop 10 issue x lang cells by refusal rate:")
    top = (
        df.groupby(["issue_id", "issue_category", "lang"])["refusal_pct"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(1)
    )
    print(top.to_frame())

    # ---------------------------------------------- 3. stance distribution
    section("3. STANCE DISTRIBUTION (% of all responses, refusals included)")

    dist = (
        df.groupby("model")["stance"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .unstack(fill_value=0)
    )
    print(dist)
    dist.to_csv(TABLES / "nb0_stance_distribution.csv")

    print("\nOverall, non-refusals only:")
    print(df["stance_num"].value_counts(normalize=True).mul(100).round(1).sort_index())

    # --------------------------------------------- 4. rep-to-rep reliability
    section("4. REP-TO-REP RELIABILITY  — the measurement noise floor")

    # Each prompt_id x model was run 3x at temperature 1.0. Spread across reps is
    # sampling noise, not signal. Any condition effect smaller than this is not real.
    wide = df.pivot_table(
        index=["model", "prompt_id"], columns="rep", values="stance_num", aggfunc="first"
    )
    wide = wide.dropna()  # cells where all three reps produced a non-refusal stance

    print(f"prompt x model cells with a stance in all 3 reps: {len(wide):,}")

    wide["sd"] = wide[["real_r1", "real_r2", "real_r3"]].std(axis=1)
    wide["all_same"] = wide[["real_r1", "real_r2", "real_r3"]].nunique(axis=1) == 1

    rel = wide.groupby("model").agg(
        mean_within_cell_sd=("sd", "mean"),
        pct_all_3_identical=("all_same", lambda s: 100 * s.mean()),
    ).round(2)
    print("\n", rel)
    rel.to_csv(TABLES / "nb0_rep_reliability.csv")

    # Correlation between reps at the level things are actually reported at:
    # the model x issue x lang cell mean. This is the number that matters — noise
    # averages out over 100 templates, so cell means are far more stable than
    # individual responses.
    cell = df.pivot_table(
        index=["model", "issue_id", "lang"], columns="rep", values="stance_num", aggfunc="mean"
    ).dropna()
    print(f"\nmodel x issue x lang cell means (n={len(cell)}), correlation between reps:")
    print(cell.corr().round(3))

    print(f"\n\nTables written to {TABLES}/")


# =====================================================================================
# 2. nb0c2 -- combined EN + ZH stance-distribution figure, A vs B issues
# =====================================================================================
#
# One figure, two panels sharing the same rows: English responses on the left,
# Chinese responses on the right. One diverging stacked bar per model within each
# A/B issue, neutral bin straddling the origin, refusals hatched. The issue wording
# in the gutter is shown in ENGLISH only, once, on the far left; both panels share
# it. Background is pure white.

# Short issue labels, matching the vocabulary already used in the RQ2 lollipop and
# the RQ3 heatmap. The full policy wording lives in Appendix A; printing it here cost
# roughly twice the words for no extra information.
ISSUE_SHORT = {
    "A1": "Pre-deployment safety evaluations",
    "A2": "Human oversight of high-stakes decisions",
    "A3": "Restrictions on bio/chem weapons uses",
    "A4": "Documentation and record-keeping",
    "A5": "Protections for children and minors",
    "A6": "Baseline developer accountability",
    "B1": "Pre-deployment government authorization",
    "B2": "Binding law over voluntary commitments",
    "B3": "Mandatory training-data disclosure",
    "B4": "Open release of model weights",
    "B5": "Export controls on advanced AI chips",
    "B6": "Training on proprietary model outputs",
    "B7": "Real-time facial recognition by police",
    "B8": "Government-mandated content filtering",
    "B9": "UN as the main channel for AI governance",
    "B10": "G7/OECD-led standard-setting",
    "B11": "Deregulation to boost domestic AI",
}


def natural_key(issue_id: str):
    fam, num = issue_id[0], issue_id[1:]
    return (fam, int(num))


def issue_labels_ab() -> dict:
    i = pd.read_csv(ISSUES_CSV)
    i = i[(i["framing"] == "neutral") & (i["family"].isin(["A", "B"]))]
    missing = sorted(set(i["issue_id"]) - set(ISSUE_SHORT))
    if missing:
        raise ValueError(f"no short label for {missing}")
    return {iid: ISSUE_SHORT[iid] for iid in i["issue_id"]}


def draw_panel_ab(ax, df, issues, ypos, lang):
    """Diverging stacked bars for one language onto a shared y-layout."""
    d = df[(df["family"].isin(["A", "B"])) & (df["lang"] == lang)]
    rows = []
    for iss in issues:
        for mdl in MODEL_ORDER:
            g = d[(d["issue_id"] == iss) & (d["model"] == mdl)]
            rows.append((pmf(g["stance_num"]), float(g["is_refusal"].mean())))

    for (p, ref), yy in zip(rows, ypos):
        if np.isnan(p).any():
            continue
        left = -(p[0] + p[1] + p[2] / 2)  # centre the neutral bin on the origin
        for k, share in zip(STANCES, p):
            if share <= 0:
                continue
            ax.barh(yy, share, left=left, height=0.86, color=STANCE_COLOR[k],
                    edgecolor="white", linewidth=0.8, zorder=3)
            left += share
        if ref > 0:
            ax.barh(yy, ref, left=left + 0.03, height=0.86, color=REFUSAL_COLOR,
                    edgecolor="white", linewidth=0.8, hatch="///", zorder=3)

    # neutral line spans only the bar area, not the full axes -- so top/bottom margin
    # can add whitespace without the line growing up toward the title
    ax.plot([0, 0], [min(ypos) - 0.7, max(ypos) + 0.7],
            color=INK_MUTED, lw=0.9, zorder=4, alpha=0.7, clip_on=False)
    # trim the mostly-empty con (right) side so the two panels sit closer
    ax.set_xlim(-1.06, 1.02)
    ax.set_xticks([-1, 0, 1], ["", "3", ""])  # center = neutral, i.e. stance 3 on the 1-5 scale
    ax.set_xlabel("<-- pro    con -->", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(False)  # no gridlines at all; the only vertical rule is the axvline at 0
    ax.set_axisbelow(True)
    for sp in ax.spines.values():   # no box/left/between-panel spines; only the axvline stays
        sp.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def run_stance_distributions_ab() -> None:
    _apply_white_style()

    df = load_master()
    labels = issue_labels_ab()
    issues = sorted(
        df[df["family"].isin(["A", "B"])]["issue_id"].unique(), key=natural_key
    )

    # shared y-layout: one row per (issue, model), extra gap where family changes
    ypos, ylabels, boundaries, family_gap = [], [], [], None
    y, prev_fam = 0.0, None
    issue_centre = {}
    for iss in issues:
        fam = iss[0]
        if prev_fam is not None:
            if fam != prev_fam:
                y += 1.35
                family_gap = y - 0.6
            else:
                y += 0.95  # more vertical room between issues so wording stops stacking
                boundaries.append(y - 0.48)
        for mdl in MODEL_ORDER:
            ypos.append(y)
            ylabels.append(MODEL_DISPLAY[mdl])
            y += 1.0
        issue_centre[iss] = ypos[-len(MODEL_ORDER)] + (len(MODEL_ORDER) - 1) / 2.0
        prev_fam = fam

    n_rows = len(ypos)
    h = 0.072 * y + 1.5  # taller, to fill the portrait page
    W = 7.4
    fig, (ax_en, ax_zh) = plt.subplots(
        1, 2, figsize=(W, h), sharey=True,
        gridspec_kw={"wspace": 0.008},
    )

    draw_panel_ab(ax_en, df, issues, ypos, "en")
    draw_panel_ab(ax_zh, df, issues, ypos, "zh")

    # the only horizontal rule: a soft separator between the A and B blocks
    for ax in (ax_en, ax_zh):
        if family_gap is not None:
            ax.axhline(family_gap, color="#c4c3bf", lw=0.9, zorder=2)

    # The six rows in every issue block are always in MODEL_ORDER, so the names are
    # carried once by the key below rather than repeated 17 times down the axis.
    ax_en.set_yticks(ypos, [""] * len(ypos))
    ax_en.tick_params(axis="y", length=0)
    ax_en.invert_yaxis()
    # clamp tight to the bars: kills the top/bottom whitespace and stops the neutral
    # line from extending up into the title (sharey applies this to both panels)
    ax_en.set_ylim(max(ypos) + 1.6, min(ypos) - 3.2)  # extra whitespace above the first row

    # left gutter uses FIGURE-fraction x (robust to the narrow panels) + data-y.
    # No per-issue id codes; the English wording labels each issue, once, for both panels.
    # Wording sits close to the model labels to cut the horizontal gap.
    fig.canvas.draw()  # so transData is valid before we build the blended transform
    gutter = blended_transform_factory(fig.transFigure, ax_en.transData)
    for iss in issues:
        wrapped = textwrap.fill(labels.get(iss, iss), width=26)
        ax_en.text(0.262, issue_centre[iss], wrapped, transform=gutter, ha="right",
                   va="center", fontsize=6.4, color=INK_MUTED, zorder=5)
    # A/B block labels sit right up against the (now wider) wording
    for fam, name in [("A", "A baselines"), ("B", "B contested")]:
        fam_rows = [issue_centre[i] for i in issues if i[0] == fam]
        ax_en.text(0.047, np.mean(fam_rows), name, transform=gutter, ha="center",
                   va="center", rotation=90, fontsize=8, fontweight="bold", color=INK,
                   zorder=5)
    if family_gap is not None:
        ax_en.plot([0.04, 0.29], [family_gap, family_gap], transform=gutter,
                   color="#c4c3bf", lw=0.8, zorder=5, clip_on=False)

    ax_en.set_title("English prompts", pad=3, fontsize=9.5, fontweight="bold")
    ax_zh.set_title("Chinese prompts", pad=3, fontsize=9.5, fontweight="bold")

    handles = [Patch(facecolor=STANCE_COLOR[k], label=STANCE_LABEL[k]) for k in STANCES]
    handles.append(Patch(facecolor=REFUSAL_COLOR, hatch="///", label="refusal"))
    fig.legend(handles=handles, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), handlelength=1.3, columnspacing=1.2,
               fontsize=8.5, frameon=False)
    key = "  ·  ".join(MODEL_DISPLAY[m] for m in MODEL_ORDER)
    fig.text(0.5, 0.0, f"Six rows per issue, top to bottom:  {key}",
             ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)

    fig.subplots_adjust(left=0.28, right=0.985, top=0.965, bottom=0.10)
    # bbox_inches="tight" trims the surrounding whitespace (incl. the bottom gap)
    out = FIG / "nb0c_stance_diverging_AB_combined"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"figure -> {out}.pdf|png  ({W:.1f}x{h:.1f} in, portrait)")


# =====================================================================================
# 3. nb0d2 -- combined EN + ZH stance-distribution figure, B parent vs C variant (RQ3)
# =====================================================================================
#
# nb0c2's figure reused for the B<->C attribution comparison. Instead of A baselines
# vs B contested, each block here is a neutrally-phrased B parent followed directly
# by its named-jurisdiction C variant, so an attribution shift shows up as the
# C sub-block sliding pro or con relative to the B sub-block right above it.
#
# Restricted to the eight CLEAN minimal-label pairs (the jurisdiction is attached to
# an otherwise-unchanged policy); the three actor/role and two reverse-phrased
# variants are left to the full item-level table. Stance is drawn on
# stance_parent_scale so "pro" always means pro-the-parent policy (a no-op for these
# eight, none of which are reversed).

# the eight clean minimal-label pairs, each as (B parent, C variant), in issue order
PAIRS = [
    ("B1", "C1.1"), ("B2", "C2.1"), ("B3", "C3.1"), ("B4", "C4.1"),
    ("B5", "C5.1"), ("B7", "C7.2"), ("B8", "C8.1"), ("B11", "C11.1"),
]
ORDER = [iss for pair in PAIRS for iss in pair]          # [B1, C1.1, B2, C2.1, ...]
BLOCK_OF = {iss: k for k, pair in enumerate(PAIRS) for iss in pair}
IS_VARIANT = {c: True for _, c in PAIRS}


def issue_labels_bc() -> dict:
    i = pd.read_csv(ISSUES_CSV)
    i = i[(i["framing"] == "neutral") & (i["family"].isin(["B", "C"]))]
    return {r.issue_id: r.issue_en for r in i.itertuples()}


def draw_panel_bc(ax, df, ypos, lang):
    d = df[(df["issue_id"].isin(ORDER)) & (df["lang"] == lang)]
    rows = []
    for iss in ORDER:
        for mdl in MODEL_ORDER:
            g = d[(d["issue_id"] == iss) & (d["model"] == mdl)]
            rows.append((pmf(g["stance_parent_scale"]), float(g["is_refusal"].mean())))

    for (p, ref), yy in zip(rows, ypos):
        if np.isnan(p).any():
            continue
        left = -(p[0] + p[1] + p[2] / 2)  # centre the neutral bin on the origin
        for k, share in zip(STANCES, p):
            if share <= 0:
                continue
            ax.barh(yy, share, left=left, height=0.86, color=STANCE_COLOR[k],
                    edgecolor="white", linewidth=0.8, zorder=3)
            left += share
        if ref > 0:
            ax.barh(yy, ref, left=left + 0.03, height=0.86, color=REFUSAL_COLOR,
                    edgecolor="white", linewidth=0.8, hatch="///", zorder=3)

    ax.plot([0, 0], [min(ypos) - 0.7, max(ypos) + 0.7],
            color=INK_MUTED, lw=0.9, zorder=4, alpha=0.7, clip_on=False)
    ax.set_xlim(-1.06, 1.02)
    ax.set_xticks([-1, 0, 1], ["", "3", ""])
    ax.set_xlabel("<-- pro    con -->", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(False)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def save_single_language(
    df,
    labels,
    ypos,
    ylabels,
    block_gaps,
    within_lines,
    issue_centre,
    height,
    lang,
):
    """Save one full-page language panel for the appendix."""
    lang_name = "English" if lang == "en" else "Chinese"
    fig, ax = plt.subplots(figsize=(7.4, height))
    draw_panel_bc(ax, df, ypos, lang)

    for gap in block_gaps:
        ax.axhline(gap, color="#c4c3bf", lw=0.9, zorder=2)
    for line in within_lines:
        ax.axhline(line, color="#e6e5e1", lw=0.7, zorder=1)

    ax.set_yticks(ypos, ylabels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_ylim(max(ypos) + 1.6, min(ypos) - 3.2)

    fig.canvas.draw()
    gutter = blended_transform_factory(fig.transFigure, ax.transData)
    for iss in ORDER:
        wrapped = textwrap.fill(labels.get(iss, iss), width=31)
        weight = "normal" if IS_VARIANT.get(iss) else "bold"
        color = INK if IS_VARIANT.get(iss) else INK_MUTED
        ax.text(0.305, issue_centre[iss], wrapped, transform=gutter, ha="right",
                va="center", fontsize=6.8, color=color, fontweight=weight, zorder=5)
    for b, c in PAIRS:
        mid = (issue_centre[b] + issue_centre[c]) / 2
        ax.text(0.05, mid, "parent\nvs.\nnamed", transform=gutter, ha="center",
                va="center", fontsize=6.4, color=INK_MUTED, linespacing=1.4, zorder=5)

    ax.set_title(f"{lang_name} prompts", pad=3, fontsize=10.5, fontweight="bold")
    handles = [Patch(facecolor=STANCE_COLOR[k], label=STANCE_LABEL[k]) for k in STANCES]
    handles.append(Patch(facecolor=REFUSAL_COLOR, hatch="///", label="refusal"))
    fig.legend(handles=handles, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handlelength=1.3, columnspacing=1.2,
               fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.39, right=0.985, top=0.97, bottom=0.09)

    out = FIG / f"nb0d2_stance_diverging_BC_clean_{lang}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"figure -> {out}.pdf|png  (full-page {lang_name} panel)")


def run_stance_distributions_bc() -> None:
    _apply_white_style()
    df = load_master()
    labels = issue_labels_bc()

    # shared y-layout: one row per (issue, model); big gap between pair-blocks,
    # a thin separator between the B parent and its C variant within a block.
    ypos, ylabels = [], []
    block_gaps, within_lines = [], []
    issue_centre = {}
    y, prev_block, prev_iss = 0.0, None, None
    for iss in ORDER:
        blk = BLOCK_OF[iss]
        if prev_iss is not None:
            if blk != prev_block:
                y += 1.35
                block_gaps.append(y - 0.6)
            else:
                y += 0.95
                within_lines.append(y - 0.48)
        for mdl in MODEL_ORDER:
            ypos.append(y)
            ylabels.append(MODEL_DISPLAY[mdl])
            y += 1.0
        issue_centre[iss] = ypos[-len(MODEL_ORDER)] + (len(MODEL_ORDER) - 1) / 2.0
        prev_block, prev_iss = blk, iss

    h = 0.072 * y + 1.5
    W = 7.4
    fig, (ax_en, ax_zh) = plt.subplots(
        1, 2, figsize=(W, h), sharey=True, gridspec_kw={"wspace": 0.008},
    )
    draw_panel_bc(ax_en, df, ypos, "en")
    draw_panel_bc(ax_zh, df, ypos, "zh")

    for ax in (ax_en, ax_zh):
        for g in block_gaps:
            ax.axhline(g, color="#c4c3bf", lw=0.9, zorder=2)
        for t in within_lines:
            ax.axhline(t, color="#e6e5e1", lw=0.7, zorder=1)

    ax_en.set_yticks(ypos, ylabels, fontsize=5.0)
    ax_en.invert_yaxis()
    ax_en.set_ylim(max(ypos) + 1.6, min(ypos) - 3.2)

    fig.canvas.draw()
    gutter = blended_transform_factory(fig.transFigure, ax_en.transData)
    for iss in ORDER:
        wrapped = textwrap.fill(labels.get(iss, iss), width=29)
        weight = "normal" if IS_VARIANT.get(iss) else "bold"
        color = INK if IS_VARIANT.get(iss) else INK_MUTED
        ax_en.text(0.235, issue_centre[iss], wrapped, transform=gutter, ha="right",
                   va="center", fontsize=5.7, color=color, fontweight=weight, zorder=5)
    # per-block bracket label: parent (grey) above, named variant (ink) below
    for k, (b, c) in enumerate(PAIRS):
        mid = (issue_centre[b] + issue_centre[c]) / 2
        ax_en.text(0.045, mid, "parent\nvs.\nnamed", transform=gutter, ha="center",
                   va="center", fontsize=5.6, color=INK_MUTED, linespacing=1.4, zorder=5)

    ax_en.set_title("English prompts", pad=3, fontsize=9.5, fontweight="bold")
    ax_zh.set_title("Chinese prompts", pad=3, fontsize=9.5, fontweight="bold")

    handles = [Patch(facecolor=STANCE_COLOR[k], label=STANCE_LABEL[k]) for k in STANCES]
    handles.append(Patch(facecolor=REFUSAL_COLOR, hatch="///", label="refusal"))
    fig.legend(handles=handles, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handlelength=1.3, columnspacing=1.2,
               fontsize=8.5, frameon=False)

    fig.subplots_adjust(left=0.30, right=0.985, top=0.965, bottom=0.09)
    out = FIG / "nb0d2_stance_diverging_BC_clean"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"figure -> {out}.pdf|png  ({W:.1f}x{h:.1f} in, portrait)")

    for lang in ("en", "zh"):
        save_single_language(
            df, labels, ypos, ylabels, block_gaps, within_lines,
            issue_centre, h, lang,
        )


# =====================================================================================
# 4. nb0e -- the template effect: how the four writing-task categories shape stance
# =====================================================================================
#
# Every prompt wraps an issue in one of 100 IssueBench templates, 25 in each of four
# categories (Academic/Analytical, Argumentative, Journalistic/Commentary,
# Policy/Professional). nb1 treats the template as a random effect -- noise to average
# over. This asks the descriptive question directly: does the KIND of writing task move
# the stance, holding the issue fixed?
#
# Three things are measured, each pooled the right way:
#
# 1. Decisiveness. The share of answers that sit neutral (3) vs. lean (2/4) vs. take a
#    firm side (1/5), FOLDED so pro and con collapse together. Folding removes the
#    per-issue polarity, so this is one of the few things that pools legitimately across
#    all 30 issues. Refusals are appended, hatched, as in nb0b.
#
# 2. Refusal rate, per category x language -- the format itself changes how often a
#    model declines.
#
# 3. Direction. Whether a category pulls pro or con RELATIVE TO the same issue's overall
#    mean. Computed within each (issue, language) cell -- never on raw pooled stance,
#    whose pro direction differs by issue -- with the issue as the bootstrap unit.

CATS = ["Academic/Analytical", "Journalistic/Commentary", "Argumentative", "Policy/Professional"]
LANGS = ["en", "zh"]
LANG_NAME = {"en": "English", "zh": "Chinese"}

# folded decisiveness ramp: same distance-from-neutral idea, no pro/con hue
FOLD_COLOR = {"neutral": "#d8d7d3", "leans": "#9db8d8", "firm": "#2f5896"}
FOLD_LABEL = {"neutral": "neutral (3)", "leans": "leans (2/4)", "firm": "firm side (1/5)"}


def fold(stance: pd.Series) -> dict:
    """Share neutral / leaning / firm among answered responses (polarity folded out)."""
    v = stance.dropna()
    n = len(v)
    if n == 0:
        return {"neutral": np.nan, "leans": np.nan, "firm": np.nan}
    return {
        "neutral": (v == 3).mean(),
        "leans": v.isin([2, 4]).mean(),
        "firm": v.isin([1, 5]).mean(),
    }


# ------------------------------------------------------------------ decisiveness + refusal


def fig_decisiveness(
    df: pd.DataFrame,
    out_name: str = "nb0e_template_decisiveness",
    scope_label: str | None = None,
) -> None:
    """Grouped horizontal stacked bars: folded stance composition per category x language."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ypos, ylabels, rows = [], [], []
    y = 0.0
    boundaries = []
    for ci, cat in enumerate(CATS):
        if ci:
            boundaries.append(y - 0.30)
        for lang in LANGS:
            d = df[(df["template_category"] == cat) & (df["lang"] == lang)]
            rows.append(fold(d["stance_num"]))
            ypos.append(y)
            ylabels.append(LANG_NAME[lang])
            y += 1.0
        y += 0.7

    for comp, yy in zip(rows, ypos):
        left = 0.0
        for key in ["firm", "leans", "neutral"]:  # firm at the left, neutral at the right
            share = comp[key]
            ax.barh(yy, share, left=left, height=0.66, color=FOLD_COLOR[key],
                    edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
            if share > 0.06:
                ax.text(left + share / 2, yy, f"{share*100:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if key == "firm" else INK, zorder=4)
            left += share

    for b in boundaries:
        ax.axhline(b, color=GRID, lw=0.8, zorder=1)
    ax.set_yticks(ypos, ylabels)
    ax.invert_yaxis()

    # category labels in the left gutter
    gutter = blended_transform_factory(ax.transAxes, ax.transData)
    for ci, cat in enumerate(CATS):
        centre = ci * 2.7 + 0.5
        ax.text(-0.30, centre, cat.replace("/", " /\n"), transform=gutter, ha="left",
                va="center", fontsize=8.5, fontweight="bold", color=INK)

    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of answered responses  (refusals negligible: <1% in every cell)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=FOLD_COLOR[k], label=FOLD_LABEL[k]) for k in ["neutral", "leans", "firm"]]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              handlelength=1.1, columnspacing=1.2)
    title = ("Does the writing task make the model take a side?\n"
             "Folded stance composition by template category")
    if scope_label:
        title += f" — {scope_label}"
    ax.set_title(title, pad=10)
    fig.subplots_adjust(left=0.26)
    save(fig, FIG / out_name)


# ------------------------------------------------------------------ direction (within-issue)


def within_issue_direction(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Per category x lang: mean within-issue stance deviation, with issue-bootstrap CI."""
    a = df.dropna(subset=["stance_num"]).copy()
    a["iss_mean"] = a.groupby(["issue_id", "lang"])["stance_num"].transform("mean")

    out = []
    for lang in LANGS:
        sub = a[a["lang"] == lang]
        issues = sub["issue_id"].unique()
        # per issue x category deviation of the category mean from the issue mean
        per = (
            sub.groupby(["issue_id", "template_category"])
            .apply(lambda g: g["stance_num"].mean() - g["iss_mean"].iloc[0], include_groups=False)
            .rename("dev").reset_index()
        )
        wide = per.pivot(index="issue_id", columns="template_category", values="dev")
        for cat in CATS:
            vals = wide[cat].dropna().values
            point = vals.mean()
            boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({"template_category": cat, "lang": lang,
                        "dev_mean": point, "ci_lo": lo, "ci_hi": hi, "n_issues": len(vals)})
    return pd.DataFrame(out)


def fig_direction(dirn: pd.DataFrame) -> None:
    """Dot plot: within-issue stance deviation (pro <-- 0 --> con) per category, EN vs ZH."""
    lang_color = {"en": "#2a78d6", "zh": "#e34948"}  # two well-separated hues, EN vs ZH

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ypos = {cat: len(CATS) - 1 - i for i, cat in enumerate(CATS)}
    offset = {"en": 0.14, "zh": -0.14}

    for _, r in dirn.iterrows():
        y = ypos[r["template_category"]] + offset[r["lang"]]
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=lang_color[r["lang"]], lw=1.6, zorder=2)
        ax.scatter(r["dev_mean"], y, color=lang_color[r["lang"]], s=42, zorder=3,
                   edgecolor="white", linewidth=0.8)

    ax.axvline(0, color=INK_MUTED, lw=0.9, alpha=0.7, zorder=1)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()))
    ax.set_ylim(-0.6, len(CATS) - 0.4)
    ax.set_xlabel("within-issue stance deviation      <-- more pro        more con -->")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([0], [0], marker="o", color=lang_color[l], lw=0, markersize=7,
                          markeredgecolor="white", label=LANG_NAME[l]) for l in LANGS]
    ax.legend(handles=handles, loc="lower right", title="prompt language")
    ax.set_title("Which writing tasks pull pro vs. con? (relative to each issue's own mean)", pad=10)
    fig.subplots_adjust(left=0.22)
    save(fig, FIG / "nb0e_template_direction")


# ------------------------------------------------------------------ per-model robustness


def fig_by_model(df: pd.DataFrame) -> None:
    """Firm-side share across the four categories, one line per model -- same shape, all six."""
    a = df.dropna(subset=["stance_num"]).copy()
    a["firm"] = a["stance_num"].isin([1, 5]).astype(int)
    share = a.groupby(["model", "template_category"])["firm"].mean() * 100

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = range(len(CATS))
    for mdl in MODEL_ORDER:
        y = [share.get((mdl, c), np.nan) for c in CATS]
        col = ORIGIN_COLOR[MODEL_ORIGIN[mdl]]
        ax.plot(x, y, color=col, lw=1.4, marker=MODEL_MARKER[mdl], markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3, label=mdl)

    ax.set_xticks(list(x), ["Academic /\nAnalytical", "Journalistic /\nCommentary",
                            "Argumentative", "Policy /\nProfessional"])
    ax.set_xlim(-0.25, len(CATS) - 0.75)
    ax.set_ylim(0, 70)
    ax.set_ylabel("firm-side responses (stance 1 or 5, %)")
    ax.set_title("Every model takes a side most under the Policy/Professional task\n"
                 "(hue = origin: blue US, green CN, amber EU)", pad=10)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, title="model", handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16)
    save(fig, FIG / "nb0e_template_by_model")


def run_template_effect() -> None:
    apply_style()
    rng = np.random.default_rng(0)

    df = load_master()

    # summary table: decisiveness, refusal, extremity
    a = df.dropna(subset=["stance_num"]).copy()
    a["extreme"] = (a["stance_num"] - 3).abs()
    rows = []
    for cat in CATS:
        for lang in LANGS:
            d = df[(df["template_category"] == cat) & (df["lang"] == lang)]
            comp = fold(d["stance_num"])
            rows.append({
                "template_category": cat, "lang": lang,
                "pct_neutral": round(comp["neutral"] * 100, 1),
                "pct_leans": round(comp["leans"] * 100, 1),
                "pct_firm": round(comp["firm"] * 100, 1),
                "mean_extremity": round(d["stance_num"].sub(3).abs().mean(), 3),
                "pct_refusal": round(d["is_refusal"].mean() * 100, 2),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "nb0e_template_summary.csv", index=False)
    print("DECISIVENESS / REFUSAL by template category x language")
    print(summary.to_string(index=False), "\n")

    dirn = within_issue_direction(df, rng)
    dirn.round(3).to_csv(TABLES / "nb0e_template_direction.csv", index=False)
    print("WITHIN-ISSUE DIRECTION (pro<0 / con>0), issue-bootstrap 95% CI")
    print(dirn.round(3).to_string(index=False), "\n")

    fig_decisiveness(df)
    fig_decisiveness(
        df[df["family"] == "B"],
        out_name="nb0e_template_decisiveness_B",
        scope_label="core issues only",
    )
    fig_direction(dirn)
    fig_by_model(df)
    print(f"figures -> {FIG}/nb0e_template_*.pdf|png")
    print(f"tables  -> {TABLES}/nb0e_template_*.csv")


if __name__ == "__main__":
    run_coverage_refusals()
    run_stance_distributions_ab()
    run_stance_distributions_bc()
    run_template_effect()
