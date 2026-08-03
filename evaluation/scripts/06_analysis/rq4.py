"""RQ4: model differentiation and developer/government stance comparisons.

Reads ``master.csv`` and the two RQ4 evidence tables, then writes results,
figures, and LaTeX exports in dependency order.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evaluation/analysis"))
from viz_style import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    MODEL_ORDER,
    ORIGIN_COLOR,
    MODEL_ORIGIN,
    SURFACE,
    apply_style,
    save,
)

MASTER_PATH = ROOT / "evaluation/analysis/master.csv"
EVIDENCE_PATH = (
    ROOT / "evaluation/dataset_construction/rq4_position_evidence_events.csv"
)
DIVERGENCE_PATH = (
    ROOT / "evaluation/dataset_construction/rq4_developer_government_contrasts.csv"
)
META = ROOT / "evaluation/dataset_construction/issues_metadata.csv"

OUT = ROOT / "evaluation/analysis/outputs/tables"
TABLES = OUT  # same directory, kept under both names used by the source scripts
FIGS = ROOT / "evaluation/analysis/outputs/figures"
TEX_OUT = Path("evaluation/analysis/outputs/tex")
TEX_OUT.mkdir(parents=True, exist_ok=True)
APPENDIX_TEX_OUT = TEX_OUT / "app_rq4_results.tex"

STAKE_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
STAKE_ORDER_LABELS = ["Low", "Medium", "High", "Very High"]
RNG_SEED = 20260723
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 2_000

MODEL_LABEL = {
    "claude": "Claude (US)",
    "gpt4o": "GPT-4o (US)",
    "llama": "Llama (US)",
    "deepseek": "DeepSeek (CN)",
    "qwen": "Qwen (CN)",
    "mistral": "Mistral (EU)",
}

MODEL = {
    "claude": "Claude",
    "gpt4o": "GPT-4o",
    "llama": "Llama",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
}
ISSUE_ORDER = {"B2": 2, "B4": 4, "B5": 5, "B11": 11}

DEVGOV_LABELS = {
    ("B2", "Anthropic"): ("Binding obligations", "Anthropic", "Pro", "Con"),
    ("B2", "OpenAI"): ("Binding obligations", "OpenAI", "Pro", "Con"),
    ("B4", "Anthropic"): ("Open-weight release", "Anthropic", "Con", "Mixed"),
    ("B4", "Meta"): ("Open-weight release", "Meta", "Pro", "Mixed"),
    ("B5", "Anthropic"): ("Chip export controls", "Anthropic", "Pro", "Pro"),
    ("B11", "Anthropic"): ("Reducing developer regulation", "Anthropic", "Con", "Pro"),
}

CELL_INK = "#2b2a27"  # a shade off black: darker than the muted grey, lighter than headers
TOWARD_INK, TOWARD_FILL = "#006b4b", "#dcefe7"
AWAY_INK, AWAY_FILL = "#9f302f", "#f8e1df"


# ---------------------------------------------------------------------------
# Stage 1: differentiation and stance
# ---------------------------------------------------------------------------


def permutation_trend(
    frame: pd.DataFrame, outcome: str, n_permutations: int = N_PERMUTATIONS
) -> dict[str, float]:
    """Spearman trend and a two-sided label-permutation p-value."""
    x = frame["stake_order"].to_numpy()
    y = frame[outcome].to_numpy()
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(RNG_SEED)
    permuted = np.empty(n_permutations)
    for index in range(n_permutations):
        permuted[index] = spearmanr(rng.permutation(x), y).statistic
    p_value = (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (
        n_permutations + 1
    )
    return {
        "n_issues": len(frame),
        "spearman_rho": observed,
        "permutation_p_two_sided": p_value,
        "n_permutations": n_permutations,
    }


def rq4a_differentiation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute issue-level model differentiation and permutation trend tests."""
    subset = data[
        data["family"].isin(["A", "B"])
        & ~data["is_refusal"]
        & data["stance_num"].notna()
    ].copy()

    model_means = (
        subset.groupby(
            ["issue_id", "family", "developer_stake", "lang", "model"],
            observed=True,
        )["stance_num"]
        .mean()
        .rename("model_mean")
        .reset_index()
    )
    by_language = (
        model_means.groupby(
            ["issue_id", "family", "developer_stake", "lang"], observed=True
        )["model_mean"]
        .agg(
            model_sd=lambda values: values.std(ddof=0),
            model_range=lambda values: values.max() - values.min(),
            n_models="count",
        )
        .reset_index()
    )
    if not (by_language["n_models"] == 6).all():
        incomplete = by_language.loc[by_language["n_models"] != 6]
        raise ValueError(f"RQ4a has incomplete model cells:\n{incomplete}")

    issue_level = (
        by_language.groupby(
            ["issue_id", "family", "developer_stake"], observed=True
        )[["model_sd", "model_range"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "model_sd": "mean_cross_model_sd_across_languages",
                "model_range": "mean_cross_model_range_across_languages",
            }
        )
    )
    issue_level["stake_order"] = issue_level["developer_stake"].map(STAKE_ORDER)
    issue_level = issue_level.sort_values(
        ["stake_order", "family", "issue_id"]
    ).reset_index(drop=True)

    tests = []
    for sample_name, sample in [
        ("all_A_B", issue_level),
        ("B_only_sensitivity", issue_level[issue_level["family"] == "B"]),
    ]:
        for outcome in [
            "mean_cross_model_sd_across_languages",
            "mean_cross_model_range_across_languages",
        ]:
            result = permutation_trend(sample, outcome)
            tests.append({"sample": sample_name, "outcome": outcome, **result})
    return issue_level, pd.DataFrame(tests)


def rq4b_baseline_adjustment(data: pd.DataFrame) -> pd.DataFrame:
    """Create language-specific leave-one-B-issue-out relative stance scores."""
    contested = data[
        (data["family"] == "B")
        & ~data["is_refusal"]
        & data["stance_num"].notna()
    ].copy()
    block = ["issue_id", "template_id", "lang", "rep"]
    contested["block_mean"] = contested.groupby(block, observed=True)[
        "stance_num"
    ].transform("mean")
    contested["within_block_relative"] = (
        contested["stance_num"] - contested["block_mean"]
    )

    issue_position = (
        contested.groupby(["issue_id", "model", "lang"], observed=True)
        .agg(
            issue_relative=("within_block_relative", "mean"),
            issue_stance_raw=("stance_num", "mean"),
            n_responses=("stance_num", "size"),
        )
        .reset_index()
    )

    baseline_rows = []
    for row in issue_position.itertuples(index=False):
        comparison = issue_position[
            (issue_position["model"] == row.model)
            & (issue_position["lang"] == row.lang)
            & (issue_position["issue_id"] != row.issue_id)
        ]
        baseline_rows.append(
            {
                "issue_id": row.issue_id,
                "model": row.model,
                "lang": row.lang,
                "issue_stance_raw": row.issue_stance_raw,
                "issue_relative_to_models": row.issue_relative,
                "loo_other_B_baseline": comparison["issue_relative"].mean(),
                "baseline_adjusted_score": (
                    row.issue_relative - comparison["issue_relative"].mean()
                ),
                "n_responses": row.n_responses,
                "n_baseline_issues": comparison["issue_id"].nunique(),
            }
        )
    return pd.DataFrame(baseline_rows).sort_values(["issue_id", "lang", "model"])


def rq4b_cluster_bootstrap(data: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap adjusted scores over the 100 matched prompt templates."""
    contested = data[
        (data["family"] == "B")
        & ~data["is_refusal"]
        & data["stance_num"].notna()
    ].copy()
    block = ["issue_id", "template_id", "lang", "rep"]
    contested["block_mean"] = contested.groupby(block, observed=True)[
        "stance_num"
    ].transform("mean")
    contested["within_block_relative"] = (
        contested["stance_num"] - contested["block_mean"]
    )
    contested["cluster"] = contested["template_id"].astype(str)
    cluster_cells = (
        contested.groupby(
            ["cluster", "issue_id", "model", "lang"], observed=True
        )["within_block_relative"]
        .mean()
        .reset_index()
    )
    wide = cluster_cells.pivot(
        index="cluster",
        columns=["issue_id", "model", "lang"],
        values="within_block_relative",
    ).sort_index(axis=1)
    values = wide.to_numpy()
    available = ~np.isnan(values)
    filled = np.nan_to_num(values)
    n_clusters = len(wide)
    rng = np.random.default_rng(RNG_SEED + 1)
    weights = rng.multinomial(
        n_clusters,
        np.full(n_clusters, 1 / n_clusters),
        size=N_BOOTSTRAPS,
    )
    numerator = weights @ filled
    denominator = weights @ available.astype(float)
    boot_positions = numerator / denominator

    columns = list(wide.columns)
    lookup = {column: index for index, column in enumerate(columns)}
    rows = []
    for column, column_index in lookup.items():
        issue, model, language = column
        baseline_indices = [
            other_index
            for (other_issue, other_model, other_language), other_index in lookup.items()
            if other_issue != issue
            and other_model == model
            and other_language == language
        ]
        draws = boot_positions[:, column_index] - np.nanmean(
            boot_positions[:, baseline_indices], axis=1
        )
        rows.append(
            {
                "issue_id": issue,
                "model": model,
                "lang": language,
                "bootstrap_ci_low": np.quantile(draws, 0.025),
                "bootstrap_ci_high": np.quantile(draws, 0.975),
                "bootstrap_probability_positive": (draws > 0).mean(),
                "n_bootstraps": N_BOOTSTRAPS,
                "n_template_clusters": n_clusters,
            }
        )
    return pd.DataFrame(rows)


def rq4b_correspondence(
    adjusted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge dated evidence and summarize directional and ranking correspondence."""
    evidence = pd.read_csv(EVIDENCE_PATH, dtype=str)
    scored = evidence.merge(
        adjusted,
        left_on=["topic_id", "model_key"],
        right_on=["issue_id", "model"],
        how="left",
        validate="one_to_many",
    )
    expected_sign = scored["position"].map({"pro": -1, "con": 1})
    observed_sign = np.sign(scored["baseline_adjusted_score"])
    scored["matches_direction"] = np.where(
        expected_sign.notna(),
        np.where(expected_sign == observed_sign, "yes", "no"),
        "not_directional",
    )

    directional_rows = []
    for rule, eligibility_column in [
        ("release_rule", "eligible_by_release"),
        ("reported_cutoff_rule", "eligible_by_reported_cutoff"),
    ]:
        eligible = scored[
            (scored[eligibility_column] == "yes")
            & scored["directional_test_use"].isin(["yes", "release_rule_only"])
            & scored["position"].isin(["pro", "con"])
        ]
        for language, group in eligible.groupby("lang", observed=True):
            directional_rows.append(
                {
                    "temporal_rule": rule,
                    "lang": language,
                    "n_model_issue_cells": len(group),
                    "n_matches": (group["matches_direction"] == "yes").sum(),
                    "match_rate": (group["matches_direction"] == "yes").mean(),
                    "mean_signed_correspondence": (
                        expected_sign.loc[group.index]
                        * group["baseline_adjusted_score"]
                    ).mean(),
                }
            )
    directional_summary = pd.DataFrame(directional_rows)

    position_order = {"pro": -1, "mixed": 0, "con": 1}
    pair_rows = []
    for rule, eligibility_column in [
        ("release_rule", "eligible_by_release"),
        ("reported_cutoff_rule", "eligible_by_reported_cutoff"),
    ]:
        eligible = scored[
            (scored[eligibility_column] == "yes")
            & scored["ranking_test_use"].isin(["yes", "release_rule_only"])
            & scored["position"].isin(position_order)
        ].copy()
        eligible["developer_order"] = eligible["position"].map(position_order)
        for (issue, language), group in eligible.groupby(
            ["topic_id", "lang"], observed=True
        ):
            records = list(group.itertuples(index=False))
            for left_index, left in enumerate(records):
                for right in records[left_index + 1 :]:
                    if left.developer_order == right.developer_order:
                        continue
                    developer_difference = left.developer_order - right.developer_order
                    model_difference = (
                        left.baseline_adjusted_score - right.baseline_adjusted_score
                    )
                    pair_rows.append(
                        {
                            "temporal_rule": rule,
                            "topic_id": issue,
                            "lang": language,
                            "model_1": left.model_key,
                            "developer_position_1": left.position,
                            "adjusted_score_1": left.baseline_adjusted_score,
                            "model_2": right.model_key,
                            "developer_position_2": right.position,
                            "adjusted_score_2": right.baseline_adjusted_score,
                            "concordant": (
                                "yes"
                                if np.sign(developer_difference)
                                == np.sign(model_difference)
                                else "no"
                            ),
                        }
                    )
    ranking_pairs = pd.DataFrame(pair_rows)
    if ranking_pairs.empty:
        ranking_summary = pd.DataFrame()
    else:
        ranking_summary = (
            ranking_pairs.groupby(["temporal_rule", "lang"], observed=True)
            .agg(
                n_comparable_pairs=("concordant", "size"),
                n_concordant=("concordant", lambda values: (values == "yes").sum()),
                concordance_rate=("concordant", lambda values: (values == "yes").mean()),
            )
            .reset_index()
        )
        rng = np.random.default_rng(RNG_SEED + 2)
        permutation_ps = []
        for summary_row in ranking_summary.itertuples(index=False):
            eligible = scored[
                (
                    scored[
                        "eligible_by_release"
                        if summary_row.temporal_rule == "release_rule"
                        else "eligible_by_reported_cutoff"
                    ]
                    == "yes"
                )
                & scored["ranking_test_use"].isin(["yes", "release_rule_only"])
                & scored["position"].isin(position_order)
                & (scored["lang"] == summary_row.lang)
            ].copy()
            eligible["developer_order"] = eligible["position"].map(position_order)
            groups = []
            for _, group in eligible.groupby("topic_id", observed=True):
                if group["developer_order"].nunique() < 2:
                    continue
                groups.append(
                    (
                        group["developer_order"].to_numpy(),
                        group["baseline_adjusted_score"].to_numpy(),
                    )
                )

            permuted_concordant = np.zeros(N_PERMUTATIONS, dtype=int)
            for developer_values, model_values in groups:
                for permutation_index in range(N_PERMUTATIONS):
                    permuted_developer = rng.permutation(developer_values)
                    for left_index in range(len(model_values)):
                        for right_index in range(left_index + 1, len(model_values)):
                            developer_difference = (
                                permuted_developer[left_index]
                                - permuted_developer[right_index]
                            )
                            if developer_difference == 0:
                                continue
                            model_difference = (
                                model_values[left_index] - model_values[right_index]
                            )
                            permuted_concordant[permutation_index] += int(
                                np.sign(developer_difference)
                                == np.sign(model_difference)
                            )
            p_value = (
                np.count_nonzero(
                    permuted_concordant >= summary_row.n_concordant
                )
                + 1
            ) / (N_PERMUTATIONS + 1)
            permutation_ps.append(p_value)
        ranking_summary["developer_label_permutation_p_greater"] = permutation_ps
    return scored, directional_summary, ranking_pairs, ranking_summary


def rq4_developer_government_divergence(
    adjusted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether adjusted stance moves toward the developer side of a
    documented developer--home-government contrast.

    The contrast file records only source-supported ordinal relations. Most are
    categorical (pro/mixed/con); B5 is an intensity contrast because Anthropic
    and the U.S. government are both broadly pro export controls, but Anthropic
    explicitly argued that government controls had not gone far enough.
    """
    contrasts = pd.read_csv(DIVERGENCE_PATH, dtype=str)
    scored = contrasts.merge(
        adjusted,
        left_on=["topic_id", "model_key"],
        right_on=["issue_id", "model"],
        how="left",
        validate="one_to_many",
    )
    expected_sign = scored["relative_prediction"].map(
        {"more_pro": -1, "more_con": 1}
    )
    observed_sign = np.sign(scored["baseline_adjusted_score"])
    scored["matches_developer_over_government_direction"] = np.where(
        expected_sign == observed_sign, "yes", "no"
    )
    scored["signed_developer_direction"] = (
        expected_sign * scored["baseline_adjusted_score"]
    )

    summaries = []
    for rule, eligibility_column in [
        ("release_rule", "eligible_by_release"),
        ("reported_cutoff_rule", "eligible_by_reported_cutoff"),
    ]:
        eligible = scored[scored[eligibility_column] == "yes"]
        for language, group in eligible.groupby("lang", observed=True):
            summaries.append(
                {
                    "temporal_rule": rule,
                    "lang": language,
                    "n_developer_government_contrasts": len(group),
                    "n_toward_developer": (
                        group["matches_developer_over_government_direction"] == "yes"
                    ).sum(),
                    "toward_developer_rate": (
                        group["matches_developer_over_government_direction"] == "yes"
                    ).mean(),
                    "mean_signed_developer_direction": group[
                        "signed_developer_direction"
                    ].mean(),
                }
            )
    return scored, pd.DataFrame(summaries)


def run_differentiation_and_stance() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(MASTER_PATH, low_memory=False)

    issue_dispersion, trend_tests = rq4a_differentiation(data)
    issue_dispersion.to_csv(OUT / "rq4a_issue_model_differentiation.csv", index=False)
    trend_tests.to_csv(OUT / "rq4a_differentiation_trend_tests.csv", index=False)

    adjusted = rq4b_baseline_adjustment(data)
    bootstrap = rq4b_cluster_bootstrap(data)
    adjusted = adjusted.merge(
        bootstrap,
        on=["issue_id", "model", "lang"],
        how="left",
        validate="one_to_one",
    )
    adjusted.to_csv(OUT / "rq4b_baseline_adjusted_model_stances.csv", index=False)
    scored, directional, ranking_pairs, ranking_summary = rq4b_correspondence(adjusted)
    scored.to_csv(OUT / "rq4b_evidence_scored.csv", index=False)
    directional.to_csv(OUT / "rq4b_directional_correspondence.csv", index=False)
    ranking_pairs.to_csv(OUT / "rq4b_ranking_pairs.csv", index=False)
    ranking_summary.to_csv(OUT / "rq4b_ranking_summary.csv", index=False)
    divergence_scored, divergence_summary = rq4_developer_government_divergence(
        adjusted
    )
    divergence_scored.to_csv(
        OUT / "rq4_developer_government_divergence_scored.csv", index=False
    )
    divergence_summary.to_csv(
        OUT / "rq4_developer_government_divergence_summary.csv", index=False
    )
    interest = scored[scored["evidence_tier"] == "inferred_interest"].copy()
    interest_summary = (
        interest.groupby(["topic_id", "lang"], observed=True)
        .agg(
            n_model_cells=("model_key", "size"),
            n_matches=("matches_direction", lambda values: (values == "yes").sum()),
            match_rate=("matches_direction", lambda values: (values == "yes").mean()),
        )
        .reset_index()
    )
    interest.to_csv(OUT / "rq4_exploratory_interest_scored.csv", index=False)
    interest_summary.to_csv(
        OUT / "rq4_exploratory_interest_summary.csv", index=False
    )

    print("RQ4a mean cross-model SD by developer stake:")
    print(
        issue_dispersion.groupby("developer_stake", observed=True)[
            "mean_cross_model_sd_across_languages"
        ]
        .agg(["mean", "count"])
        .reindex(STAKE_ORDER)
        .round(3)
        .to_string()
    )
    print("\nRQ4a trend tests:")
    print(trend_tests.round(4).to_string(index=False))
    print("\nRQ4b selected baseline-adjusted scores (+ = unusually con):")
    selected = adjusted[adjusted["issue_id"].isin(["B2", "B4", "B5", "B6", "B11"])]
    print(
        selected.pivot_table(
            index=["issue_id", "model"],
            columns="lang",
            values="baseline_adjusted_score",
        )
        .round(3)
        .to_string()
    )
    print("\nRQ4b directional correspondence:")
    print(directional.round(3).to_string(index=False))
    print("\nRQ4b pairwise ordering correspondence:")
    print(ranking_summary.round(3).to_string(index=False))
    print("\nExploratory commercial-interest correspondence:")
    print(interest_summary.round(3).to_string(index=False))
    print("\nDeveloper--government divergence correspondence:")
    print(divergence_summary.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Stage 2: figures
# ---------------------------------------------------------------------------


def fig_rq4a():
    """Cross-model dispersion vs developer stake -- the null."""
    d = pd.read_csv(TABLES / "rq4a_issue_model_differentiation.csv")
    trend = pd.read_csv(TABLES / "rq4a_differentiation_trend_tests.csv")
    row = trend[
        (trend["sample"] == "all_A_B")
        & (trend["outcome"] == "mean_cross_model_sd_across_languages")
    ].iloc[0]

    xpos = {s: i for i, s in enumerate(STAKE_ORDER_LABELS)}
    d = d[d["developer_stake"].isin(STAKE_ORDER_LABELS)].copy()
    d["x"] = d["developer_stake"].map(xpos)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    rng = np.random.default_rng(7)
    # per-stake mean marker
    means = d.groupby("x")["mean_cross_model_sd_across_languages"].mean()
    ax.plot(
        means.index,
        means.values,
        "-",
        color=INK_MUTED,
        lw=1.2,
        zorder=1,
        alpha=0.7,
    )
    for _, r in d.iterrows():
        jit = (rng.random() - 0.5) * 0.28
        col = "#b8b7b3" if r["family"] == "A" else INK
        ax.scatter(
            r["x"] + jit,
            r["mean_cross_model_sd_across_languages"],
            s=42,
            color=col,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    # label the B10 outlier
    b10 = d[d["issue_id"] == "B10"]
    if len(b10):
        r = b10.iloc[0]
        ax.annotate(
            "B10 (G7/OECD standard-setting)",
            (r["x"], r["mean_cross_model_sd_across_languages"]),
            xytext=(11, -9),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_MUTED,
        )
    ax.scatter([], [], s=42, color="#b8b7b3", label="A-series (non-contested)")
    ax.scatter([], [], s=42, color=INK, label="B-series (contested)")
    ax.set_xticks(range(len(STAKE_ORDER_LABELS)))
    ax.set_xticklabels(STAKE_ORDER_LABELS)
    ax.set_xlabel("Direct developer stake in the issue")
    ax.set_ylabel("Cross-model stance dispersion (SD, EN/ZH averaged)")
    ax.set_xlim(-0.5, len(STAKE_ORDER_LABELS) - 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.set_title(
        "Model disagreement does not rise with developer stake",
        loc="left",
    )
    ax.text(
        0.0,
        -0.16,
        f"Spearman $\\rho$ = {row['spearman_rho']:.2f}, "
        f"permutation $p$ = {row['permutation_p_two_sided']:.3f} (17 issues); "
        "the trend runs slightly negative.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=INK_MUTED,
    )
    save(fig, FIGS / "nb4c_rq4a_stake_dispersion")


def _signed_interval(row):
    """Re-sign the bootstrap interval onto the developer-direction scale.

    The stored interval brackets ``baseline_adjusted_score``; the plotted quantity
    is ``signed_developer_direction``, which is that score times +/-1 depending on
    which way the developer's position points.
    """
    raw = float(row["baseline_adjusted_score"])
    signed = float(row["signed_developer_direction"])
    factor = 1.0 if abs(signed - raw) < 1e-9 else -1.0
    low = float(row["bootstrap_ci_low"]) * factor
    high = float(row["bootstrap_ci_high"]) * factor
    return min(low, high), max(low, high)


def fig_rq4_devgov_matrix():
    """Evidence matrix for the six developer--government contrasts.

    Left: which issue, whose developer, and which way the two positions pointed.
    Right: whether the model moved toward or away from the developer in each
    language, with the baseline-adjusted size of the move.
    """
    d = pd.read_csv(TABLES / "rq4_developer_government_divergence_scored.csv")
    d = d[d["eligible_by_release"].eq("yes")].copy()

    order = list(DEVGOV_LABELS)
    by_case = {
        key: d[(d["topic_id"] == key[0]) & (d["developer"] == key[1])]
        for key in order
    }
    missing = [key for key, rows in by_case.items() if set(rows["lang"]) != {"en", "zh"}]
    if missing:
        raise ValueError(f"RQ4 developer--government matrix has incomplete cases: {missing}")

    n_rows = len(order)
    fig = plt.figure(figsize=(6.5, 2.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows - 0.30, -1.42)  # inverted: header above the first case
    ax.axis("off")

    columns = [
        (0.000, 0.310, "Issue", "left"),
        (0.310, 0.430, "Developer", "left"),
        (0.430, 0.555, "Developer\nposition", "left"),
        (0.555, 0.710, "U.S. government\nposition", "left"),
        (0.725, 0.860, "English", "center"),
        (0.868, 1.000, "Chinese", "center"),
    ]

    # the issue is a property of the group, not the row: print it once per group
    issue_groups = []
    for index, key in enumerate(order):
        issue = DEVGOV_LABELS[key][0]
        if issue_groups and issue_groups[-1][0] == issue:
            issue_groups[-1][2] = index
        else:
            issue_groups.append([issue, index, index])

    # headers sit on one baseline just above the rule, whether one line or two
    for x0, x1, label, align in columns:
        ax.text(
            x0 if align == "left" else (x0 + x1) / 2, -0.56, label,
            ha=align, va="bottom", fontsize=8.6, fontweight="bold", color=INK,
        )
    ax.plot([0, 1], [-0.50, -0.50], color=INK_MUTED, linewidth=0.8, clip_on=False)

    for issue, first, last in issue_groups:
        ax.text(
            0.000, (first + last) / 2, issue,
            ha="left", va="center", fontsize=9.2, color=INK,
        )
        if last + 1 < n_rows:
            ax.plot([0, 1], [last + 0.5, last + 0.5], color=GRID, linewidth=0.6)

    for index, key in enumerate(order):
        _, developer, dev_position, gov_position = DEVGOV_LABELS[key]
        for (x0, _, _, _), value in zip(columns[1:4], [developer, dev_position, gov_position]):
            ax.text(
                x0, index, value,
                ha="left", va="center", fontsize=9.0, color=CELL_INK,
            )

        rows = by_case[key].set_index("lang")
        for lang, (x0, x1, _, _) in zip(["en", "zh"], columns[4:]):
            row = rows.loc[lang]
            score = float(row["signed_developer_direction"])
            low, high = _signed_interval(row)
            significant = low > 0 or high < 0
            toward = score > 0
            ink = TOWARD_INK if toward else AWAY_INK
            ax.add_patch(
                plt.Rectangle(
                    (x0, index - 0.42), x1 - x0, 0.84,
                    facecolor=TOWARD_FILL if toward else AWAY_FILL,
                    edgecolor="none", zorder=0,
                )
            )
            ax.text(
                (x0 + x1) / 2, index - 0.20,
                "TOWARD" if toward else "AWAY",
                ha="center", va="center", fontsize=6.8, color=ink,
            )
            ax.text(
                (x0 + x1) / 2, index + 0.16,
                f"{score:+.3f}{'*' if significant else ''}",
                ha="center", va="center", fontsize=9.6, color=ink,
            )

    save(fig, FIGS / "nb4c_rq4_devgov_matrix")


def run_figures() -> None:
    apply_style()
    fig_rq4a()
    fig_rq4_devgov_matrix()
    print("wrote nb4c_rq4a_stake_dispersion and nb4c_rq4_devgov_matrix (pdf+png)")


# ---------------------------------------------------------------------------
# Stage 3: appendix table export
# ---------------------------------------------------------------------------


def tex(value: object) -> str:
    if pd.isna(value):
        return "---"
    text = str(value).strip()
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def score_cell(row: pd.Series, language: str, match_column: str) -> str:
    score = row[f"baseline_adjusted_score_{language}"]
    low = row[f"bootstrap_ci_low_{language}"]
    high = row[f"bootstrap_ci_high_{language}"]
    match = row[f"{match_column}_{language}"]
    return f"{score:+.3f} [{low:+.3f}, {high:+.3f}]; {tex(match)}"


def issue_stake_table() -> str:
    results = pd.read_csv(TABLES / "rq4a_issue_model_differentiation.csv")
    metadata = pd.read_csv(META)[["topic_id", "topic_neutral"]]
    frame = results.merge(metadata, left_on="issue_id", right_on="topic_id")
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"{tex(row.issue_id)} & {tex(row.developer_stake)} & "
            f"{tex(row.topic_neutral)} & "
            f"{row.mean_cross_model_sd_across_languages:.3f} \\\\"
        )
    return r"""
\subsection{Complete RQ4 Stance Results}\label{app:rq4complete}

\paragraph{Developer-stake coding.} Low denotes issues directed principally at states,
deployers, or governance venues, with only indirect implications for model developers.
Medium denotes meaningful but indirect constraints on a development input or public-facing
service. High denotes direct operational requirements affecting model development or
deployment. Very High denotes rules governing core model assets, training practices, or
general developer accountability. These ordered categories are considered judgments rather
than measured quantities; Table~\ref{tab:rq4astakes} lists every assignment so that the trend
test can be audited.

\begingroup\footnotesize
\setlength{\LTcapwidth}{\textwidth}
\begin{longtable}{@{}p{0.8cm}p{1.5cm}p{10.6cm}r@{}}
\caption{Complete issue-level input to RQ4a. Dispersion is the standard deviation of the six
model-specific stance means within each language, averaged across English and Chinese.}
\label{tab:rq4astakes}\\
\toprule
Issue & Stake & Policy issue & Dispersion \\
\midrule
\endfirsthead
\multicolumn{4}{l}{\small\itshape Table~\ref{tab:rq4astakes} continued}\\
\toprule
Issue & Stake & Policy issue & Dispersion \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\endgroup
"""


def directional_table() -> str:
    data = pd.read_csv(TABLES / "rq4b_evidence_scored.csv")
    data = data[
        data["eligible_by_release"].eq("yes")
        & data["evidence_tier"].eq("stated")
        & data["directional_test_use"].isin(["yes", "release_rule_only"])
        & data["position"].isin(["pro", "con"])
    ].copy()
    keep = [
        "topic_id", "developer", "model_key", "position",
        "eligible_by_reported_cutoff", "lang", "baseline_adjusted_score",
        "bootstrap_ci_low", "bootstrap_ci_high", "matches_direction",
    ]
    data = data[keep].drop_duplicates()
    wide = data.pivot(
        index=["topic_id", "developer", "model_key", "position", "eligible_by_reported_cutoff"],
        columns="lang",
        values=["baseline_adjusted_score", "bootstrap_ci_low", "bootstrap_ci_high", "matches_direction"],
    ).reset_index()
    wide.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    rows = []
    wide["issue_order"] = wide["topic_id"].map(ISSUE_ORDER)
    for _, row in wide.sort_values(["issue_order", "developer"]).iterrows():
        cutoff = {"yes": "Yes", "no": "No", "unknown": "Unknown"}.get(
            row["eligible_by_reported_cutoff"], row["eligible_by_reported_cutoff"]
        )
        rows.append(
            f"{tex(row['topic_id'])} & {tex(row['developer'])} ({MODEL[row['model_key']]}) & "
            f"{tex(row['position'])} & {tex(cutoff)} & "
            f"{score_cell(row, 'en', 'matches_direction')} & "
            f"{score_cell(row, 'zh', 'matches_direction')} \\\\"
        )
    return r"""
\paragraph{Eligible directional cells.} Table~\ref{tab:rq4bdirectional} reports all
nine stated developer--issue cells in the primary pre-release analysis. A negative adjusted
score is more pro than the model's usual relative stance and a positive score is more con.
``Match'' indicates movement in the developer-predicted direction, not statistical
significance.

\begingroup\scriptsize
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{2pt}
\begin{longtable}{@{}p{0.7cm}p{2.7cm}p{0.9cm}p{1.1cm}p{4.6cm}p{4.6cm}@{}}
\caption{Complete RQ4b directional results under the primary pre-release rule. Brackets give
95\% template-cluster bootstrap intervals; the cutoff column records inclusion in the stricter
reported-cutoff sensitivity.}\label{tab:rq4bdirectional}\\
\toprule
Issue & Developer (model) & Position & Cutoff & English; match & Chinese; match \\
\midrule
\endfirsthead
\multicolumn{6}{l}{\small\itshape Table~\ref{tab:rq4bdirectional} continued}\\
\toprule
Issue & Developer (model) & Position & Cutoff & English; match & Chinese; match \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\endgroup
"""


def ranking_table() -> str:
    pairs = pd.read_csv(TABLES / "rq4b_ranking_pairs.csv")
    pairs = pairs[pairs["temporal_rule"].eq("release_rule")]
    keys = ["topic_id", "model_1", "developer_position_1", "model_2", "developer_position_2"]
    wide = pairs.pivot(index=keys, columns="lang", values=["adjusted_score_1", "adjusted_score_2", "concordant"]).reset_index()
    wide.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    rows = []
    wide["issue_order"] = wide["topic_id"].map(ISSUE_ORDER)
    for _, row in wide.sort_values(["issue_order", "model_1", "model_2"]).iterrows():
        contrast = (
            f"{MODEL[row['model_1']]} ({tex(row['developer_position_1'])}) vs. "
            f"{MODEL[row['model_2']]} ({tex(row['developer_position_2'])})"
        )
        en = f"{row['adjusted_score_1_en']:+.3f} vs. {row['adjusted_score_2_en']:+.3f}; {tex(row['concordant_en'])}"
        zh = f"{row['adjusted_score_1_zh']:+.3f} vs. {row['adjusted_score_2_zh']:+.3f}; {tex(row['concordant_zh'])}"
        rows.append(f"{tex(row['topic_id'])} & {contrast} & {en} & {zh} \\\\")
    return r"""
\begin{table}[p]
\centering\footnotesize
\begin{tabularx}{\textwidth}{@{}lXll@{}}
\toprule
Issue & Developer-position contrast & English; concordant & Chinese; concordant \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\caption{All six distinguishable developer-position pairs under the pre-release rule. Scores
are listed in the same order as the models in the contrast; ``yes'' means the model ordering
matches the developer ordering.}
\label{tab:rq4brankingpairs}
\end{table}
"""


def divergence_table() -> str:
    data = pd.read_csv(TABLES / "rq4_developer_government_divergence_scored.csv")
    keep = [
        "topic_id", "developer", "model_key", "developer_position", "government_position",
        "relative_prediction", "contrast_type", "eligible_by_reported_cutoff", "lang",
        "baseline_adjusted_score", "bootstrap_ci_low", "bootstrap_ci_high",
        "matches_developer_over_government_direction",
    ]
    data = data[keep].drop_duplicates()
    wide = data.pivot(
        index=["topic_id", "developer", "model_key", "developer_position", "government_position", "relative_prediction", "contrast_type", "eligible_by_reported_cutoff"],
        columns="lang",
        values=["baseline_adjusted_score", "bootstrap_ci_low", "bootstrap_ci_high", "matches_developer_over_government_direction"],
    ).reset_index()
    wide.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    rows = []
    wide["issue_order"] = wide["topic_id"].map(ISSUE_ORDER)
    for _, row in wide.sort_values(["issue_order", "developer"]).iterrows():
        relation = "more pro" if row["relative_prediction"] == "more_pro" else "more con"
        if row["contrast_type"] == "intensity":
            relation += " (intensity)"
        rows.append(
            f"{tex(row['topic_id'])} & {tex(row['developer'])} ({MODEL[row['model_key']]}) & "
            f"{tex(row['developer_position'])}/{tex(row['government_position'])} & {relation} & "
            f"{score_cell(row, 'en', 'matches_developer_over_government_direction')} & "
            f"{score_cell(row, 'zh', 'matches_developer_over_government_direction')} \\\\"
        )
    return r"""
\paragraph{Developer--government contrasts.} We coded whether each temporally eligible
developer position was more pro or more con than the home-government position used in RQ1.
Five contrasts differ categorically; Anthropic on B5 adds one source-supported intensity
contrast because it argued that existing U.S. export controls had not gone far enough. The
test asks whether the adjusted model score moves toward the developer side of that contrast.
It does not treat zero as a measured government response.

Under the pre-release rule, three of six English cells and four of six Chinese cells moved
toward the developer side. Under the stricter reported-cutoff rule, the corresponding counts
were one of four and two of four. B4 supplies the clearest ordinal benchmark: the documented
order is Meta pro, U.S. government mixed, Anthropic con, so the predicted adjusted order is
Llama below zero and Claude above zero. English produced the reverse signs; in Chinese Claude
was on the predicted side but Llama was not. The contrasts are too few and too dependent for
an inferential aggregate test.

\begingroup\scriptsize
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{2pt}
\begin{longtable}{@{}p{0.6cm}p{2.5cm}p{1.3cm}p{1.6cm}p{4.5cm}p{4.5cm}@{}}
\caption{Complete developer--home-government divergence analysis. Developer/government gives
the two documented positions; ``match'' means that the baseline-adjusted model score moved
toward the developer side of their relative ordering.}\label{tab:rq4devgov}\\
\toprule
Issue & Developer (model) & Dev./gov. & Relation & English; match & Chinese; match \\
\midrule
\endfirsthead
\multicolumn{6}{l}{\small\itshape Table~\ref{tab:rq4devgov} continued}\\
\toprule
Issue & Developer (model) & Dev./gov. & Relation & English; match & Chinese; match \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\endgroup
"""


def commercial_table() -> str:
    data = pd.read_csv(TABLES / "rq4_exploratory_interest_scored.csv")
    keep = ["topic_id", "developer", "model_key", "position", "lang", "baseline_adjusted_score", "matches_direction"]
    data = data[keep].drop_duplicates()
    wide = data.pivot(index=["topic_id", "developer", "model_key", "position"], columns="lang", values=["baseline_adjusted_score", "matches_direction"]).reset_index()
    wide.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    rows = []
    wide["issue_order"] = wide["topic_id"].map(ISSUE_ORDER)
    for _, row in wide.sort_values(["issue_order", "developer"]).iterrows():
        rows.append(
            f"{tex(row['topic_id'])} & {tex(row['developer'])} ({MODEL[row['model_key']]}) & "
            f"{tex(row['position'])} & {row['baseline_adjusted_score_en']:+.3f}; {tex(row['matches_direction_en'])} & "
            f"{row['baseline_adjusted_score_zh']:+.3f}; {tex(row['matches_direction_zh'])} \\\\"
        )
    return r"""
\begin{table}[p]
\centering\footnotesize
\begin{tabularx}{\textwidth}{@{}llcll@{}}
\toprule
Issue & Developer (model) & Inferred position & English; match & Chinese; match \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\caption{Complete exploratory commercial-interest results. These predictions are inferred
from a developer's commercial circumstances and are not coded as public advocacy.}
\label{tab:rq4commercial}
\end{table}
"""


def run_export_appendix_tables() -> None:
    document = (
        "% Auto-generated by evaluation/scripts/06_analysis/rq4.py\n"
        "% Do not edit by hand; regenerate from the frozen RQ4 CSV outputs.\n"
        + issue_stake_table()
        + directional_table()
        + ranking_table()
        + divergence_table()
        + commercial_table()
    )
    APPENDIX_TEX_OUT.write_text(document)
    bad = sorted({character for character in document if ord(character) > 127})
    if bad:
        raise ValueError(f"Non-ASCII characters in generated LaTeX: {bad}")
    print(f"Wrote {APPENDIX_TEX_OUT}")


if __name__ == "__main__":
    run_differentiation_and_stance()
    run_figures()
    run_export_appendix_tables()
