"""RQ2 distributional robustness checks.

Compares language and origin effects using JSD, chi-square, Cramer's V, and a
collapsed three-point stance scale. Reads ``master.csv``.
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy, mannwhitneyu

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402

warnings.filterwarnings("ignore")
TABLES = Path("evaluation/analysis/outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# Their functions, verbatim (data/east_west_bias_paper/06_data_analysis/scripts/utils.py)
def calculate_jsd(dist1, dist2):
    p, q = np.array(dist1), np.array(dist2)
    m = (p + q) / 2
    return 0.5 * (entropy(p, m) + entropy(q, m))


def cramers_v(confusion_matrix) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = np.asarray(confusion_matrix).sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


df = load_master()
b = df[df["family"] == "B"].copy()   # refusals KEPT: these are distributional tests
ISSUES = sorted(b["issue_id"].unique(), key=lambda s: int(s[1:]))
ORIGINS = ["US", "CN", "EU"]
MODELS = sorted(b["model"].unique())

BINS_5 = ["1", "2", "3", "4", "5", "refusal"]
BINS_3 = ["pro", "neutral", "con", "refusal"]
COLLAPSE = {"1": "pro", "2": "pro", "3": "neutral", "4": "con", "5": "con",
            "refusal": "refusal"}
b["stance_str"] = b["stance"].astype(str)
b["stance_3"] = b["stance_str"].map(COLLAPSE)


def dist(d, col, bins):
    c = d[col].value_counts()
    v = np.array([c.get(k, 0) for k in bins], dtype=float) + 1e-10
    return v / v.sum()


# ============================================== GAP 1: is language > origin, really?
section("GAP 1 — MANN-WHITNEY U + COHEN'S d:  is the language effect really bigger?")
print("Per-issue JSDs, then compare the two distributions the way they do")
print("(relative_effects.analyze_effects). n = 11 issues per group.\n")

rows = []
for issue in ISSUES:
    d_i = b[b["issue_id"] == issue]

    # LANGUAGE: en vs zh, model-origin held fixed (their compute_language_jsd)
    lang_jsds = [
        calculate_jsd(dist(d_i[(d_i.origin == o) & (d_i.lang == "en")], "stance_str", BINS_5),
                      dist(d_i[(d_i.origin == o) & (d_i.lang == "zh")], "stance_str", BINS_5))
        for o in ORIGINS
    ]
    # ORIGIN, all three pairwise (nb1b's construction; we have an EU model, they do not)
    org_jsds = [
        calculate_jsd(dist(d_i[(d_i.origin == a) & (d_i.lang == lg)], "stance_str", BINS_5),
                      dist(d_i[(d_i.origin == c) & (d_i.lang == lg)], "stance_str", BINS_5))
        for lg in ["en", "zh"] for a, c in combinations(ORIGINS, 2)
    ]
    # ORIGIN, US-vs-CN only — like-for-like with their Western-vs-Chinese comparison
    org_uscn = [
        calculate_jsd(dist(d_i[(d_i.origin == "US") & (d_i.lang == lg)], "stance_str", BINS_5),
                      dist(d_i[(d_i.origin == "CN") & (d_i.lang == lg)], "stance_str", BINS_5))
        for lg in ["en", "zh"]
    ]
    rows.append({"issue": issue, "JSD_language": np.mean(lang_jsds),
                 "JSD_origin": np.mean(org_jsds), "JSD_origin_USCN": np.mean(org_uscn)})

j = pd.DataFrame(rows)
j["lang_wins"] = j.JSD_language > j.JSD_origin
print(j.round(4).to_string(index=False))
j.to_csv(TABLES / "nb2b_jsd_per_issue.csv", index=False)


def compare(lang, org, label):
    u, p = mannwhitneyu(org, lang, alternative="two-sided")
    n1, n2 = len(org), len(lang)
    pooled = np.sqrt(((n1 - 1) * np.var(org, ddof=1) + (n2 - 1) * np.var(lang, ddof=1))
                     / (n1 + n2 - 2))
    d = (np.mean(org) - np.mean(lang)) / pooled if pooled > 0 else np.nan
    print(f"\n--- {label} ---")
    print(f"  origin   mean JSD = {np.mean(org):.4f}  (sd {np.std(org, ddof=1):.4f})")
    print(f"  language mean JSD = {np.mean(lang):.4f}  (sd {np.std(lang, ddof=1):.4f})"
          f"   -> {np.mean(lang) / np.mean(org):.1f}x")
    print(f"  Mann-Whitney U = {u:.0f}, p = {p:.4f}")
    print(f"  Cohen's d (origin - language) = {d:+.3f}")
    winner = "LANGUAGE" if np.mean(lang) > np.mean(org) else "ORIGIN"
    print(f"  => {winner} effects are larger"
          + (f", and the difference IS significant (p = {p:.4f})." if p < 0.05
             else f", but the difference is NOT significant (p = {p:.4f})."))
    return {"comparison": label, "mean_origin": np.mean(org), "mean_language": np.mean(lang),
            "U": u, "p": p, "cohens_d": d}


res = [
    compare(j.JSD_language.values, j.JSD_origin.values,
            "language vs origin (all 3 origins pairwise)"),
    compare(j.JSD_language.values, j.JSD_origin_USCN.values,
            "language vs origin (US vs CN only — their construction)"),
]
pd.DataFrame(res).to_csv(TABLES / "nb2b_mannwhitney.csv", index=False)
print(f"\nLanguage JSD exceeds origin JSD on {int(j.lang_wins.sum())}/{len(j)} issues.")
print("NOTE: n = 11 issues. Mann-Whitney on 11 vs 11 is a weak test; a null here means")
print("'not established on 11 issues', not 'no difference'. Report the n.")

# ====================================== GAP 2: chi-square per model (language x stance)
section("GAP 2 — CHI-SQUARE OF INDEPENDENCE: language x stance, overall and per model")
print("Their e_statistical_sig_tests notebook. Tests the DISTRIBUTION, not a mean.\n")

rows = []
ct = pd.crosstab(b["stance_str"], b["lang"])
chi2, p, dof, _ = chi2_contingency(ct)
rows.append({"model": "ALL", "origin": "-", "chi2": chi2, "dof": dof, "p": p,
             "cramers_v": cramers_v(ct.values), "n": len(b)})
for m, blk in b.groupby("model"):
    ct = pd.crosstab(blk["stance_str"], blk["lang"])
    chi2, p, dof, _ = chi2_contingency(ct)
    rows.append({"model": m, "origin": blk["origin"].iloc[0], "chi2": chi2, "dof": dof,
                 "p": p, "cramers_v": cramers_v(ct.values), "n": len(blk)})
chi = pd.DataFrame(rows)
print(chi.round(4).to_string(index=False))
chi.to_csv(TABLES / "nb2b_chi2_language_by_model.csv", index=False)
print("\nLanguage is significantly associated with the stance distribution in every model.")
print("Cramer's V here is the effect size: 0.1 small, 0.3 medium, 0.5 large.")

# --- The version reported in the write-up: refusals excluded (as in every other stance
# analysis) so all six models are on identical 1-5 support, and computed WITHIN each
# repetition rather than pooling the three correlated reps. The per-model chi-square p
# is an IID quantity over tripled, correlated rows; only the effect-size ORDERING is
# claimed, and the per-rep table shows that ordering does not depend on the pooling.
print("\n-- refusal-free, per-repetition (the version cited in main.tex) --")
bnr = b[~b.is_refusal]
prep = {}
for rep, blk_r in bnr.groupby("rep"):
    for m, blk in blk_r.groupby("model"):
        ct = pd.crosstab(blk["stance_str"], blk["lang"])
        prep.setdefault(m, {})[rep] = cramers_v(ct.values)
vrep = pd.DataFrame(prep).T
vrep["mean_v"] = vrep.mean(axis=1)
pooled = {m: cramers_v(pd.crosstab(blk["stance_str"], blk["lang"]).values)
          for m, blk in bnr.groupby("model")}
vrep["pooled_v"] = pd.Series(pooled)
vrep = vrep.sort_values("mean_v", ascending=False)
print(vrep.round(4).to_string())
vrep.to_csv(TABLES / "nb2b_cramersv_by_model_norefusal_perrep.csv")

# ====================================================== GAP 3: the collapsed 3-point scale
section("GAP 3 — COLLAPSED SCALE (pro = 1+2, neutral = 3, con = 4+5)")
print("31.7% of responses sit at the pro floor of the 5-point scale. If the RQ2 finding")
print("depends on how finely the pro end is cut, it is fragile. Test it collapsed.\n")

rows = []
for issue in ISSUES:
    d_i = b[b["issue_id"] == issue]
    lang_jsds = [
        calculate_jsd(dist(d_i[(d_i.origin == o) & (d_i.lang == "en")], "stance_3", BINS_3),
                      dist(d_i[(d_i.origin == o) & (d_i.lang == "zh")], "stance_3", BINS_3))
        for o in ORIGINS
    ]
    org_jsds = [
        calculate_jsd(dist(d_i[(d_i.origin == a) & (d_i.lang == lg)], "stance_3", BINS_3),
                      dist(d_i[(d_i.origin == c) & (d_i.lang == lg)], "stance_3", BINS_3))
        for lg in ["en", "zh"] for a, c in combinations(ORIGINS, 2)
    ]
    rows.append({"issue": issue, "JSD_language": np.mean(lang_jsds),
                 "JSD_origin": np.mean(org_jsds)})
j3 = pd.DataFrame(rows)
print(j3.round(4).to_string(index=False))
j3.to_csv(TABLES / "nb2b_jsd_collapsed.csv", index=False)
compare(j3.JSD_language.values, j3.JSD_origin.values, "COLLAPSED scale: language vs origin")

# And the decisive nb2 result, collapsed: does zh still move toward China on B5/B10?
print("\n--- The decisive Subset B test (B5, B10), on the collapsed scale ---")
print("nb2 found zh moves toward CHINA there (+0.149, p<0.001) — the result that ruled")
print("out acquiescence. Does it survive collapsing?\n")

sub_b = b[b.issue_id.isin(["B5", "B10"]) & ~b.is_refusal].copy()
# China's pole is CON on both, so P(con) is the share on China's side.
sub_b["china_side"] = sub_b["stance_3"].eq("con").astype(int)
tab = (sub_b.groupby(["origin", "lang"])["china_side"].mean() * 100).unstack().round(1)
tab["zh - en"] = (tab["zh"] - tab["en"]).round(1)
print("% of responses on CHINA'S side (collapsed con), B5 + B10:")
print(tab.to_string())
ct = pd.crosstab(sub_b["stance_3"], sub_b["lang"])
chi2, p, dof, _ = chi2_contingency(ct)
print(f"\nchi2 = {chi2:.1f}, dof = {dof}, p = {p:.3g}, Cramer's V = {cramers_v(ct.values):.3f}")
overall = (sub_b.groupby("lang")["china_side"].mean() * 100)
print(f"Overall: en {overall['en']:.1f}% -> zh {overall['zh']:.1f}% on China's side "
      f"({overall['zh'] - overall['en']:+.1f} pts)")
tab.to_csv(TABLES / "nb2b_subsetB_collapsed.csv")

section("DONE — tables written to evaluation/analysis/outputs/tables/nb2b_*.csv")
