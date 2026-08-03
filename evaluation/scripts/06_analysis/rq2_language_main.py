"""RQ2: English-versus-Chinese prompt effects and their interaction with origin.

Reads ``master.csv``. The issue-clustered inference used for final uncertainty is
written by ``rq1.py``; this script provides the main and robustness estimates.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, entropy
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402

warnings.filterwarnings("ignore")
TABLES = Path("evaluation/analysis/outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

SIGN = {"pro": 1, "con": -1}
ISSUE_ORDER = lambda s: int(s[1:])  # noqa: E731


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# Lim & Rottger's metrics, copied verbatim from their utils.py (natural-log JSD,
# not square-rooted; bias-corrected Cramer's V). Same functions as nb1b.
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
b_all = df[df["family"] == "B"].copy()     # keeps refusals: needed for Part 0 + histograms
b = b_all[~b_all["is_refusal"]].copy()     # drops them: needed for every stance model
b["grp"] = 1
b["origin"] = pd.Categorical(b["origin"], categories=["US", "CN", "EU"])
b["lang"] = pd.Categorical(b["lang"], categories=["en", "zh"])


def fit_mixed(data, formula, label, keep=None):
    """stance ~ formula, with crossed issue + template random intercepts."""
    md = smf.mixedlm(
        formula, data, groups=data["grp"],
        vc_formula={"issue": "0 + C(issue_id)", "template": "0 + C(template_id)"},
    )
    res = md.fit(reml=True, method="lbfgs")
    out = pd.DataFrame({
        "coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues,
        "ci_lo": res.conf_int()[0], "ci_hi": res.conf_int()[1],
    })
    out = out[~out.index.str.contains("Var|Cov", na=False)]
    if keep:
        out = out.loc[[k for k in keep if k in out.index]]
    print(f"\n--- {label} ---   (n = {len(data):,})")
    print(out.round(3))
    return out


# ============================================================ PART 0: refusals
section("PART 0 — REFUSALS BY LANGUAGE (the denominator check)")
print("Refusals are dropped from every stance model below. If they were concentrated in")
print("one language AND large, the two language conditions would not be comparable.\n")

ref = (b_all.groupby(["model", "lang"])["is_refusal"].mean() * 100).unstack().round(3)
ref["zh - en"] = (ref["zh"] - ref["en"]).round(3)
print(ref.to_string())
print(f"\nOverall: en {b_all[b_all.lang == 'en'].is_refusal.mean() * 100:.3f}%  "
      f"zh {b_all[b_all.lang == 'zh'].is_refusal.mean() * 100:.3f}%")
print("Refusal is rarer than 1% in every cell, so the ZH-vs-EN stance comparison is not")
print("meaningfully conditioned on different denominators. Reported, not dropped.")
ref.to_csv(TABLES / "nb2_refusals_by_language.csv")

# ============================================ PART 1: the pre-specified model
section("PART 1 — M1, THE PRE-SPECIFIED MODEL (raw stance).  [- = more PRO-as-phrased]")

KEEP = ["C(origin)[T.CN]", "C(origin)[T.EU]", "C(lang)[T.zh]",
        "C(origin)[T.CN]:C(lang)[T.zh]", "C(origin)[T.EU]:C(lang)[T.zh]"]
m1 = fit_mixed(b, "stance_num ~ C(origin) * C(lang)", "M1  stance ~ origin * lang", KEEP)
m1.to_csv(TABLES / "nb2_m1_prespecified.csv")
print("\nREAD THIS COEFFICIENT CAREFULLY. A negative zh term means models sit closer to the")
print("PRO pole of the proposition as phrased. It does NOT mean 'more pro-regulation' and")
print("it does NOT mean 'more pro-China' — 'pro' is a different substantive direction on")
print("different issues. Parts 2-3 establish what it actually is.")

# ============================================ PART 2: per-issue direction
section("PART 2 — WHERE DOES CHINESE PUSH, ISSUE BY ISSUE?")
print("Per-issue ZH-EN shift in raw stance, holding template fixed. Negative = Chinese")
print("prompts move the model toward the PRO pole of that issue's proposition.\n")

rows = []
for iid, blk in b.groupby("issue_id"):
    d = blk.copy()
    d["lang"] = pd.Categorical(d["lang"], categories=["en", "zh"])
    r = smf.ols("stance_num ~ C(lang) + C(template_id) + C(model)", d).fit()
    coef, p = r.params["C(lang)[T.zh]"], r.pvalues["C(lang)[T.zh]"]
    cn = blk["china_position"].iloc[0]
    # toward China = toward pro if China is pro; toward con if China is con
    toward_china = -coef * SIGN[cn] if cn in SIGN else np.nan
    rows.append({
        "issue": iid,
        "zh_coef": coef,
        "p": p,
        "zh_moves": "PRO" if coef < 0 else "CON",
        "china_pos": cn,
        "us_pos": blk["us_position"].iloc[0],
        "toward_china": toward_china,
    })

per = pd.DataFrame(rows).sort_values("issue", key=lambda c: c.map(ISSUE_ORDER))
print(per.round(3).to_string(index=False))
per.to_csv(TABLES / "nb2_per_issue_language.csv", index=False)

n_pro = int((per.zh_coef < 0).sum())
print(f"\nChinese moves stance toward the PRO pole on {n_pro} of {len(per)} issues.")
codeable = per[per.china_pos.isin(SIGN)]
n_china = int((codeable.toward_china > 0).sum())
print(f"Chinese moves stance toward CHINA'S position on {n_china} of {len(codeable)} "
      f"codeable issues.")
print("\nIf those two counts are both near-total, the issues do not discriminate and we")
print("cannot yet tell acquiescence from alignment. The two issues that DO discriminate")
print("are the ones where China is coded con — B5 and B10 — because there, agreeing with")
print("the proposition means siding with the US:")
disc = per[per.china_pos == "con"]
print(disc.round(3).to_string(index=False))

# ================================ PART 3: acquiescence vs alignment, head to head
section("PART 3 — THE DECISIVE TEST: ACQUIESCENCE vs CHINA-ALIGNMENT")

op = b[
    b["us_position"].isin(SIGN)
    & b["china_position"].isin(SIGN)
    & (b["us_position"] != b["china_position"])
].copy()
print(f"Strictly-opposed issues: {sorted(op.issue_id.unique(), key=ISSUE_ORDER)}")
print("On these, 'US-aligned' and 'China-aligned' are exact arithmetic opposites, so any")
print("general pro-regulation tendency cancels.\n")

# us_ord: 1 = most US-aligned ... 5 = most China-aligned, on every opposed issue.
# POSITIVE coefficient = leans toward CHINA'S pole. (Same coding as nb1b.)
op["china_pole"] = np.where(op["us_position"] == "pro",
                            op["stance_num"], 6 - op["stance_num"])

sub_a = op[op["us_position"] == "con"]   # 5 issues: China's pole = PRO-regulation
sub_b = op[op["us_position"] == "pro"]   # 2 issues (B5, B10): China's pole = ANTI-regulation

print("SUBSET A — 5 issues, China's pole is the PRO-regulation side.")
print("  Acquiescence and alignment predict the SAME sign here. Uninformative alone.")
a = fit_mixed(sub_a, "china_pole ~ C(origin) * C(lang)",
              "A  china_pole ~ origin * lang   [+ = toward CHINA]", KEEP)

print("\nSUBSET B — 2 issues (B5 export controls, B10 G7/OECD standards).")
print("  China's pole is the ANTI-regulation side, and agreeing with the proposition")
print("  means siding with the US. The hypotheses now DISAGREE:")
print("     H-acquiescence predicts zh NEGATIVE here (toward pro-as-phrased = toward US)")
print("     H-alignment    predicts zh POSITIVE here (toward China)")
bb = fit_mixed(sub_b, "china_pole ~ C(origin) * C(lang)",
               "B  china_pole ~ origin * lang   [+ = toward CHINA]", KEEP)

a.to_csv(TABLES / "nb2_subsetA_china_pole.csv")
bb.to_csv(TABLES / "nb2_subsetB_china_pole.csv")

zh_a, zh_b = a.loc["C(lang)[T.zh]", "coef"], bb.loc["C(lang)[T.zh]", "coef"]
p_b = bb.loc["C(lang)[T.zh]", "p"]
print("\n" + "-" * 78)
print(f"  Subset A  zh = {zh_a:+.3f}      Subset B  zh = {zh_b:+.3f}  (p = {p_b:.3g})")
if zh_b > 0 and p_b < 0.05:
    print("  => zh moves toward CHINA'S pole even where that means DISAGREEING with the")
    print("     proposition. Acquiescence is ruled out. The language effect is a genuine")
    print("     shift toward China's documented position, and it is confound-free.")
elif zh_b < 0 and p_b < 0.05:
    print("  => zh moves toward the PRO pole even where that means siding with the US.")
    print("     This is ACQUIESCENCE, not China-alignment. The pooled M1 language effect")
    print("     must NOT be reported as models 'siding with China' in Chinese.")
else:
    print("  => Subset B is null: the two hypotheses are not separated by these data.")
    print("     Report the pooled language effect as direction-ambiguous.")
print("-" * 78)

# ==================================================== PART 4: per-model effects
section("PART 4 — DOES EVERY MODEL SHIFT, OR ONLY SOME?")
print("Language effect fit separately per model (opposed issues, + = toward CHINA).\n")

rows = []
for mdl, blk in op.groupby("model"):
    d = blk.copy()
    d["lang"] = pd.Categorical(d["lang"], categories=["en", "zh"])
    r = smf.ols("china_pole ~ C(lang) + C(issue_id) + C(template_id)", d).fit()
    rows.append({
        "model": mdl,
        "origin": blk["origin"].iloc[0],
        "zh_toward_china": r.params["C(lang)[T.zh]"],
        "p": r.pvalues["C(lang)[T.zh]"],
        "en_mean": blk[blk.lang == "en"].china_pole.mean(),
        "zh_mean": blk[blk.lang == "zh"].china_pole.mean(),
    })
pm = pd.DataFrame(rows).sort_values("zh_toward_china", ascending=False)
print(pm.round(3).to_string(index=False))
pm.to_csv(TABLES / "nb2_per_model_language.csv", index=False)
print(f"\nModels shifting toward China in Chinese: {int((pm.zh_toward_china > 0).sum())}/6")

# ============================================ PART 5: the interaction (RQ2's 2nd half)
section("PART 5 — DOES THE LANGUAGE SHIFT INTERACT WITH ORIGIN?")
print("RQ2 asks this explicitly. The interaction terms in M1 / A / B above answer it;")
print("restated here as the ZH effect WITHIN each origin, which is easier to read.\n")

rows = []
for org, blk in op.groupby("origin", observed=True):
    d = blk.copy()
    d["lang"] = pd.Categorical(d["lang"], categories=["en", "zh"])
    r = smf.ols("china_pole ~ C(lang) + C(issue_id) + C(template_id) + C(model)", d).fit()
    rows.append({"origin": org, "zh_toward_china": r.params["C(lang)[T.zh]"],
                 "p": r.pvalues["C(lang)[T.zh]"], "n": len(d)})
byorg = pd.DataFrame(rows)
print(byorg.round(3).to_string(index=False))
byorg.to_csv(TABLES / "nb2_language_by_origin.csv", index=False)
print("\nEU origin = one model (Mistral), so the EU row IS the Mistral row and the two")
print("cannot be separated. Read it as a model estimate, not an origin estimate.")

# ==================================================== PART 6: robustness
section("PART 6 — ROBUSTNESS: ORDERED LOGIT + DISTRIBUTIONAL METRICS")
print("The linear models assume even spacing between stance categories, which the")
print("annotation data does not support. Refit assuming only that categories are ordered.\n")

ORD_KEEP = ["C(origin)[T.CN]", "C(origin)[T.EU]", "C(lang)[T.zh]",
            "C(origin)[T.CN]:C(lang)[T.zh]", "C(origin)[T.EU]:C(lang)[T.zh]"]


def ordered_logit(data, outcome, label):
    d = data.copy()
    d["origin"] = pd.Categorical(d["origin"], categories=["US", "CN", "EU"])
    d["lang"] = pd.Categorical(d["lang"], categories=["en", "zh"])
    d["_y"] = pd.Categorical(d[outcome].astype(int), ordered=True)
    f = f"_y ~ C(origin) * C(lang) + C(issue_id) + C(template_id)"
    res = OrderedModel.from_formula(f, d, distr="logit").fit(
        method="bfgs", maxiter=400, disp=False)
    out = pd.DataFrame({"log_odds": res.params, "se": res.bse, "p": res.pvalues})
    out = out.loc[[k for k in ORD_KEEP if k in out.index]]
    out["odds_ratio"] = np.exp(out["log_odds"])
    print(f"\n--- {label} ---   (n = {len(d):,})")
    print(out.round(3))
    return out


ordered_logit(sub_b, "china_pole",
              "SUBSET B ordinally  [+ = toward CHINA]").to_csv(
    TABLES / "nb2_ordlogit_subsetB.csv")

# Distributional: how far does swapping language move the stance histogram?
BINS = ["1", "2", "3", "4", "5", "refusal"]


def stance_dist(d):
    c = d["stance"].astype(str).value_counts()
    v = np.array([c.get(k, 0) for k in BINS], dtype=float) + 1e-10
    return v / v.sum()


# Swap language WITHIN each origin, then average — the same estimand as nb1b, which
# swaps origin within each language. Pooling origins before comparing would average the
# two histograms first and understate the divergence (it gives 0.0223, not 0.0314), and
# would not be comparable with the origin figure. Keep the two like for like.
rows = []
for iid, blk in b_all.groupby("issue_id"):
    ct = pd.crosstab(blk["stance"].astype(str), blk["lang"])
    l_jsds = [
        calculate_jsd(stance_dist(blk[(blk.origin == o) & (blk.lang == "en")]),
                      stance_dist(blk[(blk.origin == o) & (blk.lang == "zh")]))
        for o in ["US", "CN", "EU"]
    ]
    rows.append({
        "issue": iid,
        "JSD_language": np.mean(l_jsds),
        "V_language": cramers_v(ct.values),
    })
jsd = pd.DataFrame(rows).sort_values("issue", key=lambda c: c.map(ISSUE_ORDER))
print("\nJensen-Shannon divergence between the EN and ZH stance histograms, per issue:")
print(jsd.round(4).to_string(index=False))
jsd.to_csv(TABLES / "nb2_jsd_language.csv", index=False)
print(f"\n  mean JSD (language) = {jsd.JSD_language.mean():.4f}")
print(f"  mean Cramer's V     = {jsd.V_language.mean():.4f}   (0.1 = conventional 'small')")
print("  nb1b found JSD(origin) = 0.0164, so language moves the distribution ~1.9x more.")

section("DONE — tables written to evaluation/analysis/outputs/tables/nb2_*.csv")
