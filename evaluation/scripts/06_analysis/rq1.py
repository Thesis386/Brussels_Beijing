"""RQ1: model-origin differences and alignment with home-jurisdiction positions.

Reads ``master.csv``. Includes main estimates, ordinal/distributional robustness,
variance decomposition, model-level alignment, and issue-clustered inference.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from patsy import dmatrix
from scipy.stats import binomtest, chi2_contingency, entropy, ttest_1samp
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402
from viz_style import (  # noqa: E402
    INK, INK_MUTED, MODEL_ORDER, ORIGIN_COLOR, SEQUENTIAL, SLOT, apply_style, save,
)

warnings.filterwarnings("ignore")
TABLES = Path("evaluation/analysis/outputs/tables")
FIG = Path("evaluation/analysis/outputs/figures")
TABLES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

SIGN = {"pro": 1, "con": -1}  # 'mixed' and 'unclear' are excluded, as pre-specified
JURIS = {"US": "us_position", "CN": "china_position", "EU": "eu_position"}

# origin/language contrasts kept from the origin*language design matrix by the
# ordered-logit robustness check and the clustered-inference re-estimates
KEEP = ["C(origin)[T.CN]", "C(origin)[T.EU]", "C(lang)[T.zh]",
        "C(origin)[T.CN]:C(lang)[T.zh]", "C(origin)[T.EU]:C(lang)[T.zh]"]

BINS = ["1", "2", "3", "4", "5", "refusal"]


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ---- Lim & Rottger's metrics, copied verbatim from
# ---- data/east_west_bias_paper/06_data_analysis/scripts/utils.py
# ---- (natural-log JSD, NOT square-rooted; bias-corrected Cramer's V)
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


def stance_dist(d):
    """Normalised stance histogram over the 6 bins, exactly as Lim & Rottger build it."""
    c = d["stance"].astype(str).value_counts()
    v = np.array([c.get(x, 0) for x in BINS], dtype=float)
    return v / v.sum() if v.sum() else np.ones(len(BINS)) / len(BINS)


def signed_alignment(d: pd.DataFrame, col: str) -> pd.Series:
    """Signed alignment score: +1 = fully aligned with `col`'s coded position,
    -1 = fully opposed, on the scale [-1, +1]. `d` must already be filtered to rows
    where `col` is codeable (isin(SIGN)). Shared by the RQ1b alignment table (nb1)
    and the per-model alignment table (nb1d), which computed this identically."""
    return d["pro_score"] * d[col].map(SIGN) / 2


def strictly_opposed(d: pd.DataFrame) -> pd.DataFrame:
    """The subset of rows on issues where the US and China take STRICTLY OPPOSITE
    documented positions (7 of the 11 core issues). Reused by nb1 (RQ1c), nb1b
    (ordered-logit O2-O4), nb1d (CN_opp7 / CN_b5b10) and nb2c (clustered PART 3) --
    each attaches its own signed pole column on top (us_pole / us_ord / china_pole),
    since each wants a different sign convention."""
    return d[
        (d["us_position"].isin(SIGN))
        & (d["china_position"].isin(SIGN))
        & (d["us_position"] != d["china_position"])
    ].copy()


def fit_mixed(data, formula, label):
    """Fit stance ~ formula with crossed issue + template random intercepts."""
    md = smf.mixedlm(
        formula,
        data,
        groups=data["grp"],
        vc_formula={"issue": "0 + C(issue_id)", "template": "0 + C(template_id)"},
    )
    res = md.fit(reml=True, method="lbfgs")
    out = pd.DataFrame({
        "coef": res.params,
        "se": res.bse,
        "z": res.tvalues,
        "p": res.pvalues,
        "ci_lo": res.conf_int()[0],
        "ci_hi": res.conf_int()[1],
    })
    out = out[~out.index.str.contains("Var|Cov", na=False)].round(3)
    print(f"\n--- {label} ---")
    print(out)
    print(f"    random-effect variances: "
          f"issue={res.vcomp[0]:.3f}  template={res.vcomp[1]:.3f}  residual={res.scale:.3f}")
    return res, out


def ordered_logit(data, outcome, label, absorb="C(issue_id) + C(template_id)"):
    """Proportional-odds model. Positive coef = higher outcome category."""
    d = data.copy()
    d["origin"] = pd.Categorical(d["origin"], categories=["US", "CN", "EU"])
    d["lang"] = pd.Categorical(d["lang"], categories=["en", "zh"])
    d["_y"] = pd.Categorical(d[outcome].astype(int), ordered=True)

    f = f"_y ~ C(origin) * C(lang) + {absorb}"
    res = OrderedModel.from_formula(f, d, distr="logit").fit(method="bfgs", maxiter=400, disp=False)

    out = pd.DataFrame({"log_odds": res.params, "se": res.bse, "p": res.pvalues})
    out = out.loc[[k for k in KEEP if k in out.index]]
    out["odds_ratio"] = np.exp(out["log_odds"])
    print(f"\n--- {label} ---   (n = {len(d):,})")
    print(out.round(3))
    return out


def cluster_ols(formula, data, groups, label, keep=None):
    """OLS with CRV1 cluster-robust SEs and t inference on G-1 df."""
    res = smf.ols(formula, data).fit(
        cov_type="cluster", cov_kwds={"groups": data[groups]}, use_t=True)
    out = pd.DataFrame({
        "coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues,
        "ci_lo": res.conf_int()[0], "ci_hi": res.conf_int()[1],
    })
    if keep:
        out = out.loc[[k for k in keep if k in out.index]]
    G = data[groups].nunique()
    print(f"\n--- {label} ---   (n = {len(data):,}; {G} clusters on '{groups}'; t on {G - 1} df)")
    print(out.round(4))
    return out, res


def wild_cluster_p(X, y, cl, k, B=999, seed=0):
    """Null-imposed wild cluster bootstrap (Rademacher) p for coefficient k.

    X: (n,p) design, y: (n,) outcome, cl: (n,) integer cluster codes 0..G-1.
    Returns (t_obs, p). CRV1 small-sample factor applied throughout (cancels in
    the bootstrap comparison but keeps t_obs comparable to statsmodels).
    """
    order = np.argsort(cl, kind="stable")
    X, y, cl = X[order], y[order], cl[order]
    starts = np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]])
    n, p = X.shape
    G = len(starts)
    XtX_inv = np.linalg.inv(X.T @ X)
    a_k = XtX_inv[:, k]
    cfac = G / (G - 1) * (n - 1) / (n - p)

    def tstat(y_):
        beta = XtX_inv @ (X.T @ y_)
        u = y_ - X @ beta
        g = np.add.reduceat(X * u[:, None], starts, axis=0)  # (G, p)
        se = np.sqrt(cfac * ((g @ a_k) ** 2).sum())
        return beta[k] / se

    t_obs = tstat(y)
    Xr = np.delete(X, k, axis=1)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fit_r, u_r = Xr @ br, y - Xr @ br
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(B):
        s = rng.choice([-1.0, 1.0], size=G)
        if abs(tstat(fit_r + u_r * s[cl])) >= abs(t_obs):
            hits += 1
    return t_obs, (hits + 1) / (B + 1)


def run_origin_main():
    """nb1 -- RQ1a/RQ1b/RQ1c. Formerly rq1_origin_main.py.

    RQ1a: does stance vary by origin at all? Mixed-effects model on the B-series,
    origin (and language) as fixed effects, issue and template as crossed random
    effects -- legitimate because the issue random intercept absorbs each issue's
    own pro/con direction, so origin coefficients are identified WITHIN issue.

    RQ1b: is the variation directionally consistent with home-jurisdiction policy?
    Alignment coding against the us/eu/china position codes: a modal-stance
    consistency RATE and a signed centred-stance AGGREGATE, both pre-specified.

    RQ1c: the clean origin test, on the 7 issues where US and China take strictly
    opposite positions -- the confound-free version of RQ1b.
    """
    df = load_master()

    b = df[(df["family"] == "B") & (~df["is_refusal"])].copy()
    b["truncated"] = b["finish_reason"] == "length"

    print(f"B-series non-refusal responses: {len(b):,}")
    print(f"  of which truncated (finish_reason == 'length'): {b['truncated'].sum():,}")

    # =============================================================== RQ1a: does stance vary?
    section("RQ1a — PER-ISSUE DESCRIPTIVES (no cross-issue aggregation)")

    for lang in ["en", "zh"]:
        piv = (
            b[b["lang"] == lang]
            .pivot_table(index="model", columns="issue_id", values="stance_num", aggfunc="mean")
            .round(2)
        )
        piv = piv[[f"B{i}" for i in range(1, 12)]]
        print(f"\nmean stance by model x issue  [{lang}]   (1 = pro, 3 = neutral, 5 = con)")
        print(piv)
        piv.to_csv(TABLES / f"nb1_stance_by_model_issue_{lang}.csv")

        print(f"  range across models (max - min) per issue: "
              f"{(piv.max() - piv.min()).round(2).to_dict()}")

    section("RQ1a — MIXED-EFFECTS MODEL")

    # Crossed random effects (issue and template are crossed, not nested). statsmodels
    # has no native crossed syntax, so use the standard trick: one dummy group covering
    # the whole sample, with issue and template entered as variance components.
    b["grp"] = 1
    b["origin"] = pd.Categorical(b["origin"], categories=["US", "CN", "EU"])  # US = reference
    b["lang"] = pd.Categorical(b["lang"], categories=["en", "zh"])            # en = reference

    # Model 1: the RQ1 claim. Origin x language, within issue and template.
    res1, tab1 = fit_mixed(
        b, "stance_num ~ C(origin) * C(lang)", "M1  stance ~ origin * language  (B-series, all responses)"
    )
    tab1.to_csv(TABLES / "nb1_mixed_origin_lang.csv")

    # Model 2: is the origin effect an origin effect, or one model dragging its bloc?
    b["model"] = pd.Categorical(
        b["model"], categories=["claude", "gpt4o", "llama", "deepseek", "qwen", "mistral"]
    )
    res2, tab2 = fit_mixed(b, "stance_num ~ C(model) * C(lang)", "M2  stance ~ model * language")
    tab2.to_csv(TABLES / "nb1_mixed_model_lang.csv")

    # Model 3: SENSITIVITY — drop truncated responses (the Claude artefact).
    res3, tab3 = fit_mixed(
        b[~b["truncated"]],
        "stance_num ~ C(origin) * C(lang)",
        "M3  SENSITIVITY: same as M1, truncated responses excluded",
    )
    tab3.to_csv(TABLES / "nb1_mixed_origin_lang_no_trunc.csv")

    print("\nM1 vs M3 — do the origin effects survive dropping truncated responses?")
    comp = pd.DataFrame({"M1 all": tab1["coef"], "M3 no-trunc": tab3["coef"]})
    comp["shift"] = (comp["M3 no-trunc"] - comp["M1 all"]).round(3)
    print(comp.dropna())

    # =============================================================== RQ1b: alignment
    section("RQ1b — ALIGNMENT WITH HOME-JURISDICTION POLICY")

    # pro_score: positive = the response argued PRO the issue as phrased.
    b["pro_score"] = 3 - b["stance_num"]

    print("\nissue coverage per jurisdiction (issues with a pro/con code; mixed & unclear drop out):")
    issues = b.drop_duplicates("issue_id")
    for j, col in JURIS.items():
        n = issues[col].isin(SIGN).sum()
        print(f"  {j}: {n}/11 issues  "
              f"(pro={int((issues[col] == 'pro').sum())}, con={int((issues[col] == 'con').sum())}, "
              f"dropped={11 - n})")

    # --- measure 1 (pre-specified): signed centred-stance aggregate.
    rows = []
    for j, col in JURIS.items():
        s = b[b[col].isin(SIGN)].copy()
        s["align"] = signed_alignment(s, col)
        g = s.groupby(["model", "lang"])["align"].mean().unstack()
        g.columns = [f"{j}_{c}" for c in g.columns]
        rows.append(g)
    align = pd.concat(rows, axis=1).round(3)
    align["origin"] = b.drop_duplicates("model").set_index("model")["origin"]

    # pro-regulation tendency: the confound.
    align["pro_reg"] = (b.groupby("model")["pro_score"].mean() / 2).round(3)
    align = align.sort_values("origin")
    print("\nSigned alignment (+1 = fully aligned with that jurisdiction, -1 = fully opposed):")
    print(align[["origin", "US_en", "US_zh", "CN_en", "CN_zh", "EU_en", "EU_zh", "pro_reg"]])
    align.to_csv(TABLES / "nb1_alignment_signed.csv")

    # The actual RQ1 test: does each model score highest on its OWN jurisdiction?
    print("\nHOME-BLOC TEST — each model's alignment with its own home jurisdiction,")
    print("minus its mean alignment with the other two. Positive = home-aligned.")
    home = []
    for m, r in align.iterrows():
        o = r["origin"]
        for lang in ["en", "zh"]:
            own = r[f"{o}_{lang}"]
            others = np.mean([r[f"{k}_{lang}"] for k in JURIS if k != o])
            home.append({"model": m, "origin": o, "lang": lang,
                         "own": round(own, 3), "others": round(others, 3),
                         "home_advantage": round(own - others, 3)})
    home = pd.DataFrame(home).sort_values(["origin", "model", "lang"])
    print(home.to_string(index=False))
    home.to_csv(TABLES / "nb1_home_advantage.csv", index=False)

    # --- measure 2 (pre-specified): modal-stance consistency rate.
    print("\nModal-stance consistency rate (% of model x issue cells whose modal stance")
    print("falls on the same side of neutral as the jurisdiction's documented position):")
    mode_rows = []
    for j, col in JURIS.items():
        s = b[b[col].isin(SIGN)]
        cell = s.groupby(["model", "issue_id", col])["stance_num"].agg(
            lambda x: x.mode().iloc[0]
        ).reset_index()
        cell["side"] = np.sign(3 - cell["stance_num"])          # +1 pro, 0 neutral, -1 con
        cell["want"] = cell[col].map(SIGN)
        cell["consistent"] = cell["side"] == cell["want"]        # a neutral mode is NOT consistent
        r = cell.groupby("model")["consistent"].mean().mul(100).round(1)
        mode_rows.append(r.rename(j))
    modal = pd.concat(mode_rows, axis=1)
    modal["origin"] = b.drop_duplicates("model").set_index("model")["origin"]
    modal = modal.sort_values("origin")
    print(modal)
    modal.to_csv(TABLES / "nb1_modal_consistency.csv")

    # alignment by issue category, for the model that is each bloc's cleanest case
    print("\nSigned alignment by issue category (English, own-jurisdiction code only):")
    cat_rows = []
    for m in b["model"].cat.categories:
        o = b[b["model"] == m]["origin"].iloc[0]
        col = JURIS[o]
        s = b[(b["model"] == m) & (b["lang"] == "en") & (b[col].isin(SIGN))].copy()
        s["align"] = signed_alignment(s, col)
        cat_rows.append(s.groupby("issue_category")["align"].mean().rename(f"{m} ({o})"))
    print(pd.concat(cat_rows, axis=1).round(2))

    # ================================================= RQ1c: the unconfounded origin test
    section("RQ1c — CLEAN ORIGIN TEST (US/CN strictly-opposed issues only)")

    opposed = strictly_opposed(b)

    iss = sorted(opposed["issue_id"].unique(), key=lambda s: int(s[1:]))
    print(f"strictly-opposed issues ({len(iss)}): {iss}")
    print("  direction of each (US pole = the side US policy is on):")
    for i in iss:
        r = opposed[opposed["issue_id"] == i].iloc[0]
        print(f"    {i:4s} US={r['us_position']:4s} CN={r['china_position']:4s}  {r['issue_category']}")

    opposed["us_pole"] = opposed["pro_score"] * opposed["us_position"].map(SIGN) / 2

    print("\nMean us_pole score by model x language "
          "(+1 = took the US side, -1 = took the Chinese side, 0 = neutral):")
    piv = opposed.pivot_table(index="model", columns="lang", values="us_pole", aggfunc="mean").round(3)
    piv["origin"] = opposed.drop_duplicates("model").set_index("model")["origin"]
    piv = piv.sort_values("origin")
    print(piv)
    piv.to_csv(TABLES / "nb1_uspole_by_model.csv")
    print("\n  If origin drove stance, the US models would sit ABOVE the Chinese models here.")

    print("\nMean us_pole by model x issue (English):")
    pi = (
        opposed[opposed["lang"] == "en"]
        .pivot_table(index="model", columns="issue_id", values="us_pole", aggfunc="mean")
        .round(2)[iss]
    )
    pi.loc["-- spread across models --"] = (pi.max() - pi.min()).round(2)
    print(pi)
    pi.to_csv(TABLES / "nb1_uspole_by_model_issue_en.csv")

    # the formal test, within issue and template
    res4, tab4 = fit_mixed(
        opposed,
        "us_pole ~ C(origin) * C(lang)",
        "M4  us_pole ~ origin * language   (7 strictly-opposed issues; POSITIVE = US side)",
    )
    tab4.to_csv(TABLES / "nb1_mixed_uspole_origin.csv")
    print("  Read: C(origin)[T.CN] is how much MORE the US pole a Chinese-origin model takes")
    print("        than a US-origin model, on the same issue. Origin alignment predicts it")
    print("        to be NEGATIVE and large.")

    res5, tab5 = fit_mixed(
        opposed, "us_pole ~ C(model) * C(lang)", "M5  same, per model (claude = reference)"
    )
    tab5.to_csv(TABLES / "nb1_mixed_uspole_model.csv")

    print(f"\n\nTables written to {TABLES}/")


def run_origin_robustness():
    """nb1b -- RQ1 robustness: does the continuous-scale assumption matter? Formerly
    rq1_origin_robustness.py.

    The stance scale (1 = only pro ... 5 = only con) is ORDINAL, not interval, and
    31.7% of responses sit at the floor (stance 1), which could compress differences
    between very-pro models. Two checks: PART 1 refits everything as an ordered logit
    (order-only assumption); PART 2 recomputes Lim & Rottger's own distributional
    metrics (JSD + Cramer's V) so the comparison with the predecessor paper is direct.
    If sign and significance survive both, the interval-scale assumption in nb1 is not
    load-bearing.
    """
    df = load_master()
    b_all = df[df["family"] == "B"].copy()          # keeps refusals: needed for the histograms
    b = b_all[~b_all["is_refusal"]].copy()          # drops them: needed for the stance models

    # =========================================================== PART 1: ordered logit
    section("PART 1 — ORDERED LOGIT (proportional odds). Only ORDER is assumed.")

    # O1: the headline model. Outcome = raw stance, so POSITIVE = more CON.
    o1 = ordered_logit(b, "stance_num", "O1  stance (1-5) ~ origin * lang   [+ = more CON]")
    o1.to_csv(TABLES / "nb1b_ordlogit_stance.csv")
    print("  nb1 linear M1 said: CN +0.010 (n.s.), EU +0.044 (p=.002), zh -0.105 (p<.001)")

    # O2-O4: the clean origin test, ordinally.
    #   us_ord = stance         if the US is PRO the proposition  (1 = most US-aligned)
    #   us_ord = 6 - stance     if the US is CON the proposition
    # => us_ord runs 1 (most US-aligned) .. 5 (most China-aligned) on every opposed issue.
    # A POSITIVE coefficient here = leans toward CHINA'S pole -- the OPPOSITE sign
    # convention to nb1's continuous `us_pole` (where negative = China's pole).
    op = strictly_opposed(b)
    op["us_ord"] = np.where(op["us_position"] == "pro", op["stance_num"], 6 - op["stance_num"])

    print(f"\nstrictly-opposed issues: {sorted(op['issue_id'].unique(), key=lambda s: int(s[1:]))}")
    o2 = ordered_logit(op, "us_ord", "O2  us_ord ~ origin * lang  [7 opposed issues; + = CHINA's pole]")
    o2.to_csv(TABLES / "nb1b_ordlogit_uspole.csv")
    print("  nb1 linear M4 said (sign FLIPPED there): CN -0.027 (p<.001), zh -0.109 (p<.001),")
    print("  CN:zh -0.038 (p<.001).  So here we expect all three POSITIVE.")

    # The direction split — the finding that actually matters.
    o3 = ordered_logit(
        op[op["us_position"] == "con"], "us_ord",
        "O3  SUBSET A: 5 issues, China's pole = PRO-regulation  [+ = China's pole]",
        absorb="C(issue_id) + C(template_id)",
    )
    o4 = ordered_logit(
        op[op["us_position"] == "pro"], "us_ord",
        "O4  SUBSET B: 2 issues (B5, B10), China's pole = ANTI-regulation  [+ = China's pole]",
        absorb="C(issue_id) + C(template_id)",
    )
    o3.to_csv(TABLES / "nb1b_ordlogit_subsetA.csv")
    o4.to_csv(TABLES / "nb1b_ordlogit_subsetB.csv")
    print("\n  nb1 linear said: Subset A -> CN -0.036 (p<.001), CN:zh +0.012 (n.s.)")
    print("                   Subset B -> CN -0.002 (p=.899), CN:zh -0.163 (p<.001)")
    print("  Ordinally, the Subset B CN:zh term should be POSITIVE and large, and the")
    print("  Subset B CN main effect should stay null. That is the finding to protect.")

    # =============================================== PART 2: Lim & Rottger's own metrics
    section("PART 2 — DISTRIBUTIONAL METRICS (Lim & Rottger's JSD + Cramer's V)")

    # Their headline comparison: for each issue, is the stance distribution moved MORE by
    # swapping the model's origin (holding language fixed), or by swapping the prompt
    # language (holding origin fixed)? This is exactly RQ1 vs RQ2, in their metric.
    origins, langs = ["US", "CN", "EU"], ["en", "zh"]
    rows = []
    for issue in sorted(b_all["issue_id"].unique(), key=lambda s: int(s[1:])):
        d_i = b_all[b_all["issue_id"] == issue]

        o_jsds = [
            calculate_jsd(
                stance_dist(d_i[(d_i["origin"] == a) & (d_i["lang"] == lg)]),
                stance_dist(d_i[(d_i["origin"] == c) & (d_i["lang"] == lg)]),
            )
            for lg in langs
            for a, c in [("US", "CN"), ("US", "EU"), ("CN", "EU")]
        ]
        l_jsds = [
            calculate_jsd(
                stance_dist(d_i[(d_i["origin"] == o) & (d_i["lang"] == "en")]),
                stance_dist(d_i[(d_i["origin"] == o) & (d_i["lang"] == "zh")]),
            )
            for o in origins
        ]

        ct_o = pd.crosstab(d_i["stance"].astype(str), d_i["origin"])
        ct_l = pd.crosstab(d_i["stance"].astype(str), d_i["lang"])

        rows.append({
            "issue": issue,
            "JSD_origin": np.mean(o_jsds),
            "JSD_language": np.mean(l_jsds),
            "V_origin": cramers_v(ct_o.values),
            "V_language": cramers_v(ct_l.values),
        })

    jsd = pd.DataFrame(rows).set_index("issue")
    jsd["lang/origin"] = (jsd["JSD_language"] / jsd["JSD_origin"]).round(1)
    print("\nPer issue — how much does the stance DISTRIBUTION move? (higher = bigger effect)")
    print(jsd.round(4))
    jsd.to_csv(TABLES / "nb1b_jsd_cramersv.csv")

    print("\nMEANS ACROSS THE 11 B-ISSUES:")
    print(f"  JSD, swapping model origin   : {jsd['JSD_origin'].mean():.4f}")
    print(f"  JSD, swapping prompt language: {jsd['JSD_language'].mean():.4f}"
          f"   ({jsd['JSD_language'].mean() / jsd['JSD_origin'].mean():.1f}x larger)")
    print(f"  Cramer's V, origin           : {jsd['V_origin'].mean():.4f}  (0.1 = small effect)")
    print(f"  Cramer's V, language         : {jsd['V_language'].mean():.4f}")
    print(f"\n  Issues where LANGUAGE moves the distribution more than ORIGIN: "
          f"{int((jsd['JSD_language'] > jsd['JSD_origin']).sum())}/11")

    # Same metric applied to the models individually, as the sanity check on "origin is
    # the wrong unit": swapping the MODEL should move the distribution more than origin.
    m_rows = []
    for issue in sorted(b_all["issue_id"].unique(), key=lambda s: int(s[1:])):
        d_i = b_all[b_all["issue_id"] == issue]
        models = sorted(d_i["model"].unique())
        js = [
            calculate_jsd(
                stance_dist(d_i[(d_i["model"] == a) & (d_i["lang"] == lg)]),
                stance_dist(d_i[(d_i["model"] == c) & (d_i["lang"] == lg)]),
            )
            for lg in langs
            for i, a in enumerate(models)
            for c in models[i + 1:]
        ]
        m_rows.append({"issue": issue, "JSD_model": np.mean(js)})
    mj = pd.DataFrame(m_rows).set_index("issue")
    print(f"\n  JSD, swapping the individual MODEL : {mj['JSD_model'].mean():.4f}")
    print("  (compare with JSD_origin above — if model >> origin, origin is the wrong unit)")

    print(f"\n\nTables written to {TABLES}/")


def run_stance_variance():
    """nb1e -- within-issue variance decomposition of judged stance. Formerly
    rq1_stance_variance.py.

    The issue itself dominates total stance variance by construction, so the raw
    issue share is a property of the design, not a finding. The informative
    decomposition is WITHIN issue: a conditioning ladder (share of remaining
    variance explained by template, model, origin, once issue -- and then issue +
    template -- are held fixed), plus a per-issue eta^2 breakdown. Writes
    nb1e_within_issue_variance.csv, nb1e_per_issue_variance.csv, and matching
    figures.
    """
    LABEL = {"template_id": "template", "lang": "language"}

    def eta_squared(d: pd.DataFrame, factor: str, y: str = "stance_num") -> float:
        v = d[y].to_numpy(float)
        total = ((v - v.mean()) ** 2).sum()
        between = d.groupby(factor)[y].agg(["mean", "size"])
        return float((between["size"] * (between["mean"] - v.mean()) ** 2).sum() / total)

    apply_style()
    df = load_master()
    b = df[(df["family"] == "B") & (~df["is_refusal"])].copy()

    # ---- 1. conditioning ladder (issue absorbed, then issue + template, as in m1)
    b["stance_wi"] = b["stance_num"] - b.groupby(["issue_id", "lang"])["stance_num"].transform("mean")
    b["stance_wit"] = np.nan
    for lang, d in b.groupby("lang"):
        r = smf.ols("stance_num ~ C(issue_id) + C(template_id)", d).fit()
        b.loc[d.index, "stance_wit"] = r.resid
    FACTORS_WI = ["template_id", "model", "origin"]
    wi = pd.DataFrame(
        [{"lang": lang, "held_fixed": "issue", "factor": LABEL.get(f, f),
          "eta_sq": eta_squared(d, f, "stance_wi")}
         for lang, d in b.groupby("lang") for f in FACTORS_WI]
        + [{"lang": lang, "held_fixed": "issue+template", "factor": LABEL.get(f, f),
            "eta_sq": eta_squared(d, f, "stance_wit")}
           for lang, d in b.groupby("lang") for f in ["model", "origin"]]
    )
    wi.to_csv(TABLES / "nb1e_within_issue_variance.csv", index=False)
    print("CONDITIONING LADDER (eta^2 on residualized stance)")
    print(wi.pivot(index=["held_fixed", "factor"], columns="lang", values="eta_sq").round(4))

    # Figure: single panel, issue AND template held fixed (the conditioning of
    # Equation m1), within language.
    facts = ["model", "origin"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    w = 0.38
    x = np.arange(len(facts))
    sub = wi[wi["held_fixed"] == "issue+template"]
    for i, lang in enumerate(["en", "zh"]):
        d = sub[sub["lang"] == lang].set_index("factor").loc[facts]
        pct = d["eta_sq"] * 100
        ax.bar(x + (i - 0.5) * w, pct, width=w - 0.03,
               label=lang.upper(), color=SLOT[i], zorder=3)
        for xi, v in zip(x + (i - 0.5) * w, pct):
            ax.text(xi, v + 0.16, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_ylim(top=4.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(facts)
    ax.set_ylabel("share of remaining stance variance ($\\eta^2$)")
    ax.set_title("With issue and template held fixed,\nwhat moves stance?", fontsize=11, pad=20)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(title=None, loc="upper right")
    save(fig, FIG / "nb1e_within_issue_variance")

    # ---- 2. per-issue decomposition
    FACTORS_PI = ["template_id", "model", "lang", "origin"]
    pi = pd.DataFrame(
        {"issue_id": iid, "factor": LABEL.get(f, f), "eta_sq": eta_squared(d, f)}
        for iid, d in b.groupby("issue_id")
        for f in FACTORS_PI
    )
    pi.to_csv(TABLES / "nb1e_per_issue_variance.csv", index=False)
    piv = pi.pivot(index="issue_id", columns="factor", values="eta_sq")
    piv = piv.loc[sorted(piv.index, key=lambda s: int(s[1:]))]
    print("\nPER-ISSUE VARIANCE SHARES (eta^2 within each core issue)")
    print(piv[[LABEL.get(f, f) for f in FACTORS_PI]].round(3))

    # Origin is nested within model, so eta^2(model) = eta^2(origin) + eta^2(model
    # within origin). Stack them to show origin is a sliver on almost every issue.
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(piv))
    origin = piv["origin"].to_numpy()
    model_within = (piv["model"] - piv["origin"]).clip(lower=0).to_numpy()
    ax.bar(x, origin, width=0.6, color=SLOT[4], zorder=3,
           label="origin (between origin groups)")
    ax.bar(x, model_within, bottom=origin, width=0.6, color=SLOT[0], zorder=3,
           label="model, within origin")
    ax.plot(x, piv["language"], lw=1.4, ms=5, marker="^", color=SLOT[2],
            zorder=4, label="language")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("share of stance variance ($\\eta^2$)")
    ax.set_title("Where the variation lives differs by issue")
    ax.legend(title=None, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              columnspacing=1.6, frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    save(fig, FIG / "nb1e_per_issue_variance")


def run_alignment_by_model():
    """nb1d -- per-model directional alignment table for the RQ1 write-up. Formerly
    rq1_alignment_by_model.py.

    One row per model, reporting the signed alignment measures at three levels of
    restriction: toward US/China/EU on each jurisdiction's full codeable issue set,
    the model's overall pro-regulation lean (the confound column), toward China on
    the 7 strictly-opposed US-vs-China issues, and toward China on B5+B10 only
    (where China's pole is anti-regulation). Writes nb1d_alignment_by_model.csv.
    """
    df = load_master()
    b = df[(df["family"] == "B") & (~df["is_refusal"])].copy()
    b["pro_score"] = 3 - b["stance_num"]  # positive = argued PRO the issue as phrased

    rows = []
    for m, blk in b.groupby("model"):
        row = {"model": m, "origin": blk["origin"].iloc[0]}
        for lang, bl in blk.groupby("lang"):
            # full codeable set, per jurisdiction (nb1's signed alignment)
            for j, col in JURIS.items():
                cod = bl[bl[col].isin(SIGN)]
                row[f"{j}_{lang}"] = signed_alignment(cod, col).mean()
            # strictly opposed US-vs-China issues, scored toward CHINA's pole
            opp = strictly_opposed(bl)
            row[f"CN_opp7_{lang}"] = -signed_alignment(opp, "us_position").mean()
            two = opp[opp["issue_id"].isin(["B5", "B10"])]
            row[f"CN_b5b10_{lang}"] = -signed_alignment(two, "us_position").mean()
        row["pro_reg"] = (blk["pro_score"] / 2).mean()
        rows.append(row)

    out = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]
    cols = (["origin", "pro_reg"]
            + [f"{j}_{lg}" for lg in ["en", "zh"] for j in JURIS]
            + [f"CN_opp7_{lg}" for lg in ["en", "zh"]]
            + [f"CN_b5b10_{lg}" for lg in ["en", "zh"]])
    out = out[cols].round(3)
    print(out.to_string())
    out.to_csv(TABLES / "nb1d_alignment_by_model.csv")
    print("\nWritten to", TABLES / "nb1d_alignment_by_model.csv")


def run_clustered_inference():
    """nb2c -- RQ1/RQ2 inference robustness: honest standard errors. Formerly
    rq12_clustered_inference.py.

    Every mixed model in run_origin_main() has random intercepts for issue and
    template but treats residuals as independent, ignoring (a) the three reps of
    each prompt cell sharing the prompt's systematic component, and (b) origin and
    language effects varying across issues, so the effective N for a cross-issue
    claim is closer to 11 (issues) than 39,551 (responses). Three mutually
    reinforcing re-estimates: (1) OLS with issue + template fixed effects and
    cluster-robust SEs at the issue level; (2) wild cluster bootstrap p-values
    (Rademacher, null imposed); (3) two-step, issue as the unit of analysis. Writes
    nb2c_toward_china_per_issue.csv, which rq2.py's toward-China figure section
    reads -- rq2.py must be run AFTER this function (i.e. after rq1.py).
    """
    df = load_master()
    b = df[(df["family"] == "B") & (~df["is_refusal"])].copy()
    b["origin"] = pd.Categorical(b["origin"], categories=["US", "CN", "EU"])
    b["lang"] = pd.Categorical(b["lang"], categories=["en", "zh"])
    # prompt cell: one prompt x one model (the unit the 3 reps re-sample)
    b["cell"] = (b["model"].astype(str) + "|" + b["template_id"].astype(str) + "|"
                 + b["lang"].astype(str) + "|" + b["issue_id"].astype(str))

    print(f"B-series non-refusal responses: {len(b):,} "
          f"({b['cell'].nunique():,} prompt cells, {b['issue_id'].nunique()} issues)")

    # =========================================== PART 1: M1 with honest SEs
    section("PART 1 — THE PRE-SPECIFIED M1, RE-ESTIMATED (issue-clustered, wild bootstrap)")
    print("Old (nb1 M1, mixed model, iid residuals): CN +0.010 p=.371, EU +0.044 p=.002,")
    print("zh -0.105 p<.001, CN:zh +0.035 p=.025. Point estimates should match closely;")
    print("only the uncertainty changes.")

    M1_F = "stance_num ~ C(origin) * C(lang) + C(issue_id) + C(template_id)"
    m1, _ = cluster_ols(M1_F, b, "issue_id", "M1  stance ~ origin * lang  [FE + cluster(issue)]",
                        KEEP)

    # wild cluster bootstrap on the same design
    print("\nWild cluster bootstrap (999 draws, Rademacher, null imposed), same design:")
    Xd = dmatrix(M1_F.split("~")[1], b, return_type="dataframe")
    Xn = Xd.to_numpy()
    yn = b["stance_num"].to_numpy()
    cl = pd.factorize(b["issue_id"])[0]
    wcb_rows = []
    for key in KEEP:
        k = Xd.columns.get_loc(key)
        t_obs, p_wcb = wild_cluster_p(Xn, yn, cl, k)
        wcb_rows.append({"coef": key, "t_obs": round(t_obs, 3), "p_wcb": round(p_wcb, 4)})
        print(f"  {key:38s} t = {t_obs:+.3f}   p_wcb = {p_wcb:.4f}")
    wcb = pd.DataFrame(wcb_rows)

    m1out = m1.copy()
    m1out["p_wcb"] = wcb.set_index("coef")["p_wcb"]
    m1out.round(4).to_csv(TABLES / "nb2c_m1_clustered.csv")

    # =========================================== PART 2: two-step, issue as the unit
    section("PART 2 — TWO-STEP: ESTIMATE WITHIN EACH ISSUE, TEST ACROSS THE 11 ISSUES")
    print("The most transparent restatement: each effect is estimated within each issue")
    print("(identification is clean there), and the question 'does it generalise' is put")
    print("to the 11 per-issue coefficients directly.\n")

    rows = []
    for iid, blk in b.groupby("issue_id"):
        r = smf.ols("stance_num ~ C(origin) * C(lang) + C(template_id)", blk).fit()
        rows.append({"issue": iid, **{k: r.params[k] for k in KEEP}})
    per_issue = pd.DataFrame(rows).set_index("issue")
    per_issue = per_issue.reindex(sorted(per_issue.index, key=lambda s: int(s[1:])))
    print("Per-issue coefficients:")
    print(per_issue.round(3))
    per_issue.round(4).to_csv(TABLES / "nb2c_per_issue_coefs.csv")

    print("\nAcross the 11 issues (one-sample t, 10 df; sign counts):")
    ts_rows = []
    for k in KEEP:
        v = per_issue[k].to_numpy()
        t, p = ttest_1samp(v, 0)
        npos = int((v > 0).sum())
        p_sign = binomtest(npos, len(v), 0.5).pvalue
        ts_rows.append({"coef": k, "mean": v.mean(), "sd": v.std(ddof=1), "t": t,
                        "p_t": p, "n_pos": f"{npos}/{len(v)}", "p_sign": p_sign})
    ts = pd.DataFrame(ts_rows)
    print(ts.round(4).to_string(index=False))
    ts.round(4).to_csv(TABLES / "nb2c_two_step.csv", index=False)

    # the signed (toward-China) version of the language effect — the directional claim
    print("\nSigned version — per-issue zh effect TOWARD CHINA'S documented position")
    print("(9 codeable issues; this is the claim RQ2 actually rests on):")
    tc_rows = []
    for iid, blk in b.groupby("issue_id"):
        cn = blk["china_position"].iloc[0]
        if cn not in SIGN:
            continue
        r = smf.ols("stance_num ~ C(lang) + C(template_id) + C(model)", blk).fit()
        tc_rows.append({"issue": iid, "toward_china": -r.params["C(lang)[T.zh]"] * SIGN[cn]})
    tc = pd.DataFrame(tc_rows).sort_values("issue", key=lambda c: c.map(lambda s: int(s[1:])))
    print(tc.round(3).to_string(index=False))
    v = tc["toward_china"].to_numpy()
    t_tc, p_tc = ttest_1samp(v, 0)
    npos_tc = int((v > 0).sum())
    p_sign_tc = binomtest(npos_tc, len(v), 0.5).pvalue
    print(f"\n  mean {v.mean():+.3f}, t({len(v) - 1}) = {t_tc:.2f}, p = {p_tc:.4g}; "
          f"positive on {npos_tc}/{len(v)} issues (sign test p = {p_sign_tc:.4f})")
    tc.round(4).to_csv(TABLES / "nb2c_toward_china_per_issue.csv", index=False)

    # =========================================== PART 3: the subset tests, honestly
    section("PART 3 — SUBSET A / SUBSET B (the RQ1/RQ2 identification), HONEST SEs")

    op = strictly_opposed(b)
    op["china_pole"] = np.where(op["us_position"] == "pro",
                                op["stance_num"], 6 - op["stance_num"])
    sub_a = op[op["us_position"] == "con"].copy()   # 5 issues, China's pole = PRO-regulation
    sub_b = op[op["us_position"] == "pro"].copy()   # B5, B10: China's pole = ANTI-regulation

    print("SUBSET A (5 issues) — pooled, clustered by issue (G = 5: read with care), plus")
    print("the per-issue zh coefficients, which are the honest evidence:")
    a_cl, _ = cluster_ols("china_pole ~ C(origin) * C(lang) + C(issue_id) + C(template_id)",
                          sub_a, "issue_id", "A  china_pole ~ origin * lang  [+ = toward China]",
                          KEEP)
    a_cl.round(4).to_csv(TABLES / "nb2c_subsetA_clustered.csv")
    a_rows = []
    for iid, blk in sub_a.groupby("issue_id"):
        r = smf.ols("china_pole ~ C(origin) * C(lang) + C(template_id)", blk).fit()
        a_rows.append({"issue": iid, "zh": r.params["C(lang)[T.zh]"]})
    a_pi = pd.DataFrame(a_rows)
    print("\n  per-issue zh coefficient (toward China):")
    print(a_pi.round(3).to_string(index=False))
    va = a_pi["zh"].to_numpy()
    t, p = ttest_1samp(va, 0)
    print(f"  mean {va.mean():+.3f}, t({len(va) - 1}) = {t:.2f}, p = {p:.4g}; "
          f"positive on {int((va > 0).sum())}/{len(va)}")

    print("\nSUBSET B (B5, B10) — a WITHIN-these-two-issues claim, so clustering is at the")
    print("prompt cell (the 3 reps of one prompt x model), per issue and pooled:")
    b_cl, _ = cluster_ols("china_pole ~ C(origin) * C(lang) + C(issue_id) + C(template_id)",
                          sub_b, "cell", "B pooled  [cluster = prompt cell]", KEEP)
    b_cl.round(4).to_csv(TABLES / "nb2c_subsetB_clustered.csv")
    for iid, blk in sub_b.groupby("issue_id"):
        cluster_ols("china_pole ~ C(origin) * C(lang) + C(template_id)",
                    blk, "cell", f"B per-issue: {iid}", KEEP)

    # =========================================== PART 4: per-model language effect
    section("PART 4 — PER-MODEL LANGUAGE EFFECT (Table rq2models), CLUSTERED BY ISSUE")
    print("Old: all six positive at p<.001 (iid). Re-test per model on the 7 opposed issues,")
    print("clustered by issue (G = 7, t on 6 df). The direction claim is what matters.\n")

    pm_rows = []
    for mdl, blk in op.groupby("model"):
        res = smf.ols("china_pole ~ C(lang) + C(issue_id) + C(template_id)", blk).fit(
            cov_type="cluster", cov_kwds={"groups": blk["issue_id"]}, use_t=True)
        k = "C(lang)[T.zh]"
        lo, hi = res.conf_int().loc[k]
        pm_rows.append({"model": mdl, "origin": blk["origin"].iloc[0],
                        "zh_toward_china": res.params[k], "se": res.bse[k],
                        "p_cluster": res.pvalues[k], "ci_lo": lo, "ci_hi": hi})
    pm = pd.DataFrame(pm_rows).sort_values("zh_toward_china", ascending=False)
    print(pm.round(4).to_string(index=False))
    pm.round(4).to_csv(TABLES / "nb2c_per_model_language_clustered.csv", index=False)

    # =========================================== PART 5: per-rep ordered logits (B5/B10)
    section("PART 5 — THE OR 2.94, PER REP (three independent replications)")
    print("Within one rep every prompt is decoded once, independently, so a per-rep ordered")
    print("logit has valid conventional SEs. Three reps = three independent replications of")
    print("the Subset B result. Pooled (nb1b) said: CN:zh OR = 2.94, zh OR = 1.40.\n")

    or_rows = []
    for rep, blk in sub_b.groupby("rep"):
        d = blk.copy()
        d["_y"] = pd.Categorical(d["china_pole"].astype(int), ordered=True)
        res = OrderedModel.from_formula(
            "_y ~ C(origin) * C(lang) + C(issue_id) + C(template_id)", d, distr="logit"
        ).fit(method="bfgs", maxiter=400, disp=False)
        if np.isnan(res.bse).any():  # shaky Hessian at the bfgs stopping point: polish
            res = OrderedModel.from_formula(
                "_y ~ C(origin) * C(lang) + C(issue_id) + C(template_id)", d, distr="logit"
            ).fit(method="lbfgs", maxiter=2000, disp=False)
        row = {"rep": rep, "n": len(d), "converged": res.mle_retvals.get("converged", True)}
        for key, name in [("C(lang)[T.zh]", "zh"), ("C(origin)[T.CN]:C(lang)[T.zh]", "CNxzh")]:
            row[f"OR_{name}"] = np.exp(res.params[key])
            row[f"p_{name}"] = res.pvalues[key]
        or_rows.append(row)
    ors = pd.DataFrame(or_rows)
    print(ors.round(4).to_string(index=False))
    ors.round(4).to_csv(TABLES / "nb2c_subsetB_ordlogit_per_rep.csv", index=False)

    # =========================================== PART 6: the collapsed-scale chi2, per rep
    section("PART 6 — THE DECISIVE COLLAPSED-SCALE TEST (nb2b GAP 3), PER REP")
    print("nb2b's chi2 = 258 pooled three reps as independent, which inflates chi2 (though")
    print("not Cramer's V). Per rep, each of which IS internally independent:\n")

    sb = df[(df["family"] == "B") & df["issue_id"].isin(["B5", "B10"]) & ~df["is_refusal"]].copy()
    sb["stance_3"] = sb["stance_num"].map(lambda s: "pro" if s <= 2 else ("neutral" if s == 3 else "con"))
    for rep, blk in sb.groupby("rep"):
        ct = pd.crosstab(blk["stance_3"], blk["lang"])
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.to_numpy().sum()
        v = np.sqrt(chi2 / n / (min(ct.shape) - 1))
        china = blk.groupby("lang")["stance_3"].apply(lambda s: (s == "con").mean() * 100)
        print(f"  {rep}:  chi2 = {chi2:6.1f}  p = {p:.3g}  V = {v:.3f}   "
              f"on China's side: en {china['en']:.1f}% -> zh {china['zh']:.1f}%")

    # =========================================== PART 7: verdict
    section("PART 7 — VERDICT: WHICH WRITE-UP CLAIMS SURVIVE HONEST SEs?")
    ver = pd.DataFrame([
        {"claim": "zh main effect (M1, -0.105)", "old_p": "<.001",
         "honest": f"cluster p={m1.loc['C(lang)[T.zh]', 'p']:.4f}, "
                   f"wcb p={m1out.loc['C(lang)[T.zh]', 'p_wcb']:.4f}, "
                   f"two-step p={ts.set_index('coef').loc['C(lang)[T.zh]', 'p_t']:.4f}"},
        {"claim": "toward-China zh shift (9/9 issues)", "old_p": "<.001",
         "honest": f"t-test p={p_tc:.4g}, sign test {npos_tc}/9 p={p_sign_tc:.4f}"},
        {"claim": "EU origin (M1, +0.044)", "old_p": ".002",
         "honest": f"cluster p={m1.loc['C(origin)[T.EU]', 'p']:.4f}, "
                   f"wcb p={m1out.loc['C(origin)[T.EU]', 'p_wcb']:.4f}"},
        {"claim": "CN x zh raw-scale interaction (M1, +0.035)", "old_p": ".025",
         "honest": f"cluster p={m1.loc['C(origin)[T.CN]:C(lang)[T.zh]', 'p']:.4f}, "
                   f"wcb p={m1out.loc['C(origin)[T.CN]:C(lang)[T.zh]', 'p_wcb']:.4f}"},
        {"claim": "Subset B CN x zh (within B5/B10)", "old_p": "<.001",
         "honest": f"prompt-cell cluster p={b_cl.loc['C(origin)[T.CN]:C(lang)[T.zh]', 'p']:.2g}; "
                   f"per-rep ORs {ors['OR_CNxzh'].round(2).tolist()}"},
    ])
    print(ver.to_string(index=False))
    ver.to_csv(TABLES / "nb2c_verdict.csv", index=False)

    section("DONE — tables written to evaluation/analysis/outputs/tables/nb2c_*.csv")


def run_model_similarity_jsd():
    """nb1f -- pairwise model similarity on the core issues (IssueBench Figure 6
    analogue).

    For each pair of models, the Jensen-Shannon divergence between their stance
    distributions over the six bins (1-5 plus refusal), per core issue, averaged
    across the eleven core issues. One panel per prompt language, because RQ2
    turns on the language difference and pooling would hide it. Uses the same
    calculate_jsd/stance_dist as run_origin_robustness(), so cell values are on
    the same scale as the factor-swap means reported there.
    """
    LANGS = [("en", "English prompts"), ("zh", "Chinese prompts")]
    MODEL_DISPLAY = {"claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
                     "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral"}
    # MODEL_ORDER is already US, US, US, CN, CN, EU -- the origin blocks are contiguous,
    # so the dashed rules below just need the boundaries between them
    BLOCK_EDGES = [3, 5]
    ORIGIN_BANDS = [("US", 0, 3), ("China", 3, 5), ("EU", 5, 6)]

    def matrix_for(b_all, lang):
        """Mean pairwise JSD across the core issues, for one prompt language."""
        issues = sorted(b_all["issue_id"].unique(), key=lambda s: int(s[1:]))
        n = len(MODEL_ORDER)
        mat = np.zeros((n, n))
        for i, a in enumerate(MODEL_ORDER):
            for j, c in enumerate(MODEL_ORDER):
                if j <= i:
                    continue
                per_issue = [
                    calculate_jsd(
                        stance_dist(b_all[(b_all["issue_id"] == iss)
                                          & (b_all["model"] == a) & (b_all["lang"] == lang)]),
                        stance_dist(b_all[(b_all["issue_id"] == iss)
                                          & (b_all["model"] == c) & (b_all["lang"] == lang)]),
                    )
                    for iss in issues
                ]
                mat[i, j] = mat[j, i] = np.mean(per_issue)
        return mat

    apply_style()
    df = load_master()
    # refusals stay in: they are one of the six distribution bins, as in Lim & Rottger
    b_all = df[df["family"] == "B"].copy()

    mats = {lang: matrix_for(b_all, lang) for lang, _ in LANGS}
    vmax = max(m.max() for m in mats.values())

    # sequential ramp, one hue light -> dark: the cell value is a magnitude
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + SEQUENTIAL)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9))
    for ax, (lang, title) in zip(axes, LANGS):
        mat = mats[lang]
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10, color=INK, pad=26)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(len(MODEL_ORDER)),
                      [MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=90)
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_DISPLAY[m] for m in MODEL_ORDER])
        ax.tick_params(length=0, labelsize=8.5)

        # value in every cell: the ramp carries magnitude, the number carries precision
        for i in range(len(MODEL_ORDER)):
            for j in range(len(MODEL_ORDER)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.6,
                        color="#ffffff" if v > 0.62 * vmax else INK_MUTED)

        # dashed rules between the origin blocks, as in IssueBench Figure 6
        for e in BLOCK_EDGES:
            ax.axhline(e - 0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.axvline(e - 0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))

        # origin band labels above the columns
        for name, lo, hi in ORIGIN_BANDS:
            ax.text((lo + hi - 1) / 2, -0.92, name, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=INK_MUTED)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.20, wspace=0.42)
    save(fig, FIG / "nb1f_model_similarity_jsd")

    # ship the numbers alongside the figure
    out = []
    for lang, _ in LANGS:
        t = pd.DataFrame(mats[lang], index=MODEL_ORDER, columns=MODEL_ORDER)
        t.insert(0, "lang", lang)
        out.append(t)
    pd.concat(out).to_csv(TABLES / "nb1f_model_similarity_jsd.csv")

    for lang, title in LANGS:
        mat = mats[lang]
        off = mat[np.triu_indices(len(MODEL_ORDER), k=1)]
        print(f"\n{title}: mean pairwise JSD {off.mean():.4f}  "
              f"(min {off.min():.4f}, max {off.max():.4f})")
        idx = np.triu_indices(len(MODEL_ORDER), k=1)
        pairs = sorted(zip(off, [(MODEL_ORDER[a], MODEL_ORDER[b])
                                 for a, b in zip(*idx)]))
        print(f"  closest : {pairs[0][1]}  {pairs[0][0]:.4f}")
        print(f"  furthest: {pairs[-1][1]}  {pairs[-1][0]:.4f}")

    # within- vs between-origin, the number the figure is really for
    for lang, title in LANGS:
        mat = mats[lang]
        same = [mat[i, j] for i in range(6) for j in range(i + 1, 6)
                if (i < 3) == (j < 3) and (3 <= i < 5) == (3 <= j < 5)]
        diff = [mat[i, j] for i in range(6) for j in range(i + 1, 6)
                if not ((i < 3) == (j < 3) and (3 <= i < 5) == (3 <= j < 5))]
        print(f"{title}: within-origin {np.mean(same):.4f} vs "
              f"between-origin {np.mean(diff):.4f}")

    print("\nwrote nb1f_model_similarity_jsd")


if __name__ == "__main__":
    run_origin_main()
    run_origin_robustness()
    run_stance_variance()
    run_alignment_by_model()
    run_clustered_inference()
    run_model_similarity_jsd()
