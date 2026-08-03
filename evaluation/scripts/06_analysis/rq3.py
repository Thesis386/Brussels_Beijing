"""RQ3: how naming a jurisdiction changes stance on an otherwise matched policy.

Reads ``master.csv`` and runs paired estimates, clustered robustness checks,
per-pair results, and refusal bounds. Run ``rq3_figures.py`` afterwards.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import chi2_contingency, entropy
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

TABLES = Path("evaluation/analysis/outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)
OVERLEAF = Path("evaluation/analysis/outputs/tex")
OVERLEAF.mkdir(parents=True, exist_ok=True)

PAIR_KEY = ["model", "lang", "template_id", "rep"]
# Direction of the named actor's own position ON THE PARENT SCALE.
# aligned: naming the actor states the parent proposition, which that actor supports
#   -> the actor's position is the PRO (low) pole, so moving toward it means D < 0.
# reversed: the actor's real policy is the opposite of the parent proposition
#   -> after the 6-s remap its position is the CON (high) pole, so toward it means D > 0.
NAMED_DIR = {"aligned": -1, "reversed": +1}
ISSUE_ORDER = lambda s: (int(s[1:].split(".")[0]), s)  # noqa: E731


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# Lim & Rottger's metrics, copied verbatim from their utils.py — same as nb1b / nb2.
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


def build_pairs(master, drop_refusals=False):
    """The parent (B) / kid (C) attribution pair frame shared by every RQ3 section.

    A C-variant prompt and its B-parent prompt differ in the issue string and
    NOTHING else, so they are matched on PAIR_KEY (model, lang, template_id, rep).
    Returns one row per matched (parent, kid) pair with:
        delta               stance_c - stance_b, on the PARENT's scale
        named_dir            +-1, the named actor's own direction (see NAMED_DIR)
        shift_toward_named    delta * named_dir — comparable across pairs
        origin, lang          pinned categoricals (US/CN/EU, en/zh) so statsmodels
                               does not silently pick alphabetical reference levels
        named_is_home         categorical "home"/"rival": is the named jurisdiction
                               the answering model's own origin?
        home_j                categorical "none"/"China"/"EU"/"US": named_country
                               when named_is_home == "home", else "none" — the design
                               column for the HOME_F regressions used throughout.

    drop_refusals=True   drop any pair where EITHER side is a refusal (the estimator
                         used everywhere except the worst-case refusal bounds).
    drop_refusals=False  keep every row; a refusal's stance is NaN rather than
                         dropped, so the two sections that bound the effect of
                         refusal (nb3 Part 0b and nb3b/nb3k's exact worst-case
                         bound) can impute it themselves.
    """
    bc = master[master["family"].isin(["B", "C"])].copy()
    if drop_refusals:
        bc = bc[~bc["is_refusal"]]
    else:
        bc.loc[bc["is_refusal"], "stance_parent_scale"] = np.nan

    parents = bc[bc.family == "B"][PAIR_KEY + ["issue_id", "stance_parent_scale"]].rename(
        columns={"issue_id": "variant_of", "stance_parent_scale": "stance_b"})
    kids = bc[bc.family == "C"][
        PAIR_KEY + ["issue_id", "variant_of", "pair_polarity", "named_country",
                    "issue_category", "origin", "stance_parent_scale"]
    ].rename(columns={"issue_id": "pair", "stance_parent_scale": "stance_c"})

    p = kids.merge(parents, on=PAIR_KEY + ["variant_of"], how="inner")
    p["delta"] = p["stance_c"] - p["stance_b"]
    p["named_dir"] = p["pair_polarity"].map(NAMED_DIR)
    p["shift_toward_named"] = p["delta"] * p["named_dir"]

    p["origin"] = pd.Categorical(p["origin"], categories=["US", "CN", "EU"])
    p["lang"] = pd.Categorical(p["lang"], categories=["en", "zh"])
    p["named_is_home"] = pd.Categorical(
        np.where(p["named_country"].replace({"China": "CN"}) == p["origin"], "home", "rival"),
        categories=["rival", "home"])
    p["home_j"] = pd.Categorical(
        np.where(p["named_is_home"] == "home", p["named_country"], "none"),
        categories=["none", "China", "EU", "US"])
    return p


# =============================================================================
# nb3 — the pre-specified estimator, refusals, and the decisive home test
# =============================================================================
def run_attribution_main():
    """
    Part 1 is exactly the pre-specified plan (main.tex, "Analysis plan"): "Attribution
    effects are estimated as the per-model, per-language stance difference between each
    C-series variant and its neutral B parent, with the two reversed variants remapped
    (s' = 6 - s) before differencing."

      THE DESIGN IS PAIRED, AND EXACTLY SO. A C-variant prompt and its B-parent prompt
      differ in the issue string and NOTHING else: same 100 templates, same two languages,
      same six models, same three reps. So the difference can be taken WITHIN a
      (pair x model x language x template x rep) cell — 300 matched differences per
      pair x model x language — and template, model and decoding all cancel by
      construction. This is a far tighter estimator than a between-issue comparison, and
      it is the reason RQ3 can be answered on 13 issue pairs at all.

      THE SIGN PROBLEM, AGAIN. On the parent scale, D = C - B is positive when
      attribution moves the model toward the CON pole of the parent proposition as
      phrased. "Con as phrased" is not one substantive direction across 13 pairs, so a
      pooled mean of D is an artefact of phrasing choices, exactly as in nb2.

      WHAT RESCUES IT (and this is specific to RQ3). Every C variant states the named
      jurisdiction's OWN policy: the actor is named precisely because it holds that
      position, which is why the issue set deliberately assigns C variants no
      per-jurisdiction position codes (main.tex, "Jurisdiction position coding"). So the
      named actor's position on each variant is known by construction, and its direction
      on the PARENT scale is given by pair_polarity:

          aligned  (11 pairs)  named actor is PRO the parent proposition
          reversed ( 2 pairs)  named actor is CON the parent proposition   (C2.2, C7.1)

      Signing D by that direction gives `shift_toward_named` — how far attribution moved
      the model toward the named jurisdiction's position — which IS comparable across
      pairs, and is the direct analogue of the alignment-signing used in RQ1 and RQ2.

      THE TEST THAT MATTERS. A source-cue effect that is uniform across models is a fact
      about attribution. A source-cue effect that depends on WHO is answering is the
      geopolitical claim. Because the named country is fixed within a pair, comparing
      `shift_toward_named` ACROSS model origins within a pair is confound-free: the
      phrasing, the topic and the polarity are all common. Part 3 asks whether models
      move toward their HOME jurisdiction's named policy more than other models do.

    Refusals: a pair is dropped only if EITHER side of the matched cell is a refusal, so
    every difference is taken over a like-for-like denominator. Attribution roughly
    doubles the refusal rate, which is itself an RQ3 result, so Part 0 reports it before
    any of it is differenced away.
    """
    def fit_mixed(data, formula, label, keep=None):
        """Crossed random intercepts for template and issue-pair, as in nb1/nb2."""
        md = smf.mixedlm(
            formula, data, groups=data["grp"],
            vc_formula={"pair": "0 + C(pair)", "template": "0 + C(template_id)"},
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

    df = load_master()
    bc_all = df[df["family"].isin(["B", "C"])].copy()   # keeps refusals, for Part 0

    # ============================================================ PART 0: refusals
    section("PART 0 — DOES NAMING A JURISDICTION MAKE MODELS REFUSE?")
    print("Everything after this point conditions on a non-refusal on BOTH sides of the pair.")
    print("So the refusal effect has to be read here, before it is differenced away.\n")

    ref = (bc_all.groupby(["model", "family"])["is_refusal"].mean() * 100).unstack()
    ref.columns = ["B (neutral)", "C (named)"]
    ref["C - B"] = ref["C (named)"] - ref["B (neutral)"]
    print(ref.round(3).to_string())

    ref_lang = (bc_all.groupby(["origin", "lang", "family"])["is_refusal"].mean() * 100).unstack()
    ref_lang.columns = ["B (neutral)", "C (named)"]
    ref_lang["C - B"] = ref_lang["C (named)"] - ref_lang["B (neutral)"]
    print("\nBy origin x language:")
    print(ref_lang.round(3).to_string())

    # Which named country drives it?
    cref = bc_all[bc_all.family == "C"]
    by_named = (cref.groupby(["named_country", "origin"], observed=True)["is_refusal"].mean() * 100)
    print("\nRefusal rate (%) on NAMED variants, by who is named x who is answering:")
    print(by_named.unstack().round(3).to_string())
    ref.round(4).to_csv(TABLES / "nb3_refusals_named.csv")
    by_named.unstack().round(4).to_csv(TABLES / "nb3_refusals_by_named_country.csv")
    print("\nRefusal stays rare in absolute terms, but read the matrix carefully: the highest")
    print("cell by an order of magnitude is a Chinese model asked about CHINA'S OWN named")
    print("policy (1.4%), not about a rival's, and it is driven by Chinese-language prompts.")
    print("Attribution raises refusal, and it raises it most on the HOME cells — which are")
    print("exactly the cells that produce the home effect in Part 3b. Part 0b bounds it.")

    # ============================================================ build the paired frame
    bc = bc_all[~bc_all["is_refusal"]]
    n_possible = len(bc[bc.family == "C"])
    pairs = build_pairs(df, drop_refusals=True)
    pairs["grp"] = 1

    HOME_F = "shift_toward_named ~ C(home_j) + C(model) + C(pair) + C(template_id) + C(lang)"

    n_c_total = len(bc_all[bc_all.family == "C"])
    print(f"\nMatched pairs: {len(pairs):,} of {n_c_total:,} named responses.")
    print(f"  {n_c_total - n_possible:,} dropped for a refusal on the NAMED side, "
          f"{n_possible - len(pairs):,} for a refusal on the neutral side.")

    # ---------------------------------------------------------------- PART 0b: the bound
    # ⚠️ SUPERSEDED by nb3b Part 6. The imputation below runs in ONE direction (adverse
    # to a positive home effect), which is a genuine worst case for China and the US but
    # FAVOURABLE to the EU's negative finding. nb3b computes the exact per-target
    # directional bound; quote those numbers, not these.
    section("PART 0b — CAN SELECTIVE REFUSAL MANUFACTURE THE HOME EFFECT?")
    print("The worry is specific and it is not general. Refusal is highest exactly where the")
    print("headline result lives: Chinese models, in Chinese, on China's own named policy. If a")
    print("model refuses PRECISELY WHEN IT WOULD HAVE BEEN CRITICAL, the responses that survive")
    print("are the compliant ones, and a home effect appears from selection alone.")
    print("\nSo: refit the Part 3b home effects with every dropped cell IMPUTED BACK IN at the")
    print("stance most hostile to the finding — each refusal treated as the most anti-named-actor")
    print("answer the model could have given. This is a worst case, not an estimate: the true")
    print("value cannot be more adverse than this, whatever the refusals were hiding.\n")

    full = build_pairs(df, drop_refusals=False)

    # shift = named_dir * (stance_c - stance_b). To make the shift as ADVERSE as possible:
    #   aligned  (dir -1): shift = stance_b - stance_c  -> minimise by stance_c=5, stance_b=1
    #   reversed (dir +1): shift = stance_c - stance_b  -> minimise by stance_c=1, stance_b=5
    worst_c = np.where(full["pair_polarity"] == "aligned", 5.0, 1.0)
    worst_b = np.where(full["pair_polarity"] == "aligned", 1.0, 5.0)
    full["stance_c"] = full["stance_c"].fillna(pd.Series(worst_c, index=full.index))
    full["stance_b"] = full["stance_b"].fillna(pd.Series(worst_b, index=full.index))
    full["shift_toward_named"] = full["named_dir"] * (full["stance_c"] - full["stance_b"])

    r_w = smf.ols(HOME_F, full).fit()
    print(f"Imputed rows: {len(full) - len(pairs):,} of {len(full):,} "
          f"({100 * (len(full) - len(pairs)) / len(full):.2f}%)\n")
    rows = []
    for j in ["China", "EU", "US"]:
        k = f"C(home_j)[T.{j}]"
        rows.append({
            "named": j,
            "home_worst_case": r_w.params[k],
            "ci_lo": r_w.conf_int().loc[k, 0],
            "ci_hi": r_w.conf_int().loc[k, 1],
            "p_worst_case": r_w.pvalues[k],
        })
    bound = pd.DataFrame(rows)
    bound.round(4).to_csv(TABLES / "nb3_home_refusal_bound.csv", index=False)
    print(bound.round(3).to_string(index=False))
    print("\n(Compare with the observed estimates in Part 3b below. If the worst case keeps the")
    print("sign and rough magnitude, selective refusal cannot be generating the result, and")
    print("refusal can be reported as a finding in its own right rather than as a threat.)")

    # ============================================ PART 1: the pre-specified estimator
    section("PART 1 — THE PRE-SPECIFIED ESTIMATOR: C - B, PER MODEL, PER LANGUAGE")
    print("D on the parent scale. NEGATIVE = attribution moved the model toward the PRO pole")
    print("of the parent proposition; POSITIVE = toward the CON pole. Not yet signed toward")
    print("the named actor, so do not pool these across pairs — read them pair by pair.\n")

    cell = pairs.pivot_table(index=["pair", "named_country", "pair_polarity"],
                             columns=["model"], values="delta", aggfunc="mean")
    cell = cell.reindex(sorted(cell.index, key=lambda t: ISSUE_ORDER(t[0])))
    print("Mean D by pair x model (pooled over language):")
    print(cell.round(2).to_string())
    cell.round(4).to_csv(TABLES / "nb3_delta_by_pair_model.csv")

    cell_lang = pairs.pivot_table(index=["pair", "named_country"], columns=["model", "lang"],
                                  values="delta", aggfunc="mean")
    cell_lang = cell_lang.reindex(sorted(cell_lang.index, key=lambda t: ISSUE_ORDER(t[0])))
    cell_lang.round(4).to_csv(TABLES / "nb3_delta_by_pair_model_lang.csv")
    print("\n(Full per-model x per-language table written to nb3_delta_by_pair_model_lang.csv —")
    print(" that is the pre-specified deliverable and goes to the appendix.)")

    # Per-pair summary: is the attribution effect distinguishable from zero at all?
    # ⚠️ SUPERSEDED by nb3b Part 5: with model + lang dummies the intercept below is the
    # CLAUDE-EN cell mean, not the pair mean, and the iid p ignores the rep dependence.
    # nb3b tests the grand mean with prompt-cell clustering (verdict: 13/13, not 11/13).
    rows = []
    for p, blk in pairs.groupby("pair"):
        d = blk.copy()
        # template + model absorbed; the intercept IS the mean paired shift for this pair
        r = smf.ols("delta ~ 1 + C(model) + C(lang)", d).fit()
        rows.append({
            "pair": p,
            "parent": d.variant_of.iloc[0],
            "named": d.named_country.iloc[0],
            "polarity": d.pair_polarity.iloc[0],
            "category": d.issue_category.iloc[0],
            "mean_delta": d.delta.mean(),
            "sd_delta": d.delta.std(),
            "p": r.pvalues["Intercept"],
            "toward_named": d.shift_toward_named.mean(),
        })
    per_pair = pd.DataFrame(rows).sort_values("pair", key=lambda c: c.map(ISSUE_ORDER))
    print("\nPer-pair attribution effect (D = named - neutral, on the parent scale):")
    print(per_pair.round(3).to_string(index=False))
    per_pair.round(4).to_csv(TABLES / "nb3_per_pair_effect.csv", index=False)

    n_moved = int((per_pair.p < 0.05).sum())
    print(f"\nPairs with a mean shift distinguishable from zero: {n_moved} / {len(per_pair)}")
    print(f"Mean |D| across pairs: {per_pair.mean_delta.abs().mean():.3f} stance points")
    print("For scale: nb1 reports a 0.03-point spread between origins and 0.572 points of")
    print("within-issue disagreement between the six models. Attribution should be read")
    print("against those, not against zero.")

    # ================================ PART 2: is there an attribution effect at all?
    section("PART 2 — POOLED: DOES ATTRIBUTION MOVE MODELS TOWARD THE NAMED ACTOR?")
    print("Now signed by pair_polarity, so + = attribution moved the model TOWARD the named")
    print("jurisdiction's own documented position, and the quantity pools across pairs.\n")

    KEEP2 = ["Intercept", "C(origin)[T.CN]", "C(origin)[T.EU]", "C(lang)[T.zh]"]  # ref: US, EN
    m_pool = fit_mixed(pairs, "shift_toward_named ~ C(origin) + C(lang)",
                       "shift_toward_named ~ origin + lang   [+ = toward the NAMED actor]",
                       KEEP2)
    m_pool.round(4).to_csv(TABLES / "nb3_pooled_toward_named.csv")

    grand = pairs.shift_toward_named.mean()
    print(f"\nGrand mean shift toward the named actor: {grand:+.3f} stance points")
    print("The intercept above is that shift for the reference cell (US model, EN prompt),")
    print("net of template and pair. A negative pooled shift means naming a jurisdiction made")
    print("models LESS supportive of that jurisdiction's policy than the identical unattributed")
    print("policy — a source-cue penalty, not an endorsement effect.")

    by_named_c = pairs.groupby("named_country")["shift_toward_named"].agg(["mean", "std", "count"])
    print("\nShift toward the named actor, by WHO is named:")
    print(by_named_c.round(3).to_string())
    by_named_c.round(4).to_csv(TABLES / "nb3_toward_named_by_country.csv")

    # ================================ PART 3: the decisive test — home vs rival
    section("PART 3 — THE DECISIVE TEST: DO MODELS FAVOUR THEIR OWN JURISDICTION?")
    print("Within a pair the named country, the topic and the phrasing are all FIXED, so any")
    print("difference between origins in `shift_toward_named` is a genuine differential")
    print("response to the source cue and cannot be a phrasing artefact.\n")
    print("  H-uniform  attribution shifts every model the same way (a source-cue fact)")
    print("  H-home     models move toward THEIR OWN jurisdiction's policy more than others do")
    print("             (the geopolitical claim, and the one that would corroborate RQ1)\n")

    hv = pairs.groupby(["named_is_home", "origin"], observed=True)["shift_toward_named"].mean()
    print("Mean shift toward the named actor, by whether the namer is the model's home:")
    print(hv.unstack().round(3).to_string())

    home_cells = pairs.groupby(["named_country", "origin"], observed=True)[
        "shift_toward_named"].mean().unstack()
    print("\nSame thing, spelled out: who is named (rows) x who is answering (cols).")
    print("The DIAGONAL is the home cell. H-home predicts the diagonal sits above its column.")
    print(home_cells.round(3).to_string())
    home_cells.round(4).to_csv(TABLES / "nb3_home_matrix.csv")

    KEEP3 = ["Intercept", "C(named_is_home)[T.home]", "C(lang)[T.zh]"]
    m_home = fit_mixed(pairs, "shift_toward_named ~ C(named_is_home) + C(lang) + C(model)",
                       "HOME TEST  [+ home term = models favour their own jurisdiction]",
                       KEEP3 + ["C(model)[T.deepseek]"])
    m_home.round(4).to_csv(TABLES / "nb3_home_test.csv")

    h_coef = m_home.loc["C(named_is_home)[T.home]", "coef"]
    h_p = m_home.loc["C(named_is_home)[T.home]", "p"]
    h_lo = m_home.loc["C(named_is_home)[T.home]", "ci_lo"]
    h_hi = m_home.loc["C(named_is_home)[T.home]", "ci_hi"]
    print("\n" + "-" * 78)
    print(f"  home vs rival, POOLED = {h_coef:+.3f}  [{h_lo:+.3f}, {h_hi:+.3f}]   p = {h_p:.3g}")
    print("  A pooled home term averages three jurisdictions that need not behave alike, and")
    print("  the matrix above already shows they do not: the diagonal sits at the top of its")
    print("  column for China, and at the BOTTOM of its column for the EU. So this number is")
    print("  not yet a finding. Part 3b splits it, which is what should be reported.")
    print("-" * 78)

    # The pooled home term above averages over three origins, and that turns out to hide a
    # sign reversal. Decompose it — but NOT by fitting home-vs-rival within an origin: an
    # origin's home pairs are a DIFFERENT SET of issues from its rival pairs (China is named
    # on 6 pairs, the US on 4, the EU on 3), so within one origin `named_is_home` is a
    # deterministic function of `pair` and the contrast is not identified once the pair is
    # controlled for. It would silently become a between-issue comparison.
    #
    # The identified decomposition holds the PAIR fixed and varies who is ANSWERING: on the
    # pairs that name jurisdiction J, do J's own models move toward J's policy more than the
    # other origins' models do, on those same issues, in the same templates? That is a
    # difference-in-differences, and it is the diagonal-vs-column reading of the matrix above.
    section("PART 3b — HOME EFFECT PER JURISDICTION (pair held fixed, answerer varies)")

    # One further trap, and it bites hardest exactly where the result is most surprising.
    # The EU origin IS Mistral — a single model — so "the EU model dislikes EU-named policy"
    # and "Mistral shifts toward whoever is named less than other models do" are the same
    # coefficient if the model's overall tendency is not removed. Its whole column in the
    # matrix above is low, so this is a live confound and not a hypothetical one.
    #
    # So the three home effects are estimated TOGETHER, with model AND pair fixed effects.
    # `home_j` fires only on the (named=J, origin=home(J)) cells, so each home effect is net
    # of that model's general propensity to move toward a named actor (absorbed by C(model))
    # and of the pair's general attribution effect (absorbed by C(pair)). This is the
    # specification the finding rests on.
    r = smf.ols(HOME_F, pairs).fit()
    rows = []
    for j in ["China", "EU", "US"]:
        k = f"C(home_j)[T.{j}]"
        lo, hi = r.conf_int().loc[k]
        blk = pairs[(pairs.named_country == j)]
        rows.append({
            "named": j,
            "home_models": {"China": "CN", "EU": "EU", "US": "US"}[j],
            "n_pairs": blk.pair.nunique(),
            "home_advantage": r.params[k],
            "ci_lo": lo,
            "ci_hi": hi,
            "p": r.pvalues[k],
        })
    did = pd.DataFrame(rows)
    print("Extra shift toward J's policy shown by J's OWN models on the pairs naming J,")
    print("net of each model's general attribution tendency and each pair's general effect:\n")
    print(did.round(3).to_string(index=False))
    did.round(4).to_csv(TABLES / "nb3_home_by_jurisdiction.csv", index=False)

    print("\nThis is where the pooled home term comes apart: the three effects do not carry")
    print("the same sign, so the pooled +0.108 is an average taken over a reversal and must")
    print("not be reported on its own. State the three separately.")

    # ================================ PART 4: does attribution interact with language?
    section("PART 4 — ATTRIBUTION x LANGUAGE")
    print("RQ2 found a language effect on stance. Does the source cue behave differently when")
    print("the prompt is in Chinese — in particular when the named actor is China?\n")

    lang_named = pairs.pivot_table(index="named_country", columns="lang",
                                   values="shift_toward_named", aggfunc="mean")
    lang_named["zh - en"] = lang_named["zh"] - lang_named["en"]
    print("Shift toward the named actor, by who is named x prompt language:")
    print(lang_named.round(3).to_string())
    lang_named.round(4).to_csv(TABLES / "nb3_toward_named_by_lang.csv")

    KEEP4 = ["C(lang)[T.zh]", "C(named_is_home)[T.home]",
             "C(named_is_home)[T.home]:C(lang)[T.zh]"]
    m_int = fit_mixed(pairs, "shift_toward_named ~ C(named_is_home) * C(lang) + C(model)",
                      "HOME x LANGUAGE interaction, POOLED (do not report on its own)", KEEP4)
    m_int.round(4).to_csv(TABLES / "nb3_home_x_language.csv")
    print("\nThis pooled interaction inherits exactly the defect Part 3b exposed: it averages a")
    print("home effect that is +0.34 for China and -0.29 for the EU, so a single 'home x zh'")
    print("term is an interaction estimated over a sign reversal. It is kept only as the")
    print("stepping stone to 4b, which is the version to report.")

    # ============================ PART 4b: the three-way, split by jurisdiction
    section("PART 4b — HOME x LANGUAGE, PER JURISDICTION (the three-way)")
    print("Does prompt language change how much each jurisdiction's models favour their own")
    print("named policy? Same fixed effects as 3b (model + pair + template absorb the main")
    print("effects), now interacted with language.\n")

    r3 = smf.ols(
        "shift_toward_named ~ C(home_j) * C(lang) + C(model) * C(lang) + C(pair) * C(lang)"
        " + C(template_id)",
        pairs).fit()
    rows = []
    for j in ["China", "EU", "US"]:
        k_en = f"C(home_j)[T.{j}]"
        k_zh = f"C(home_j)[T.{j}]:C(lang)[T.zh]"
        lo, hi = r3.conf_int().loc[k_zh]
        rows.append({
            "named": j,
            "home_en": r3.params[k_en],
            "home_zh": r3.params[k_en] + r3.params[k_zh],
            "zh_minus_en": r3.params[k_zh],
            "ci_lo": lo,
            "ci_hi": hi,
            "p": r3.pvalues[k_zh],
        })
    did_lang = pd.DataFrame(rows)
    print("Home advantage in each language, and how much the language changes it:")
    print(did_lang.round(3).to_string(index=False))
    did_lang.round(4).to_csv(TABLES / "nb3_home_x_language_by_jurisdiction.csv", index=False)
    print("\nC(model)*C(lang) and C(pair)*C(lang) are in the specification so that a general")
    print("ZH shift — whether a model's own (RQ2 found a large one) or an issue's — cannot be")
    print("mistaken for a language-specific change in HOME-favouring. Only the excess shift")
    print("on a model's own home cells, in Chinese, loads on these terms.")

    # ============================ how attribution itself varies by origin x language
    section("PART 4c — ORIGIN x LANGUAGE, ON THE ATTRIBUTION EFFECT ITSELF")
    print("RQ2's question, asked of RQ3's quantity: does the ZH shift toward a named actor")
    print("depend on who is answering? (Not home-specific — this is every named pair.)\n")

    cellol = pairs.pivot_table(index="origin", columns="lang",
                               values="shift_toward_named", aggfunc="mean", observed=True)
    cellol["zh - en"] = cellol["zh"] - cellol["en"]
    print(cellol.round(3).to_string())

    KEEP4C = ["C(lang)[T.zh]", "C(origin)[T.CN]", "C(origin)[T.EU]",
              "C(origin)[T.CN]:C(lang)[T.zh]", "C(origin)[T.EU]:C(lang)[T.zh]"]
    m_ol = fit_mixed(pairs, "shift_toward_named ~ C(origin) * C(lang)",
                     "shift_toward_named ~ origin * lang   [+ = toward the NAMED actor]",
                     KEEP4C)
    m_ol.round(4).to_csv(TABLES / "nb3_origin_x_language.csv")
    cellol.round(4).to_csv(TABLES / "nb3_toward_named_by_origin_lang.csv")
    print("\nThis is the direct RQ2-style specification (same shape as nb2's M1), applied to")
    print("the attribution shift rather than to stance. Reference cell is a US model on an")
    print("English prompt, as in nb1 and nb2.")

    # ================================ PART 5: robustness
    section("PART 5 — ROBUSTNESS")

    print("(a) REVERSED PAIRS. C2.2 and C7.1 are the only two whose sign depends on the 6-s")
    print("remap, so the pooled result must not rest on them. Refit with them dropped:\n")
    aligned_only = pairs[pairs.pair_polarity == "aligned"]
    m_al = fit_mixed(aligned_only, "shift_toward_named ~ C(named_is_home) + C(lang) + C(model)",
                     "HOME TEST, 11 ALIGNED PAIRS ONLY (no remap involved)", KEEP3)
    m_al.round(4).to_csv(TABLES / "nb3_home_test_aligned_only.csv")
    a_coef = m_al.loc["C(named_is_home)[T.home]", "coef"]
    print(f"\n  home term: all 13 pairs {h_coef:+.3f}   aligned-only {a_coef:+.3f}")
    print("  If these agree, the remap is doing no work and the result is not an artefact of")
    print("  the two reversed variants.")

    # The per-jurisdiction split matters more than the pooled term, and the EU's home set is
    # where the remap is most load-bearing: of the EU's 3 named pairs, one (C7.1) is reversed.
    # So the EU's negative home effect must be shown to survive dropping it, or it is a claim
    # about the remap rather than about Mistral.
    r_al = smf.ols("shift_toward_named ~ C(home_j) + C(model) + C(pair) + C(template_id) + C(lang)",
                   aligned_only).fit()
    rows = []
    for j in ["China", "EU", "US"]:
        k = f"C(home_j)[T.{j}]"
        rows.append({
            "named": j,
            "n_pairs_aligned": aligned_only[aligned_only.named_country == j].pair.nunique(),
            "home_all13": did.set_index("named").loc[j, "home_advantage"],
            "home_aligned_only": r_al.params[k],
            "p_aligned": r_al.pvalues[k],
        })
    did_al = pd.DataFrame(rows)
    print("\n  Per-jurisdiction home effect, all 13 pairs vs the 11 aligned pairs only:")
    print(did_al.round(3).to_string(index=False))
    did_al.round(4).to_csv(TABLES / "nb3_home_by_jurisdiction_aligned_only.csv", index=False)
    print("  The EU loses a third of its named pairs here (C7.1 is reversed), so if its sign")
    print("  survives on C2.1 and C3.1 alone, the finding is about the model and not the remap.")

    print("\n(b) ORDERED LOGIT on the named responses. The paired difference assumes the 1-5")
    print("scale is evenly spaced, which the annotation data does not support. Refit the home")
    print("test on the C rows alone, assuming only that the categories are ordered.\n")

    # On C rows: endorsement of the named actor's own policy = how PRO the model is toward
    # the variant as phrased (the actor holds that position by construction). Higher = more
    # supportive of the named jurisdiction. Pair fixed effects absorb the issue content, so
    # the home term is again identified by WHO IS ANSWERING within a fixed pair.
    crows = bc[bc.family == "C"].copy()
    crows["named_is_home"] = pd.Categorical(
        np.where(crows["named_country"].replace({"China": "CN"}) == crows["origin"],
                 "home", "rival"),
        categories=["rival", "home"])
    crows["endorse_named"] = 6 - crows["stance_num"]
    crows["_y"] = pd.Categorical(crows["endorse_named"].astype(int), ordered=True)
    crows["lang"] = pd.Categorical(crows["lang"], categories=["en", "zh"])

    res = OrderedModel.from_formula(
        "_y ~ C(named_is_home) + C(lang) + C(model) + C(issue_id) + C(template_id)",
        crows, distr="logit").fit(method="bfgs", maxiter=400, disp=False)
    ordo = pd.DataFrame({"log_odds": res.params, "se": res.bse, "p": res.pvalues})
    ordo = ordo.loc[[k for k in ["C(named_is_home)[T.home]", "C(lang)[T.zh]"] if k in ordo.index]]
    ordo["odds_ratio"] = np.exp(ordo["log_odds"])
    print("--- ordered logit, C rows, endorsement of the named actor's policy ---"
          f"   (n = {len(crows):,})")
    print(ordo.round(3))
    ordo.round(4).to_csv(TABLES / "nb3_ordlogit_home.csv")
    print("\nThis is a LEVELS model (how supportive), not a DIFFERENCE model (how much the")
    print("attribution moved them), so it answers a slightly different question and is here")
    print("only to check that the paired result is not an artefact of equal-interval scoring.")

    print("\n(c) DISTRIBUTIONAL. How far does attaching a name move the stance histogram,")
    print("compared with the origin and language effects already estimated?\n")
    BINS = ["1", "2", "3", "4", "5", "refusal"]

    def stance_dist(d):
        c = d["stance"].astype(str).value_counts()
        v = np.array([c.get(k, 0) for k in BINS], dtype=float) + 1e-10
        return v / v.sum()

    rows = []
    for p, blk in bc_all[bc_all.family == "C"].groupby("issue_id"):
        par = bc_all[(bc_all.family == "B") & (bc_all.issue_id == blk.variant_of.iloc[0])]
        # swap named-vs-neutral WITHIN each origin, then average — the same estimand shape as
        # nb1b (swaps origin within language) and nb2 (swaps language within origin), so the
        # three JSDs are directly comparable.
        jsds = [calculate_jsd(stance_dist(par[par.origin == o]), stance_dist(blk[blk.origin == o]))
                for o in ["US", "CN", "EU"]]
        both = pd.concat([par.assign(k="neutral"), blk.assign(k="named")])
        rows.append({
            "pair": p,
            "named": blk.named_country.iloc[0],
            "JSD_attribution": np.mean(jsds),
            "V_attribution": cramers_v(pd.crosstab(both["stance"].astype(str), both["k"]).values),
        })
    jsd = pd.DataFrame(rows).sort_values("pair", key=lambda c: c.map(ISSUE_ORDER))
    print(jsd.round(4).to_string(index=False))
    jsd.round(4).to_csv(TABLES / "nb3_jsd_attribution.csv", index=False)
    print(f"\n  mean JSD (attribution) = {jsd.JSD_attribution.mean():.4f}")
    print("  nb1b found JSD(origin) = 0.0164; nb2 found JSD(language) = 0.0314.")
    print("  Ranking these three is the cleanest way to say how much attribution matters")
    print("  relative to the two conditions RQ1 and RQ2 already tested.")

    section("DONE — tables written to evaluation/analysis/outputs/tables/nb3_*.csv")


# =============================================================================
# nb3b — RQ3 robustness: honest SEs, per-model home effects, leave-one-pair-out,
# and an exact directional worst-case refusal bound.
# =============================================================================
def run_attribution_robustness():
    """
    WHY THIS EXISTS — four gaps in nb3, none of which touches the point estimates:

      (a) DEPENDENCE. nb3's OLS and mixed models treat 46,618 paired differences as
          independent. They are not: the three reps re-sample the same prompt, and the
          home effects are identified from only 13 issue pairs. Everything cross-pair
          is re-estimated here with CLUSTER-ROBUST SEs at the PAIR level (t on G-1 df,
          G = 13) plus a wild cluster bootstrap; within-pair claims are clustered at
          the prompt cell.
      (b) ORIGIN IS THE WRONG UNIT — nb1's own lesson, never applied to the RQ3
          headline. The +0.338 China home effect pools DeepSeek and Qwen; the US
          effect pools three models. Decomposed per model here.
      (c) FEW PAIRS. China's home effect rests on 6 pairs, the EU's on 3. A
          leave-one-pair-out refit shows whether any single pair carries a result.
      (d) THE WORST-CASE BOUND WAS MISLABELLED. nb3 Part 0b imputes every refusal in
          the direction adverse to a POSITIVE home effect. That is adverse for China
          and the US but FAVOURABLE to the EU's negative finding. Superseded here by
          an exact extremal bound per target: the home coefficient is linear in the
          imputed stances (OLS with a fixed design), so each refusal cell can be set
          at whichever end of its feasible range pushes the target coefficient
          hardest against the observed finding. No heuristic; this is the exact
          worst case over all possible imputations.

    Every number nb3/main.tex currently reports that this section re-tests is printed
    next to its honest replacement.
    """
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
        """Null-imposed wild cluster bootstrap (Rademacher) p for coefficient k."""
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
            g = np.add.reduceat(X * u[:, None], starts, axis=0)
            se = np.sqrt(cfac * ((g @ a_k) ** 2).sum())
            return beta[k] / se

        t_obs = tstat(y)
        Xr = np.delete(X, k, axis=1)
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        fit_r, u_r = Xr @ br, y - Xr @ br
        rng = np.random.default_rng(seed)
        hits = sum(
            abs(tstat(fit_r + u_r * rng.choice([-1.0, 1.0], size=G)[cl])) >= abs(t_obs)
            for _ in range(B)
        )
        return t_obs, (hits + 1) / (B + 1)

    HOME_RHS = "C(home_j) + C(model) + C(pair) + C(template_id) + C(lang)"
    HOME_F = "shift_toward_named ~ " + HOME_RHS
    JKEYS = {j: f"C(home_j)[T.{j}]" for j in ["China", "EU", "US"]}
    HOME_OF = {"China": "CN", "EU": "EU", "US": "US"}

    # ============================================================ build the paired frames
    # identical construction to nb3, kept in lockstep (via the shared build_pairs)
    df = load_master()
    bc_all = df[df["family"].isin(["B", "C"])].copy()
    bc = bc_all[~bc_all["is_refusal"]].copy()

    pairs = build_pairs(df, drop_refusals=True)
    pairs["cell"] = (pairs["model"].astype(str) + "|" + pairs["template_id"].astype(str)
                     + "|" + pairs["lang"].astype(str) + "|" + pairs["pair"].astype(str))
    print(f"matched pairs: {len(pairs):,} across {pairs['pair'].nunique()} issue pairs")

    old = pd.read_csv(TABLES / "nb3_home_by_jurisdiction.csv").set_index("named")
    old_bound = pd.read_csv(TABLES / "nb3_home_refusal_bound.csv").set_index("named")

    # ============================================================ PART 1: home test, honest SEs
    section("PART 1 — THE HOME TEST, CLUSTERED BY PAIR (G = 13) + WILD BOOTSTRAP")
    print("Old (nb3 Part 3b, iid OLS): China +0.338 p<.001, EU -0.289 p<.001, US +0.049 p=.010.")

    h_cl, _ = cluster_ols(HOME_F, pairs, "pair", "home effects  [cluster = pair]",
                          list(JKEYS.values()))

    print("\nWild cluster bootstrap (999 draws, Rademacher, null imposed):")
    Xd = dmatrix(HOME_RHS, pairs, return_type="dataframe")
    Xn = Xd.to_numpy()
    yn = pairs["shift_toward_named"].to_numpy()
    cl = pd.factorize(pairs["pair"])[0]
    wcb = {}
    for j, key in JKEYS.items():
        t_obs, p_w = wild_cluster_p(Xn, yn, cl, Xd.columns.get_loc(key))
        wcb[j] = p_w
        print(f"  {j:6s} home effect: t = {t_obs:+.3f}   p_wcb = {p_w:.4f}")

    out1 = h_cl.copy()
    out1["p_wcb"] = [wcb[j] for j in JKEYS]
    out1["old_p_iid"] = old["p"].reindex(list(JKEYS)).to_numpy()
    out1.round(4).to_csv(TABLES / "nb3b_home_clustered.csv")

    # ============================================================ PART 2: per-model home effects
    section("PART 2 — THE HOME EFFECT PER MODEL (nb1's lesson, applied to RQ3)")
    print("The pooled China effect averages DeepSeek and Qwen; the US effect averages three")
    print("models. Model-specific home dummies, same fixed effects, clustered by pair.")
    print("(home_mistral is identical to the pooled EU effect: one model.)\n")

    HOME_MODELS = [("claude", "US"), ("gpt4o", "US"), ("llama", "US"),
                   ("deepseek", "China"), ("qwen", "China"), ("mistral", "EU")]
    for m, j in HOME_MODELS:
        pairs[f"home_{m}"] = ((pairs["model"] == m) & (pairs["named_country"] == j)).astype(int)
    pm_f = ("shift_toward_named ~ " + " + ".join(f"home_{m}" for m, _ in HOME_MODELS)
            + " + C(model) + C(pair) + C(template_id) + C(lang)")
    pm_cl, _ = cluster_ols(pm_f, pairs, "pair", "per-model home effects",
                           [f"home_{m}" for m, _ in HOME_MODELS])
    pm_out = pm_cl.copy()
    pm_out.insert(0, "named", [j for _, j in HOME_MODELS])
    pm_out.round(4).to_csv(TABLES / "nb3b_home_per_model.csv")

    # ============================================================ PART 3: leave-one-pair-out
    section("PART 3 — LEAVE-ONE-PAIR-OUT (is any single pair carrying a home effect?)")
    print("Refit the Part 1 model 13 times, dropping one pair each time. China's effect is")
    print("identified from 6 pairs, the US's from 4, the EU's from 3 — so the EU rows are")
    print("the ones to watch.\n")

    loo_rows = []
    for pk in sorted(pairs["pair"].unique(), key=ISSUE_ORDER):
        sub = pairs[pairs["pair"] != pk]
        res = smf.ols(HOME_F, sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["pair"]}, use_t=True)
        loo_rows.append({
            "dropped": pk,
            "named_there": pairs.loc[pairs["pair"] == pk, "named_country"].iloc[0],
            **{f"{j}": res.params[key] for j, key in JKEYS.items()},
            **{f"p_{j}": res.pvalues[key] for j, key in JKEYS.items()},
        })
    loo = pd.DataFrame(loo_rows)
    print(loo.round(3).to_string(index=False))
    loo.round(4).to_csv(TABLES / "nb3b_home_leave_one_out.csv", index=False)
    print("\nRange of each home effect across the 13 refits:")
    for j in JKEYS:
        print(f"  {j:6s} [{loo[j].min():+.3f}, {loo[j].max():+.3f}]   "
              f"(full-sample {h_cl.loc[JKEYS[j], 'coef']:+.3f}); "
              f"max p across refits = {loo[f'p_{j}'].max():.4f}")

    # ============================================================ PART 4: home x language
    section("PART 4 — HOME x LANGUAGE PER JURISDICTION (nb3 Part 4b), CLUSTERED BY PAIR")
    print("Old (iid): China zh-en +0.055 p=.142, EU -0.153 p=.003, US +0.106 p=.005.\n")

    r3 = smf.ols(
        "shift_toward_named ~ C(home_j) * C(lang) + C(model) * C(lang) + C(pair) * C(lang)"
        " + C(template_id)",
        pairs).fit(cov_type="cluster", cov_kwds={"groups": pairs["pair"]}, use_t=True)
    rows = []
    for j, key in JKEYS.items():
        k_zh = f"{key}:C(lang)[T.zh]"
        lo, hi = r3.conf_int().loc[k_zh]
        rows.append({"named": j, "home_en": r3.params[key],
                     "zh_minus_en": r3.params[k_zh], "ci_lo": lo, "ci_hi": hi,
                     "p_cluster": r3.pvalues[k_zh]})
    hxl = pd.DataFrame(rows)
    print(hxl.round(4).to_string(index=False))
    hxl.round(4).to_csv(TABLES / "nb3b_home_x_language_clustered.csv", index=False)

    # also nb3 Part 4c (origin x language on the attribution shift itself)
    KEEP4C = ["C(lang)[T.zh]", "C(origin)[T.CN]", "C(origin)[T.EU]",
              "C(origin)[T.CN]:C(lang)[T.zh]", "C(origin)[T.EU]:C(lang)[T.zh]"]
    ol_cl, _ = cluster_ols(
        "shift_toward_named ~ C(origin) * C(lang) + C(pair) + C(template_id)",
        pairs, "pair", "origin x language on the attribution shift (nb3 Part 4c)", KEEP4C)
    ol_cl.round(4).to_csv(TABLES / "nb3b_origin_x_language_clustered.csv")

    # ============================================================ PART 5: per-pair effect
    section("PART 5 — PER-PAIR ATTRIBUTION EFFECT, CLUSTERED AT THE PROMPT CELL")
    print("nb3 Part 1 said 11 of 13 pairs moved (iid p). Two fixes: (i) nb3's regression had")
    print("model + lang dummies, so its intercept (and p) was the CLAUDE-EN cell, not the")
    print("pair's mean shift — test the grand mean directly instead; (ii) cluster at the")
    print("prompt cell, which is the within-pair dependence (the 3 reps of one prompt).\n")

    pp_rows = []
    for pk, blk in pairs.groupby("pair"):
        res = smf.ols("delta ~ 1", blk).fit(
            cov_type="cluster", cov_kwds={"groups": blk["cell"]}, use_t=True)
        pp_rows.append({"pair": pk, "named": blk["named_country"].iloc[0],
                        "mean_delta": blk["delta"].mean(),
                        "p_cluster": res.pvalues["Intercept"]})
    pp = pd.DataFrame(pp_rows).sort_values("pair", key=lambda c: c.map(ISSUE_ORDER))
    print(pp.round(4).to_string(index=False))
    pp.round(4).to_csv(TABLES / "nb3b_per_pair_clustered.csv", index=False)
    print(f"\nPairs with a shift distinguishable from zero: {int((pp.p_cluster < .05).sum())}/13"
          f"  (nb3 iid version said 11/13)")

    # ============================================================ PART 6: exact worst-case bound
    section("PART 6 — EXACT DIRECTIONAL WORST-CASE REFUSAL BOUND (supersedes nb3 Part 0b)")
    print("nb3's bound imputed every refusal against a POSITIVE home effect — adverse for")
    print("China/US but FAVOURABLE to the EU's negative finding. Here, per target: the home")
    print("coefficient is linear in the imputed stances, so each refusal cell is set at the")
    print("end of its feasible range that pushes THAT coefficient hardest against the")
    print("observed finding. Exact worst case over all imputations; refit clustered by pair.\n")

    full = build_pairs(df, drop_refusals=False)

    # feasible range of each row's shift given whichever side(s) were refused
    d = full["named_dir"].to_numpy(float)
    c_lo, c_hi = full["stance_c"].fillna(1).to_numpy(), full["stance_c"].fillna(5).to_numpy()
    b_lo, b_hi = full["stance_b"].fillna(1).to_numpy(), full["stance_b"].fillna(5).to_numpy()
    corners = [d * (c - b_) for c in (c_lo, c_hi) for b_ in (b_lo, b_hi)]
    y_lo, y_hi = np.minimum.reduce(corners), np.maximum.reduce(corners)
    imputed = full["stance_c"].isna() | full["stance_b"].isna()
    print(f"rows needing imputation: {int(imputed.sum()):,} of {len(full):,} "
          f"({100 * imputed.mean():.2f}%)")

    Xf = dmatrix(HOME_RHS, full, return_type="dataframe")
    Xfn = Xf.to_numpy()
    XtX = Xfn.T @ Xfn
    y_obs = (d * (full["stance_c"] - full["stance_b"])).to_numpy()  # NaN on imputed rows
    cl_full = full["pair"].to_numpy()

    bound_rows = []
    for j, key in JKEYS.items():
        k = Xf.columns.get_loc(key)
        e_k = np.zeros(Xfn.shape[1]); e_k[k] = 1.0
        w = Xfn @ np.linalg.solve(XtX, e_k)      # coef_k = w . y  (exact, X fixed)
        finding = h_cl.loc[key, "coef"]
        attack_down = finding > 0                # push a positive finding down, a negative one up
        y_adv = np.where((w > 0) == attack_down, y_lo, y_hi)
        y = np.where(imputed, y_adv, y_obs)
        fit = smf.ols("y ~ " + HOME_RHS, full.assign(y=y)).fit(
            cov_type="cluster", cov_kwds={"groups": cl_full}, use_t=True)
        lo, hi = fit.conf_int().loc[key]
        bound_rows.append({
            "named": j, "observed": finding, "worst_case": fit.params[key],
            "ci_lo": lo, "ci_hi": hi, "p_cluster": fit.pvalues[key],
            "old_heuristic_bound": old_bound.loc[j, "home_worst_case"],
        })
    bound = pd.DataFrame(bound_rows)
    print("\nEach row: that jurisdiction's home effect when every refusal is imputed at the")
    print("stance most hostile to THAT effect (other coefficients in the fit are not bounds):")
    print(bound.round(4).to_string(index=False))
    bound.round(4).to_csv(TABLES / "nb3b_home_worst_case_directional.csv", index=False)
    print("\nNB the old nb3 Part 0b bound was computed in one direction for all three and was")
    print("only a genuine worst case for China and the US; its EU number was favourable to")
    print("the finding. This table replaces it.")

    # ============================================================ PART 6b: ordinal, per rep
    section("PART 6b — THE ORDINAL HOME CHECK, PER REP")
    print("nb3 Part 5(b) fit an ordered logit on the C rows (OR 1.34) with iid SEs, which")
    print("counts the 3 reps as independent. Refit per rep: within one rep, prompts are")
    print("decoded once, so each rep is an independent replication. NOTE this still does not")
    print("cluster by pair, so it is a check on the SCALE assumption only — inference for the")
    print("home effect is taken from the pair-clustered linear models above.\n")

    crows = bc[bc.family == "C"].copy()
    crows["named_is_home"] = pd.Categorical(
        np.where(crows["named_country"].replace({"China": "CN"}) == crows["origin"],
                 "home", "rival"),
        categories=["rival", "home"])
    crows["endorse_named"] = 6 - crows["stance_num"]
    crows["lang"] = pd.Categorical(crows["lang"], categories=["en", "zh"])

    ord_rows = []
    for rep, blk in crows.groupby("rep"):
        d = blk.copy()
        d["_y"] = pd.Categorical(d["endorse_named"].astype(int), ordered=True)
        res = OrderedModel.from_formula(
            "_y ~ C(named_is_home) + C(lang) + C(model) + C(issue_id) + C(template_id)",
            d, distr="logit").fit(method="bfgs", maxiter=400, disp=False)
        k = "C(named_is_home)[T.home]"
        ord_rows.append({"rep": rep, "n": len(d), "log_odds": res.params[k],
                         "OR_home": np.exp(res.params[k]), "p_iid": res.pvalues[k]})
    ordr = pd.DataFrame(ord_rows)
    print(ordr.round(4).to_string(index=False))
    ordr.round(4).to_csv(TABLES / "nb3b_ordlogit_home_per_rep.csv", index=False)
    print(f"\n  OR range across the three reps: "
          f"[{ordr.OR_home.min():.2f}, {ordr.OR_home.max():.2f}]  (pooled nb3 value: 1.34)")

    # ============================================================ PART 7: verdict
    section("PART 7 — VERDICT")
    ver = pd.DataFrame([
        {"claim": f"{j} home effect ({old.loc[j, 'home_advantage']:+.3f})",
         "old_p_iid": f"{old.loc[j, 'p']:.2g}",
         "cluster_p": f"{h_cl.loc[key, 'p']:.4f}",
         "wcb_p": f"{wcb[j]:.4f}",
         "loo_range": f"[{loo[j].min():+.3f}, {loo[j].max():+.3f}]",
         "worst_case": f"{bound.set_index('named').loc[j, 'worst_case']:+.3f}"}
        for j, key in JKEYS.items()
    ])
    print(ver.to_string(index=False))
    ver.to_csv(TABLES / "nb3b_verdict.csv", index=False)

    section("DONE — tables written to evaluation/analysis/outputs/tables/nb3b_*.csv")


# =============================================================================
# nb3c — RQ3, issue by issue: the paired delta between every named variant and its
# neutral parent, for every model.
# =============================================================================
def run_per_pair_deltas():
    """
    WHY THIS EXISTS
    nb3/nb3b answer RQ3 with pooled estimates (a grand mean, a home-vs-rival term, three
    per-jurisdiction home effects). Those pool over thirteen pairs that name three different
    countries on eight different topics, and the pooling hides more than it shows: the grand
    mean is -0.011 (nothing) while individual pairs move by up to 0.87 points in opposite
    directions. The primary RQ3 deliverable should therefore be the PAIR x MODEL table, and
    the aggregates should be a short summary of it rather than a substitute for it.

    WHAT D IS
        D = stance(named variant) - stance(neutral parent), on the PARENT'S scale
            (the two reversed variants, C2.2 and C7.1, are remapped s' = 6 - s first)

        D < 0  attribution moved the model toward the PRO pole of the parent proposition
        D > 0  attribution moved it toward the CON pole

    D is signed by the parent's phrasing, so it is NOT comparable across pairs and is never
    pooled here. What IS comparable across pairs is D signed by the named actor's own
    direction (`shift_toward_named` = named_dir * D), and that column is given alongside, but
    the per-pair table is the thing to read.

    EVERY DIFFERENCE IS EXACTLY PAIRED. A C-prompt and its B-parent differ in the issue string
    and nothing else: same template, same language, same model, same repetition. So each cell
    of the table below is a mean over 600 matched differences (100 templates x 2 languages x
    3 reps), and template, model and decoding cancel within the difference by construction.

    INFERENCE. Each cell is a claim about ONE pair and ONE model, so it does not generalise
    across issues and the only dependence to remove is the repetition structure: standard
    errors are clustered at the prompt cell (template x language), i.e. the unit the three
    reps re-sample. n_clusters = 200 per cell.

    OUTPUTS
        nb3c_delta_by_pair_model.csv        the main table (D, per pair x model)
        nb3c_delta_by_pair_model_lang.csv   the same, split by prompt language
        nb3c_delta_by_pair_model_long.csv   long form with SEs, CIs and p-values per cell
        nb3c_pair_summary.csv               one row per pair: what it is, and what it did
        nb3c_home_vs_rival_by_pair.csv      per pair: the named country's own models vs the rest
        nb3c_table_main.tex                 LaTeX, the pair x model table
    """
    def section(t):
        print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")

    MODELS = ["claude", "gpt4o", "llama", "deepseek", "qwen", "mistral"]
    ORIGIN_OF = {"claude": "US", "gpt4o": "US", "llama": "US",
                 "deepseek": "CN", "qwen": "CN", "mistral": "EU"}

    df = load_master()
    meta = pd.read_csv("evaluation/dataset_construction/issues_metadata.csv", encoding="utf-8-sig")
    TOPIC = dict(zip(meta["topic_id"], meta["topic_neutral"]))

    pairs = build_pairs(df, drop_refusals=True)
    pairs["cell"] = pairs["template_id"].astype(str) + "|" + pairs["lang"].astype(str)
    pairs["is_home"] = pairs["named_is_home"] == "home"

    PAIRS = sorted(pairs["pair"].unique(), key=ISSUE_ORDER)
    print(f"{len(pairs):,} matched differences across {len(PAIRS)} pairs x {len(MODELS)} models")
    print("Each pair x model cell = 600 matched differences (100 templates x 2 langs x 3 reps).")

    # ------------------------------------------------------------------ PART 1: the main table
    section("PART 1 — D = named - neutral, ON THE PARENT'S SCALE, BY PAIR x MODEL")
    print("NEGATIVE = attribution moved the model toward the PRO pole of the PARENT proposition.")
    print("POSITIVE = toward the CON pole. D is signed by the parent's phrasing, so it is NOT")
    print("comparable across rows — read each row against its own parent proposition.")
    print("SEs clustered at the prompt cell (template x language); 200 clusters per cell.\n")

    rows = []
    for p in PAIRS:
        for m in MODELS:
            d = pairs[(pairs["pair"] == p) & (pairs["model"] == m)]
            r = smf.ols("delta ~ 1", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["cell"]}, use_t=True)
            lo, hi = r.conf_int().loc["Intercept"]
            rows.append({
                "pair": p, "parent": d["variant_of"].iloc[0],
                "named": d["named_country"].iloc[0],
                "polarity": d["pair_polarity"].iloc[0],
                "category": d["issue_category"].iloc[0],
                "model": m, "origin": ORIGIN_OF[m],
                "is_home": bool(d["is_home"].iloc[0]),
                "n": len(d),
                "mean_b": d["stance_b"].mean(), "mean_c": d["stance_c"].mean(),
                "delta": r.params["Intercept"], "se": r.bse["Intercept"],
                "ci_lo": lo, "ci_hi": hi, "p": r.pvalues["Intercept"],
                "toward_named": d["shift_toward_named"].mean(),
            })
    long = pd.DataFrame(rows)
    long.round(4).to_csv(TABLES / "nb3c_delta_by_pair_model_long.csv", index=False)

    wide = long.pivot_table(index=["pair", "named", "polarity"], columns="model", values="delta")
    wide = wide[MODELS].reindex(sorted(wide.index, key=lambda t: ISSUE_ORDER(t[0])))
    print(wide.round(2).to_string())
    wide.round(4).to_csv(TABLES / "nb3c_delta_by_pair_model.csv")

    print("\nWith significance (prompt-cell clustered; *** p<.001, ** p<.01, * p<.05):")
    disp = long.copy()
    disp["cellstr"] = disp.apply(lambda r: f"{r['delta']:+.2f}{stars(r['p'])}"
                                 + ("^" if r["is_home"] else ""), axis=1)
    dtab = disp.pivot_table(index=["pair", "named"], columns="model", values="cellstr",
                            aggfunc="first")[MODELS]
    dtab = dtab.reindex(sorted(dtab.index, key=lambda t: ISSUE_ORDER(t[0])))
    print(dtab.to_string())
    print("^ = the model's own home jurisdiction is the one named in that row.")

    # ------------------------------------------------------------------ per language
    lang_wide = long_lang = None
    rows = []
    for p in PAIRS:
        for m in MODELS:
            for lg in ["en", "zh"]:
                d = pairs[(pairs["pair"] == p) & (pairs["model"] == m) & (pairs["lang"] == lg)]
                r = smf.ols("delta ~ 1", d).fit(
                    cov_type="cluster", cov_kwds={"groups": d["template_id"]}, use_t=True)
                rows.append({"pair": p, "named": d["named_country"].iloc[0], "model": m,
                             "lang": lg, "delta": r.params["Intercept"],
                             "se": r.bse["Intercept"], "p": r.pvalues["Intercept"]})
    long_lang = pd.DataFrame(rows)
    long_lang.round(4).to_csv(TABLES / "nb3c_delta_by_pair_model_lang.csv", index=False)
    lang_wide = long_lang.pivot_table(index=["pair", "named"], columns=["model", "lang"],
                                      values="delta")
    lang_wide = lang_wide.reindex(sorted(lang_wide.index, key=lambda t: ISSUE_ORDER(t[0])))
    print("\n(Per-language version written to nb3c_delta_by_pair_model_lang.csv — appendix.)")

    # ------------------------------------------------------------------ PART 2: per-pair summary
    section("PART 2 — ONE ROW PER PAIR: WHAT THE ISSUE IS, AND WHAT ATTRIBUTION DID TO IT")

    rows = []
    for p in PAIRS:
        d = pairs[pairs["pair"] == p]
        r = smf.ols("delta ~ 1", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["cell"]}, use_t=True)
        spread = long[long["pair"] == p]["delta"]
        rows.append({
            "pair": p, "parent": d["variant_of"].iloc[0], "named": d["named_country"].iloc[0],
            "polarity": d["pair_polarity"].iloc[0], "category": d["issue_category"].iloc[0],
            "mean_b": d["stance_b"].mean(), "mean_c": d["stance_c"].mean(),
            "delta": r.params["Intercept"], "p": r.pvalues["Intercept"],
            "spread_across_models": spread.max() - spread.min(),
            "models_same_sign": int(max((spread > 0).sum(), (spread < 0).sum())),
            "toward_named": d["shift_toward_named"].mean(),
        })
    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))
    summary.round(4).to_csv(TABLES / "nb3c_pair_summary.csv", index=False)

    print("\nRead the two columns on the right: `spread_across_models` is how much the six models")
    print("disagree about the SAME attribution, and `models_same_sign` how many of the six moved")
    print("the same way. Where all six agree, attribution is a fact about the source cue; where")
    print("they split, it is a fact about the models.")
    print(f"\nMean |D| across pairs: {summary.delta.abs().mean():.3f}   "
          f"mean spread across models within a pair: {summary.spread_across_models.mean():.3f}")
    print(f"Pairs where all six models moved the same way: "
          f"{int((summary.models_same_sign == 6).sum())}/{len(summary)}")

    # ------------------------------------------------------------------ PART 3: home vs rival, per pair
    section("PART 3 — PER PAIR: DO THE NAMED COUNTRY'S OWN MODELS RESPOND DIFFERENTLY?")
    print("Within a pair the policy, the phrasing and the templates are fixed, so the ONLY thing")
    print("varying is who is answering. This is the per-issue version of the pooled home effect,")
    print("and it is where that effect should be checked before it is believed.\n")
    print("`home` = mean D for the named country's own models; `rival` = mean D for the others.")
    print("`toward_named` columns flip the sign on the two reversed pairs so + = toward the")
    print("named actor's own position. The gap is home - rival, signed toward the named actor.\n")

    rows = []
    for p in PAIRS:
        d = pairs[pairs["pair"] == p]
        nd = d["named_dir"].iloc[0]
        h, rv = d[d.is_home], d[~d.is_home]
        r = smf.ols("shift_toward_named ~ C(is_home) + C(model) + C(lang) + C(template_id)", d)
        # C(model) and C(is_home) are collinear within a pair (home is a function of model),
        # so the per-pair home contrast must be fit WITHOUT model fixed effects.
        r = smf.ols("shift_toward_named ~ C(is_home) + C(lang)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["cell"]}, use_t=True)
        k = "C(is_home)[T.True]"
        lo, hi = r.conf_int().loc[k]
        rows.append({
            "pair": p, "named": d["named_country"].iloc[0],
            "home_models": ", ".join(sorted(h["model"].unique())),
            "D_home": h["delta"].mean(), "D_rival": rv["delta"].mean(),
            "toward_named_home": h["shift_toward_named"].mean(),
            "toward_named_rival": rv["shift_toward_named"].mean(),
            "gap": r.params[k], "ci_lo": lo, "ci_hi": hi, "p": r.pvalues[k],
        })
    hv = pd.DataFrame(rows)
    print(hv.round(3).to_string(index=False))
    hv.round(4).to_csv(TABLES / "nb3c_home_vs_rival_by_pair.csv", index=False)

    print("\nNOTE ON THIS TEST. Within a pair, `is_home` is a deterministic function of which")
    print("models are answering, so the contrast cannot be adjusted for model identity — it")
    print("compares two DIFFERENT sets of models on the same issue. A gap therefore does not")
    print("distinguish 'these models respond to the label' from 'these models differ anyway'.")
    print("The pooled specification in nb3b removes the models' general attribution tendency;")
    print("the per-pair numbers here are descriptive and should be read as such.")

    npos = int((hv["gap"] > 0).sum())
    print(f"\nPairs where the named country's own models moved MORE toward it: {npos}/{len(hv)}")
    for j in ["China", "EU", "US"]:
        s = hv[hv["named"] == j]
        print(f"  named {j:6s}: gap positive on {int((s['gap'] > 0).sum())}/{len(s)} pairs "
              f"(mean {s['gap'].mean():+.3f}, range [{s['gap'].min():+.3f}, {s['gap'].max():+.3f}])")

    # ------------------------------------------------------------------ PART 3b: the cautions
    section("PART 3b — THREE CAUTIONS THE PER-PAIR VIEW FORCES, AND THE POOLED VIEW HID")

    print("(1) THE TWO REVERSED PAIRS ARE NOT A CLEAN ATTRIBUTION MANIPULATION, AND THEY DOMINATE")
    print("    EVERY POOLED MAGNITUDE. C2.2 and C7.1 differ from their parents in TWO ways at once:")
    print("    the actor is named AND the proposition is stated in the opposite direction (the US")
    print("    relies on voluntary commitments *over* legal obligations; the EU *prohibits* facial")
    print("    recognition). The 6-s remap puts their stance on the parent's scale, but the model")
    print("    was still asked about a different proposition, so D there confounds attribution with")
    print("    the reversal. They are the two largest effects in the design, and they inflate every")
    print("    pooled magnitude that includes them:\n")
    rev = summary[summary.polarity == "reversed"]
    aln = summary[summary.polarity == "aligned"]
    mag = pd.DataFrame([
        {"quantity": "mean |D| (stance points)",
         "all 13 pairs": summary.delta.abs().mean(),
         "11 aligned pairs": aln.delta.abs().mean()},
    ])
    print(mag.round(3).to_string(index=False))
    print(f"\n    The two reversed pairs carry |D| = {rev.delta.abs().mean():.3f}, against "
          f"{aln.delta.abs().mean():.3f} for the eleven aligned ones.")

    jsd_f = TABLES / "nb3_jsd_attribution.csv"
    if jsd_f.exists():
        j = pd.read_csv(jsd_f).merge(summary[["pair", "polarity"]], on="pair")
        ja, jr = j[j.polarity == "aligned"], j[j.polarity == "reversed"]
        print("\n    The distributional claim is affected far more severely. Mean Jensen-Shannon")
        print("    divergence between the named and neutral stance histograms:")
        print(f"      all 13 pairs      : {j.JSD_attribution.mean():.4f}")
        print(f"      11 aligned pairs  : {ja.JSD_attribution.mean():.4f}")
        print(f"      2 reversed pairs  : {jr.JSD_attribution.mean():.4f}  (0.430 and 0.256)")
        print(f"\n    Against JSD(origin) = 0.0164 and JSD(language) = 0.0314, attribution on the")
        print(f"    ALIGNED pairs is {ja.JSD_attribution.mean() / 0.0164:.1f}x origin and "
              f"{ja.JSD_attribution.mean() / 0.0314:.1f}x language — NOT the 5x / 2.7x the")
        print("    13-pair figure implies. Any claim that attribution is the largest condition in")
        print("    the design must be made on the aligned pairs, and is much more modest there.")

    print("\n(2) THE CHINA HOME EFFECT IS RELATIVE, NOT AN ENDORSEMENT EFFECT. Pair by pair, the")
    print("    Chinese models' OWN movement toward China's labelled policy is inconsistent in sign:")
    hc = long[(long.named == "China") & long.is_home]
    piv = hc.pivot_table(index="pair", columns="model", values="toward_named")[["deepseek", "qwen"]]
    piv = piv.reindex(sorted(piv.index, key=ISSUE_ORDER))
    print(piv.round(3).to_string())
    n_neg = int((piv.mean(axis=1) < 0).sum())
    print(f"\n    On {n_neg} of 6 China-named pairs the Chinese models became LESS favourable to")
    print("    China's policy once it was labelled as China's. Their mean shift toward the named")
    print(f"    actor is only {hc.toward_named.mean():+.3f}. What is consistent (6/6) is the GAP:")
    print("    the other four models moved AWAY from China's policy on being told it was China's,")
    print("    and by more. The home effect is therefore mostly a rival PENALTY, and should be")
    print("    reported as US- and EU-origin models discounting a China-labelled policy rather")
    print("    than as Chinese models endorsing one.")
    riv = long[(long.named == "China") & (~long.is_home)]
    print(f"    Mean shift toward China: home models {hc.toward_named.mean():+.3f}, "
          f"rivals {riv.toward_named.mean():+.3f}.")

    print("\n(3) THE US HOME EFFECT IS NOT MERELY SMALL, IT IS INCONSISTENT IN SIGN ACROSS ITS OWN")
    print("    PAIRS. The per-pair home-vs-rival gaps for US-named pairs run "
          f"{sorted(hv[hv.named == 'US']['gap'].round(3).tolist())}, i.e. two positive and two")
    print("    negative. The pooled null is not a small true effect obscured by noise; there is no")
    print("    stable direction to estimate. The EU's three gaps, by contrast, are all negative")
    print(f"    ({sorted(hv[hv.named == 'EU']['gap'].round(3).tolist())}), and China's six all")
    print(f"    positive ({sorted(hv[hv.named == 'China']['gap'].round(3).tolist())}).")

    # ------------------------------------------------------------------ PART 4: the LaTeX table
    section("PART 4 — LATEX")

    NICE = {"claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
            "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral"}
    # short gloss of each parent proposition, so a reader can sign D without leaving the table
    GLOSS = {
        "B1": "pre-deployment authorization", "B2": "legal obligations over voluntary",
        "B3": "training-data disclosure", "B4": "open weight release",
        "B5": "export controls on AI chips", "B6": "training on proprietary outputs",
        "B7": "police facial recognition", "B8": "mandatory content filtering",
        "B9": "international governance body", "B10": "G7/OECD-led standards",
        "B11": "deregulation for competitiveness",
    }
    lines = [
        r"% Auto-generated by evaluation/scripts/06_analysis/rq3.py",
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r" & & \multicolumn{3}{c}{US-origin} & \multicolumn{2}{c}{CN-origin} & EU \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
        r"Named variant (parent proposition) & Named & "
        + " & ".join(NICE[m] for m in MODELS) + r" \\",
        r"\midrule",
    ]
    for p in PAIRS:
        blk = long[long["pair"] == p].set_index("model")
        named, parent = blk["named"].iloc[0], blk["parent"].iloc[0]
        cells = [
            (rf"$\mathbf{{{blk.loc[m, 'delta']:+.2f}}}$" if blk.loc[m, "is_home"]
             else f"${blk.loc[m, 'delta']:+.2f}$")
            for m in MODELS
        ]
        rev = r"$^{\dagger}$" if blk["polarity"].iloc[0] == "reversed" else ""
        role = r"$^{\ddagger}$" if p in {"C6.1", "C9.1", "C10.1"} else ""
        label = rf"{p}{rev}{role} \emph{{({GLOSS[parent]})}}"
        lines.append(f"{label} & {named} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{The source-attribution effect, issue by issue. Each cell is $D = $ stance on "
        r"the named variant minus stance on its neutral parent, on the parent's scale, averaged "
        r"over up to 600 non-refusal matched differences (100 templates $\times$ 2 languages $\times$ 3 repetitions). "
        r"Matching holds model, template, language, and repetition fixed. A \emph{positive} value "
        r"means movement toward the \emph{con} pole of the parent proposition. Bold cells are home "
        r"cells. $^{\ddagger}$Actor/role variants specify companies, a proposal, or support as well "
        r"as a jurisdiction and are interpreted as secondary source-cue tests. $^{\dagger}$Reversed "
        r"variants state the proposition in the opposite direction and are retained as wording "
        r"checks, not attribution-effect estimates.}",
        r"\label{tab:rq3deltas}", r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (TABLES / "nb3c_table_main.tex").write_text(tex, encoding="utf-8")
    if OVERLEAF.exists():
        (OVERLEAF / "tab_rq3deltas.tex").write_text(tex, encoding="utf-8")
        print(f"wrote {OVERLEAF / 'tab_rq3deltas.tex'}")
    print(tex)

    section("DONE — tables written to evaluation/analysis/outputs/tables/nb3c_*.csv/.tex")


# =============================================================================
# nb3j — RQ3 headline table: mean stance shift on the eight clean attribution pairs.
# =============================================================================
def run_clean8_table():
    """
    One row per clean minimal-label variant, sorted most-critical first; the value is the
    mean shift_toward_named (negative = naming the jurisdiction made models more critical
    of its own policy). Emits a bare booktabs tabular so it can sit inside a wraptable in
    the main text. Same matched-difference construction as nb3b/nb3i (via build_pairs).
    """
    MINIMAL = ["C1.1", "C11.1", "C3.1", "C2.1", "C5.1", "C8.1", "C7.2", "C4.1"]
    LABEL = {
        "C1.1": "China: pre-deployment authorization",
        "C11.1": "US: deregulation for domestic AI",
        "C3.1": "EU: training-data disclosure",
        "C2.1": "EU: binding law over voluntary",
        "C5.1": "US: chip export controls",
        "C8.1": "China: content filtering",
        "C7.2": "China: police facial recognition",
        "C4.1": "China: open model weights",
    }

    df = load_master()
    p = build_pairs(df, drop_refusals=True)
    p = p[p["pair"].isin(MINIMAL)]

    en = p[p.lang == "en"].groupby("pair")["shift_toward_named"].mean()
    zh = p[p.lang == "zh"].groupby("pair")["shift_toward_named"].mean()
    order = p.groupby("pair")["shift_toward_named"].mean().loc[MINIMAL].sort_values().index  # most critical first

    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Named policy & EN & ZH \\", r"\midrule"]
    for pair in order:
        lines.append(f"{LABEL[pair]} & ${en[pair]:+.2f}$ & ${zh[pair]:+.2f}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OVERLEAF / "tab_rq3clean8.tex").write_text("\n".join(lines) + "\n")

    tab = pd.DataFrame({"EN": en, "ZH": zh}).loc[order]
    print(tab.round(2).to_string())
    print(f"\nmean|shift|  EN={en.loc[MINIMAL].abs().mean():.2f}  ZH={zh.loc[MINIMAL].abs().mean():.2f}")
    print(f"con-ward     EN={(en.loc[MINIMAL] < 0).sum()}/8  ZH={(zh.loc[MINIMAL] < 0).sum()}/8")
    print(f"wrote {OVERLEAF / 'tab_rq3clean8.tex'}")


# =============================================================================
# nb3i — RQ3 home effect across the three nested variant sets (clean 8 / aligned 11 / all 13).
# =============================================================================
def run_home_by_variant_set():
    """
    Shows that the home-versus-rival gap does not depend on which variants are included:
    stable for China and the EU across all three sets, unstable for the US (only reaches
    significance on the cleanest eight). Same estimator as nb3b (pair-clustered OLS on the
    matched shift_toward_named), just refit on each subset, using the shared build_pairs.
    """
    HOME_F = "shift_toward_named ~ C(home_j) + C(model) + C(pair) + C(template_id) + C(lang)"
    MINIMAL = {"C1.1", "C2.1", "C3.1", "C4.1", "C5.1", "C7.2", "C8.1", "C11.1"}
    REVERSED = {"C2.2", "C7.1"}

    def home_effect(pairs):
        res = smf.ols(HOME_F, pairs).fit(cov_type="cluster",
                                         cov_kwds={"groups": pairs["pair"]}, use_t=True)
        out = {}
        for j in ["China", "EU", "US"]:
            k = f"C(home_j)[T.{j}]"
            out[j] = (res.params[k], res.pvalues[k])
        return out

    df = load_master()
    pairs = build_pairs(df, drop_refusals=True)
    sets = {
        "Clean 8": pairs[pairs["pair"].isin(MINIMAL)],
        "Aligned 11": pairs[~pairs["pair"].isin(REVERSED)],
        "All 13": pairs,
    }
    results = {name: home_effect(sub) for name, sub in sets.items()}
    nvar = {name: sub["pair"].nunique() for name, sub in sets.items()}

    # console
    print(f"{'':12s} " + "  ".join(f"{n:>16s}" for n in sets))
    for j in ["China", "EU", "US"]:
        row = "  ".join(f"{results[n][j][0]:+.2f}{stars(results[n][j][1]):<3s} (p={results[n][j][1]:.3f})"
                        for n in sets)
        print(f"{j:12s} {row}")

    # LaTeX
    cols = list(sets)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\begin{tabular}{lccc}", r"\toprule",
        "Named jurisdiction & " + " & ".join(
            f"{c} ({nvar[c]} var.)" for c in cols) + r" \\",
        r"\midrule",
    ]
    disp = {"China": "China (DeepSeek, Qwen)", "EU": "EU (Mistral)",
            "US": "US (Claude, GPT-4o, Llama)"}
    for j in ["China", "EU", "US"]:
        cells = " & ".join(f"${results[c][j][0]:+.2f}$\\textsuperscript{{{stars(results[c][j][1])}}}"
                           if stars(results[c][j][1]) else f"${results[c][j][0]:+.2f}$"
                           for c in cols)
        lines.append(f"{disp[j]} & {cells} " + r"\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Home-versus-rival gap $\beta_{\text{home}}$ (Eq.~\ref{eq:rq3home}) re-estimated "
        r"on three nested variant sets: the eight clean minimal-label pairs, the eleven "
        r"proposition-fixed variants, and all thirteen. Positive means a jurisdiction's own "
        r"models were more favorable to its named policy than rival models were. The China and "
        r"EU gaps are stable and significant across every set; the US gap ranges from null to "
        r"$+0.19$ and reaches significance only on the cleanest eight, so we report no reliable "
        r"US home effect. Pair-clustered inference; \textsuperscript{*}$p<.05$, "
        r"\textsuperscript{**}$p<.01$, \textsuperscript{***}$p<.001$.}",
        r"\label{tab:rq3homesets}", r"\end{table}",
    ]
    tex = "\n".join(lines)
    (OVERLEAF / "tab_rq3homesets.tex").write_text(tex + "\n")
    print(f"\nwrote {OVERLEAF / 'tab_rq3homesets.tex'}")


# =============================================================================
# nb3k — export the two RQ3 home-effect robustness checks the appendix reports.
# Appendix robustness export.
# =============================================================================
def run_robustness_table():
    """
    Two claims in the RQ3 section rest on checks that nb3b computes but that had no
    appendix exhibit: "carried by no single variant" (leave-one-variant-out) and
    "worst-case imputation leaves the China and EU home effects intact" (the exact
    directional refusal bound). nb3b runs both on all thirteen variants only; the
    headline in the main text is stated on the clean eight, so both are re-run here
    on each set and written out as one table.

    Estimator and refusal bound follow nb3b exactly; the pair construction follows
    nb3i (both now go through the shared build_pairs). Only the variant subsets and
    the output format differ.
    """
    HOME_RHS = "C(home_j) + C(model) + C(pair) + C(template_id) + C(lang)"
    HOME_F = "shift_toward_named ~ " + HOME_RHS
    JKEYS = {j: f"C(home_j)[T.{j}]" for j in ["China", "EU", "US"]}
    MINIMAL = {"C1.1", "C2.1", "C3.1", "C4.1", "C5.1", "C7.2", "C8.1", "C11.1"}

    def home_fit(pairs):
        return smf.ols(HOME_F, pairs).fit(cov_type="cluster",
                                          cov_kwds={"groups": pairs["pair"]}, use_t=True)

    def leave_one_out(pairs):
        """Refit dropping each variant in turn; return the range and worst p per jurisdiction."""
        rows = []
        for dropped in sorted(pairs["pair"].unique()):
            res = home_fit(pairs[pairs["pair"] != dropped])
            row = {"dropped": dropped}
            for j, k in JKEYS.items():
                row[j] = res.params.get(k, np.nan)
                row[f"p_{j}"] = res.pvalues.get(k, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def worst_case(full, observed):
        """Exact directional refusal bound (nb3b Part 6): the home coefficient is linear
        in the imputed stances, so each refused cell is set at whichever end of its
        feasible 1--5 range pushes that coefficient hardest against the observed finding."""
        d = full["named_dir"].to_numpy(float)
        c_lo, c_hi = full["stance_c"].fillna(1).to_numpy(), full["stance_c"].fillna(5).to_numpy()
        b_lo, b_hi = full["stance_b"].fillna(1).to_numpy(), full["stance_b"].fillna(5).to_numpy()
        corners = [d * (c - b_) for c in (c_lo, c_hi) for b_ in (b_lo, b_hi)]
        y_lo, y_hi = np.minimum.reduce(corners), np.maximum.reduce(corners)
        imputed = full["stance_c"].isna() | full["stance_b"].isna()
        y_obs = (d * (full["stance_c"] - full["stance_b"])).to_numpy()

        X = dmatrix(HOME_RHS, full, return_type="dataframe")
        Xn = X.to_numpy()
        XtX = Xn.T @ Xn
        out = {}
        for j, key in JKEYS.items():
            e_k = np.zeros(Xn.shape[1])
            e_k[X.columns.get_loc(key)] = 1.0
            w = Xn @ np.linalg.solve(XtX, e_k)          # coef = w . y, design fixed
            attack_down = observed[j] > 0
            y_adv = np.where((w > 0) == attack_down, y_lo, y_hi)
            fit = smf.ols("y ~ " + HOME_RHS, full.assign(y=np.where(imputed, y_adv, y_obs))).fit(
                cov_type="cluster", cov_kwds={"groups": full["pair"]}, use_t=True)
            out[j] = (fit.params[key], fit.pvalues[key])
        return out, int(imputed.sum()), len(full)

    def fmt_p(p):
        return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")

    df = load_master()
    kept = build_pairs(df, drop_refusals=True)
    withref = build_pairs(df, drop_refusals=False)
    sets = {"Clean 8": MINIMAL, "All 13": set(kept["pair"].unique())}

    results = {}
    for name, keep in sets.items():
        sub = kept[kept["pair"].isin(keep)]
        full = withref[withref["pair"].isin(keep)]
        observed = {j: home_fit(sub).params[k] for j, k in JKEYS.items()}
        loo = leave_one_out(sub)
        wc, n_imp, n_tot = worst_case(full, observed)
        results[name] = {"observed": observed, "loo": loo, "wc": wc,
                         "n_imp": n_imp, "n_tot": n_tot, "n_var": sub["pair"].nunique()}
        print(f"\n=== {name} ({results[name]['n_var']} variants) ===")
        print(f"refused cells imputed in the bound: {n_imp:,} of {n_tot:,} "
              f"({100 * n_imp / n_tot:.2f}%)")
        for j in JKEYS:
            print(f"  {j:6s} observed {observed[j]:+.3f} | "
                  f"LOO [{loo[j].min():+.3f}, {loo[j].max():+.3f}] "
                  f"max p {loo[f'p_{j}'].max():.4f} | "
                  f"worst case {wc[j][0]:+.3f} (p={wc[j][1]:.4f})")
        loo.round(4).to_csv(TABLES / f"nb3k_loo_{name.split()[1]}.csv", index=False)

    disp = {"China": "China (DeepSeek, Qwen)", "EU": "EU (Mistral)",
            "US": "US (Claude, GPT-4o, Llama)"}
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Named jurisdiction & Estimate & \shortstack{Leave-one-out\\range}"
        r" & \shortstack{Largest\\leave-one-out $p$} & \shortstack{Worst-case\\refusal bound} \\",
    ]
    for name in sets:
        r = results[name]
        lines += [r"\midrule",
                  rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{name} variants}}}} \\"]
        for j in JKEYS:
            lo, hi = r["loo"][j].min(), r["loo"][j].max()
            lines.append(
                f"{disp[j]} & \\({r['observed'][j]:+.2f}\\) & "
                f"\\([{lo:+.2f}, {hi:+.2f}]\\) & "
                f"\\({fmt_p(r['loo'][f'p_{j}'].max())}\\) & "
                f"\\({r['wc'][j][0]:+.2f}\\) (\\({fmt_p(r['wc'][j][1])}\\)) \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Two robustness checks on the home-versus-rival gap regression "
        r"of Section~\ref{sec:rq3}. The leave-one-out columns refit the model dropping "
        r"each variant in turn and report the range of the coefficient across those refits "
        r"and the largest $p$-value any of them produced. The final column reports the exact "
        r"worst-case refusal bound: because the coefficient is linear in the imputed stances, "
        r"every omitted refusal is set at whichever end of the 1--5 scale pushes that "
        r"coefficient hardest against the observed estimate. The China and EU gaps survive "
        r"both checks on both variant sets. The US gap survives them on the clean eight but "
        r"not once all thirteen variants are included, which is the instability reported in "
        r"Section~\ref{sec:rq3}.}",
        r"\label{tab:rq3robust}", r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (OVERLEAF / "tab_rq3robust.tex").write_text(tex)
    print(f"\nwrote {OVERLEAF / 'tab_rq3robust.tex'}")


if __name__ == "__main__":
    run_attribution_main()
    run_attribution_robustness()
    run_per_pair_deltas()
    run_clean8_table()
    run_home_by_variant_set()
    run_robustness_table()
