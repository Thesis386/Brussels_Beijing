"""
Stage 06 — Build the master analysis dataset.

Concatenates the judged stances from the three real runs (real_r1/r2/r3) into one
tidy table, merges issue metadata (family, jurisdiction position codes, pair
polarity, developer-divergence flags) and template category, and derives the
analysis columns every downstream notebook shares:

  - origin            : model origin (US / CN / EU)
  - stance_num        : stance as float 1-5 (NaN for refusals)
  - is_refusal        : bool
  - stance_centered   : stance_num - 3 (negative = pro side, positive = con side)
  - stance_parent_scale: stance_num remapped onto the neutral parent's scale
                        (6 - s for pair_polarity == 'reversed', else unchanged;
                        NaN for refusals). Use ONLY for B<->C paired comparisons
                        (RQ3); per-issue reporting uses stance_num as stored.

All coding decisions (origin map, polarity remap, refusal handling) live here so
that notebooks cannot disagree with each other. Writes:

  evaluation/analysis/master.csv

Usage (from project root):
  .venv/bin/python evaluation/scripts/pipeline/build_master.py
"""

from pathlib import Path

import pandas as pd

RUNS_DIR = Path("evaluation/runs")
REPS = ["real_r1", "real_r2", "real_r3"]
ISSUES_METADATA_CSV = Path("evaluation/dataset_construction/issues_metadata.csv")
TEMPLATES_CSV = Path("evaluation/prompt_ingredients/templates.csv")
OUT_CSV = Path("evaluation/analysis/master.csv")

ORIGIN = {
    "claude": "US", "gpt4o": "US", "llama": "US",
    "deepseek": "CN", "qwen": "CN",
    "mistral": "EU",
}

# columns carried over from issues_metadata.csv (topic_id -> issue_id)
ISSUE_META_COLS = [
    "topic_id", "family", "analysis_layer", "category", "regulated_actor",
    "developer_stake", "refusal_risk", "variant_of", "variant_type",
    "pair_polarity", "named_country", "developer_diverges",
    "us_position", "eu_position", "china_position",
]


def main():
    # --- stances, all three reps ---
    parts = []
    for rep in REPS:
        df = pd.read_csv(RUNS_DIR / rep / "stances" / "all_stances.csv", encoding="utf-8-sig")
        df.insert(0, "rep", rep)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    expected = len(REPS) * 6 * 2 * 30 * 100  # reps x models x langs x issues x templates
    assert len(df) == expected, f"expected {expected} rows, got {len(df)}"
    dupes = df.duplicated(subset=["rep", "response_id"]).sum()
    assert dupes == 0, f"{dupes} duplicate (rep, response_id) rows"

    bad = set(df["stance"].astype(str).unique()) - set("12345") - {"refusal"}
    assert not bad, f"unexpected stance labels: {bad}"

    # --- derived stance columns ---
    df["stance"] = df["stance"].astype(str)
    df["is_refusal"] = df["stance"] == "refusal"
    df["stance_num"] = pd.to_numeric(df["stance"], errors="coerce")
    df["stance_centered"] = df["stance_num"] - 3
    df["origin"] = df["model"].map(ORIGIN)
    assert df["origin"].notna().all(), "unmapped model in origin map"

    # --- issue metadata ---
    meta = pd.read_csv(ISSUES_METADATA_CSV, encoding="utf-8-sig")[ISSUE_META_COLS]
    meta = meta.rename(columns={"topic_id": "issue_id", "category": "issue_category"})
    df = df.merge(meta, on="issue_id", how="left", validate="many_to_one")
    assert df["family"].notna().all(), "issue_id missing from issues_metadata.csv"

    # polarity remap for the two reversed named variants (C2.2, C7.1): put their
    # stance on the neutral parent's scale so B<->C differences are directional
    reversed_mask = df["pair_polarity"] == "reversed"
    df["stance_parent_scale"] = df["stance_num"].where(~reversed_mask, 6 - df["stance_num"])

    # --- template category ---
    templates = pd.read_csv(TEMPLATES_CSV, encoding="utf-8-sig")[["template_id", "category"]]
    templates = templates.rename(columns={"category": "template_category"})
    df = df.merge(templates, on="template_id", how="left", validate="many_to_one")
    assert df["template_category"].notna().all(), "template_id missing from templates.csv"

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # --- summary ---
    print(f"wrote {OUT_CSV}  ({len(df):,} rows, {df.shape[1]} cols)")
    print(f"\nrows per rep:\n{df.groupby('rep').size().to_string()}")
    print(f"\nrefusals by model x lang (all reps pooled):")
    print(df[df.is_refusal].groupby(["model", "lang"]).size().unstack(fill_value=0).to_string())
    print(f"\nmean stance (numeric only) by origin x lang, B-series:")
    b = df[(df.family == "B") & ~df.is_refusal]
    print(b.pivot_table(index="origin", columns="lang", values="stance_num").round(2).to_string())
    print(f"\nreversed-variant rows remapped: {int(reversed_mask.sum()):,} "
          f"({sorted(df.loc[reversed_mask, 'issue_id'].unique())})")


if __name__ == "__main__":
    main()
