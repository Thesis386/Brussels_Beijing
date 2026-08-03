"""Shared loaders for the shipped analysis data and undistributed raw responses."""

import glob
import json
from pathlib import Path

import pandas as pd

RUNS = Path("evaluation/runs")
MASTER_CSV = Path("evaluation/analysis/master.csv")
REPS = ["real_r1", "real_r2", "real_r3"]

# issue families → the three analysis subsets
FAMILY_OF_SUBSET = {"baseline": "A", "core": "B", "named": "C"}


def load_master() -> pd.DataFrame:
    """The master analysis table: all reps, stance + issue/template metadata joined."""
    return pd.read_csv(MASTER_CSV)


def load_responses(reps=REPS) -> pd.DataFrame:
    """The raw model answers (prompt + response text), for questions about content.

    Pools the given reps (adds a `rep` column). Successful responses only, deduped
    by prompt_id within each rep x model file (keep last).
    """
    rows = []
    for rep in reps:
        for f in glob.glob(str(RUNS / rep / "responses" / "*.jsonl")):
            seen = {}
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("response") and not r.get("error"):
                    r["rep"] = rep
                    seen[r["prompt_id"]] = r
            rows.extend(seen.values())
    return pd.DataFrame(rows)


def mean_table(df: pd.DataFrame, index="model", columns="lang") -> pd.DataFrame:
    """Mean stance pivot (1=pro … 5=con). Refusals are NaN in stance_num, so excluded.

    NB: only meaningful within a single issue or an alignment-corrected subset —
    never report raw means aggregated across issues with differing pro directions.
    """
    return df.pivot_table(index=index, columns=columns, values="stance_num", aggfunc="mean").round(2)


def family_subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Slice to 'baseline' (A), 'core' (B), or 'named' (C) issues."""
    return df[df["family"] == FAMILY_OF_SUBSET[name]]
