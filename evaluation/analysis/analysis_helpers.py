"""Shared loaders for the analysis table and the raw model responses."""

from pathlib import Path

import pandas as pd

RESPONSES_DIR = Path("evaluation/analysis/responses")
MASTER_CSV = Path("evaluation/analysis/master.csv")
REPS = ["real_r1", "real_r2", "real_r3"]

# issue families → the three analysis subsets
FAMILY_OF_SUBSET = {"baseline": "A", "core": "B", "named": "C"}


def load_master() -> pd.DataFrame:
    """The master analysis table: all reps, stance + issue/template metadata joined."""
    return pd.read_csv(MASTER_CSV)


def load_responses(reps=REPS) -> pd.DataFrame:
    """The raw model answers (prompt + response text), for questions about content.

    Pools the given reps. Non-empty, non-error responses only; join to master on
    (rep, response_id).
    """
    df = pd.concat(
        [pd.read_parquet(RESPONSES_DIR / f"responses_{rep}.parquet") for rep in reps],
        ignore_index=True,
    )
    return df[df["response"].fillna("").ne("") & ~df["error"]].reset_index(drop=True)


def mean_table(df: pd.DataFrame, index="model", columns="lang") -> pd.DataFrame:
    """Mean stance pivot (1=pro … 5=con). Refusals are NaN in stance_num, so excluded.

    NB: only meaningful within a single issue or an alignment-corrected subset —
    never report raw means aggregated across issues with differing pro directions.
    """
    return df.pivot_table(index=index, columns=columns, values="stance_num", aggfunc="mean").round(2)


def family_subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Slice to 'baseline' (A), 'core' (B), or 'named' (C) issues."""
    return df[df["family"] == FAMILY_OF_SUBSET[name]]
