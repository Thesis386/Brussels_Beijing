"""
Stage 03 — Generate prompts (no API calls, no machine translation).

Mirrors Lim & Röttger's 03_template/a_generate_prompts.py: fill each template's `X`
placeholder with an issue phrasing, per language.
  EN row = template_en with X -> issue_en
  ZH row = template_zh with X -> issue_zh

Design decision (2026-07): only the NEUTRAL issue phrasing is sent to the models
(100 templates x 30 issues x 2 langs = 6,000 prompts). The pro/con phrasings stay in
issues.csv and are used exclusively by the stage-05 stance judge (topic_pro / topic_con
slots in the judge template). Use --framings to override for special runs.

Reads (canonical, from stage 02):
  evaluation/prompt_ingredients/templates.csv   (template_id, category, template_en, template_zh)
  evaluation/prompt_ingredients/issues.csv      (issue_id, framing, issue_en, issue_zh, ...)

Writes:
  evaluation/runs/<run_id>/prompts.csv
    prompt_id, issue_id, template_id, category, lang, prompt

prompt_id = "<issue_id>_<template_id>_<lang>"   e.g. A1_T07_zh
(neutral-only runs omit the framing segment and column entirely — everything is
neutral; a multi-framing run via --framings keeps both to avoid id collisions)

Usage (from project root):
  python evaluation/scripts/03_generate_prompts/a_generate_prompts.py --run-id real_r1
  # --templates N samples N templates (fixed seed, reproducible); omit for all 100
  # --framings neutral,pro,con to also generate pro/con prompts (not the default design)
"""

import argparse
from pathlib import Path

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

INGREDIENTS = Path("evaluation/prompt_ingredients")
TEMPLATES_CSV = INGREDIENTS / "templates.csv"
ISSUES_CSV = INGREDIENTS / "issues.csv"
RUNS_DIR = Path("evaluation/runs")

PLACEHOLDER = "X"


def md5_short(p: Path) -> str:
    """Fingerprint of an ingredient file — proves which issue/template wording a run used."""
    return hashlib.md5(p.read_bytes()).hexdigest()[:10]


def git_commit_short() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def update_run_config(run_dir: Path, stage: str, entry: dict) -> None:
    """Append this invocation's settings to <run_dir>/run_config.json (reproducibility manifest)."""
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.setdefault(stage, []).append(entry)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run folder under evaluation/runs/")
    ap.add_argument("--templates", type=int, default=None, help="Sample N templates (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducible)")
    ap.add_argument("--framings", default="neutral",
                    help="Comma-separated framings to send to models (default: neutral only; "
                         "pro/con phrasings are judge-only)")
    args = ap.parse_args()

    templates = pd.read_csv(TEMPLATES_CSV, encoding="utf-8-sig")
    issues = pd.read_csv(ISSUES_CSV, encoding="utf-8-sig")

    framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    unknown_framings = set(framings) - set(issues["framing"].unique())
    if unknown_framings:
        raise ValueError(f"unknown framings {sorted(unknown_framings)}")
    issues = issues[issues["framing"].isin(framings)].reset_index(drop=True)

    if args.templates and args.templates < len(templates):
        templates = templates.sample(n=args.templates, random_state=args.seed).reset_index(drop=True)

    neutral_only = framings == ["neutral"]

    rows = []
    for t in templates.itertuples(index=False):
        for iss in issues.itertuples(index=False):
            for lang, tmpl_col, issue_col in (("en", t.template_en, iss.issue_en),
                                              ("zh", t.template_zh, iss.issue_zh)):
                prompt = str(tmpl_col).replace(PLACEHOLDER, str(issue_col))
                row = {
                    "prompt_id":   (f"{iss.issue_id}_{t.template_id}_{lang}" if neutral_only
                                    else f"{iss.issue_id}_{iss.framing}_{t.template_id}_{lang}"),
                    "issue_id":    iss.issue_id,
                    "template_id": t.template_id,
                    "category":    t.category,
                    "lang":        lang,
                    "prompt":      prompt,
                }
                if not neutral_only:
                    row["framing"] = iss.framing
                rows.append(row)

    df = pd.DataFrame(rows)
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "prompts.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # reproducibility manifest: which templates, which ingredient file versions, which seed
    update_run_config(run_dir, "03_generate_prompts", {
        "ts":            datetime.now(timezone.utc).isoformat(),
        "git_commit":    git_commit_short(),
        "n_templates":   len(templates),
        "template_ids":  sorted(templates["template_id"].tolist()),
        "framings":      framings,
        "seed":          args.seed,
        "n_prompts":     len(df),
        "ingredients":   {"templates.csv": md5_short(TEMPLATES_CSV),
                          "issues.csv":    md5_short(ISSUES_CSV)},
    })

    print(f"{len(templates)} templates x {issues['issue_id'].nunique()} issues x "
          f"{len(framings)} framing(s) x 2 langs")
    print(f"WROTE {out}  ({len(df)} prompts: {(df.lang=='en').sum()} en, {(df.lang=='zh').sum()} zh)")


if __name__ == "__main__":
    main()
