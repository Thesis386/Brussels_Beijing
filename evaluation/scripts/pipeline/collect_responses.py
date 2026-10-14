"""
Stage 04 — Collect model responses (inference).

Mirrors Lim & Röttger's 04_model_response/generate_model_response.py: threaded OpenRouter
calls with retry + exponential backoff, one model at a time, resumable. We write one
append-only JSONL per model (safer for concurrent 6-model runs and clean resume than
Lim's rewrite-the-whole-table-per-row approach; the methodology is identical otherwise).

Each writing prompt is sent as-is (no system prompt).

Design decision (2026-07): the real eval samples at TEMPERATURE 1 and is repeated 3x
(5x if budget allows), one run directory per repetition (real_r1, real_r2, ...). Each
repetition = 6,000 neutral prompts x 6 models = 36,000 responses. Note: Lim & Röttger's
collection code sets NO temperature (API default, i.e. ~1) and no max_tokens, single
pass — so temp 1 matches their effective setup; our additions are the explicit
temperature, the 4096 cap, and the repetitions. Document in the thesis methods.

Reads:
  <run_dir>/prompts.csv   (from stage 03; columns: prompt_id, issue_id, framing, template_id, lang, prompt)

Writes:
  <run_dir>/responses/<model>.jsonl   — one JSON object per line:
      {prompt_id, issue_id, framing, template_id, lang, model, prompt, response, finish_reason, error, ts}

Usage (from project root):
  # real eval, one repetition (repeat with --run-id real_r2, real_r3, ...):
  python evaluation/scripts/pipeline/generate_prompts.py --run-id real_r1
  caffeinate -is .venv/bin/python evaluation/scripts/pipeline/collect_responses.py \
      --run-id real_r1 --parallel-models
  # prompts.csv must already exist (stage 03) — this script never generates it.
  # Before any API call it prints the full plan (models, counts, temp, cost surface)
  # and asks for confirmation; --yes skips the prompt for unattended runs.
  # --parallel-models runs all models concurrently (each writes its own file);
  # omit it to run models one at a time. caffeinate keeps the Mac awake overnight.
  # Resume is automatic — rerun the same command to fill only missing/errored rows.
  # Smoke test: --limit 20
  # After a completed run: git add evaluation/runs/<run_id> && git commit && git push
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── the 6 evaluation models (OpenRouter ids) ────────────────────────────────
MODELS = {
    "claude":   "anthropic/claude-sonnet-4-6",
    "llama":    "meta-llama/llama-4-maverick",
    "mistral":  "mistralai/mistral-large-2512",
    "deepseek": "deepseek/deepseek-chat-v3.1",
    "qwen":     "qwen/qwen3-235b-a22b-2507",
    "gpt4o":    "openai/gpt-4o",
}

RUNS_DIR    = Path("evaluation/runs")
MAX_TOKENS  = 4096     # Raised from 2048 on 2026-07-10 after partial rep-1 data showed
                       # language-skewed truncation (claude 31%/mistral 40% truncated, 3-4x more
                       # often in ZH — a confound with both IVs). Truncated records were purged
                       # and re-collected at 4096; no stance judging had occurred.
MAX_RETRIES = 5       # unattended overnight runs: ride out transient 429s/5xx
BACKOFF_CAP = 30      # seconds; sleep min(2**attempt, cap) between retries
file_lock   = threading.Lock()


def git_commit_short() -> str:
    """Short git commit hash for the run manifest (blank if unavailable)."""
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


def call_model(client: OpenAI, model_id: str, prompt: str, temperature: float) -> tuple[str, str]:
    """Send the writing prompt (no system message). Returns (text, finish_reason)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            choice = resp.choices[0]
            return (choice.message.content or ""), (choice.finish_reason or "")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, BACKOFF_CAP))   # capped exponential backoff
            else:
                return f"ERROR: {e}", "error"
    return "ERROR: unknown", "error"


def already_done(out_path: Path) -> set[str]:
    """prompt_ids already collected with a non-error response (for resume)."""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("response") and not str(r["response"]).startswith("ERROR:"):
                    done.add(r["prompt_id"])
    return done


def append_record(out_path: Path, record: dict) -> None:
    with file_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_model(client: OpenAI, model_key: str, prompts: pd.DataFrame, run_dir: Path,
              workers: int, temperature: float) -> None:
    model_id = MODELS[model_key]
    out_dir  = run_dir / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.jsonl"

    done = already_done(out_path)
    todo = prompts[~prompts["prompt_id"].isin(done)]
    print(f"[{model_key}] {len(done)} done, {len(todo)} to collect  ->  {out_path}")

    def work(row):
        text, finish = call_model(client, model_id, row["prompt"], temperature)
        append_record(out_path, {
            "prompt_id":     row["prompt_id"],
            "issue_id":      row["issue_id"],
            # neutral-only prompts.csv has no framing column; stages 05/06 still key on it
            "framing":       row.get("framing", "neutral"),
            "template_id":   row["template_id"],
            "lang":          row["lang"],
            "model":         model_key,
            "prompt":        row["prompt"],
            "response":      text,
            "finish_reason": finish,
            "error":         text.startswith("ERROR:"),
            "ts":            datetime.now(timezone.utc).isoformat(),
        })
        return row["prompt_id"], text.startswith("ERROR:")

    n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, r) for _, r in todo.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            _, err = fut.result()
            n_err += err
            if i % 50 == 0 or i == len(futures):
                print(f"  [{model_key}] {i}/{len(futures)} ({n_err} errors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run folder name under evaluation/runs/")
    ap.add_argument("--prompts", default=None, help="prompts.csv (default: <run_dir>/prompts.csv)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="Comma-separated model keys (default: all 6)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (real-eval design: 1.0)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N prompts (smoke test)")
    ap.add_argument("--parallel-models", action="store_true",
                    help="Collect all models concurrently (each model writes its own file)")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt (unattended runs)")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_id
    prompts_path = Path(args.prompts) if args.prompts else run_dir / "prompts.csv"
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path} (run stage 03 first)")

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model keys {unknown}; choose from {list(MODELS)}")

    prompts = pd.read_csv(prompts_path, encoding="utf-8-sig")
    if args.limit:
        prompts = prompts.head(args.limit)

    # ── preflight plan: show exactly what will happen before any API spend ──
    todo_by_model = {}
    for k in model_keys:
        done = already_done(run_dir / "responses" / f"{k}.jsonl")
        todo_by_model[k] = len(prompts[~prompts["prompt_id"].isin(done)])
    total_calls = sum(todo_by_model.values())

    print(f"\n=== PLAN for run '{args.run_id}' ===")
    framings = sorted(prompts["framing"].unique()) if "framing" in prompts.columns else ["neutral"]
    print(f"prompts file : {prompts_path}  ({len(prompts)} prompts: "
          f"{(prompts.lang=='en').sum()} en, {(prompts.lang=='zh').sum()} zh; "
          f"framings: {framings})")
    print(f"temperature  : {args.temperature}   max_tokens: {MAX_TOKENS}   no system prompt")
    print(f"execution    : {'ALL MODELS IN PARALLEL' if args.parallel_models else 'one model at a time'}, "
          f"{args.workers} workers per model")
    print("models (API calls to make, after resume-dedup):")
    for k in model_keys:
        print(f"  {k:9s} {MODELS[k]:40s} {todo_by_model[k]:>6} to collect")
    print(f"TOTAL API calls this invocation: {total_calls}")
    if total_calls == 0:
        print("Nothing to collect — all responses already present.")
        return
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted — nothing sent.")
            return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not found in .env")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # reproducibility manifest: record exactly what this invocation collected
    update_run_config(run_dir, "04_collect_responses", {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
        "models":     {k: MODELS[k] for k in model_keys},
        "n_prompts":  len(prompts),
        "limit":      args.limit,
        "workers":    args.workers,
        "max_tokens": MAX_TOKENS,
        "temperature": args.temperature,
        "parallel_models": args.parallel_models,
    })

    if args.parallel_models:
        with ThreadPoolExecutor(max_workers=len(model_keys)) as model_pool:
            futures = {model_pool.submit(run_model, client, k, prompts, run_dir,
                                         args.workers, args.temperature): k
                       for k in model_keys}
            for fut in as_completed(futures):
                fut.result()   # surface any exception with its traceback
    else:
        for model_key in model_keys:
            run_model(client, model_key, prompts, run_dir, args.workers, args.temperature)

    # ── completeness summary ──
    print(f"\n=== SUMMARY for run '{args.run_id}' ===")
    all_ok = True
    for k in model_keys:
        good = already_done(run_dir / "responses" / f"{k}.jsonl")
        n_good = len(prompts[prompts["prompt_id"].isin(good)])
        ok = n_good == len(prompts)
        all_ok &= ok
        print(f"  {'OK   ' if ok else 'SHORT'} {k}: {n_good}/{len(prompts)} good responses")
    if not all_ok:
        print("Some models are short — rerun the same command to resume (only missing rows are sent).")
    print("Responses in", run_dir / "responses")


if __name__ == "__main__":
    main()
