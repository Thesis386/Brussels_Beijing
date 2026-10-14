# From Brussels to Beijing: How Model Origin and Prompt Language Shape LLM Stances on AI Governance

This repository contains the final replication materials for a study of how model
origin, prompt language, and jurisdiction attribution shape LLM stances on AI
governance. The analysis covers 30 issues, 100 writing-assistance templates, two
languages, six models, and three repetitions (108,000 judged responses).

## Contents

```text
evaluation/
  analysis/               # master.csv and shared analysis utilities
  dataset_construction/   # issue grounding and RQ4 evidence records
  prompt_ingredients/     # final English and Chinese issues/templates
  scripts/06_analysis/    # analysis and thesis-figure scripts
```

`evaluation/analysis/master.csv` is the analysis dataset. It contains one row per
judged response, joined to the metadata and derived variables used by the scripts.

The issue-selection record is in `evaluation/dataset_construction/`, including source
quotes, documented jurisdiction positions, excluded candidates, and the two RQ4
developer/government evidence tables.

## Response data

The full text of all 108,000 responses is attached to the
[`data-v1` release](https://github.com/Thesis386/Brussels_Beijing/releases/tag/data-v1)
as one Parquet file per repetition (~85 MB each). Download them into
`evaluation/analysis/responses/`:

```bash
gh release download data-v1 -R Thesis386/Brussels_Beijing -D evaluation/analysis/responses
```

Each file has one row per (model, prompt), with columns `rep`, `response_id`,
`prompt_id`, `model`, `issue_id`, `template_id`, `lang`, `framing`, `prompt`,
`response`, `finish_reason`, `error`, and `ts`. Where a request was retried, only the
final attempt is kept, which is the response that was judged. Join to stances and
metadata on `(rep, response_id)`:

```python
import pandas as pd
resp = pd.read_parquet("evaluation/analysis/responses/responses_real_r1.parquet")
master = pd.read_csv("evaluation/analysis/master.csv")
df = resp.merge(master, on=["rep", "response_id"], suffixes=("", "_m"))
```

| Model key | OpenRouter ID | Origin |
|-----------|---------------|--------|
| `claude` | `anthropic/claude-sonnet-4-6` | US |
| `gpt4o` | `openai/gpt-4o` | US |
| `llama` | `meta-llama/llama-4-maverick` | US |
| `deepseek` | `deepseek/deepseek-chat-v3.1` | CN |
| `qwen` | `qwen/qwen3-235b-a22b-2507` | CN |
| `mistral` | `mistralai/mistral-large-2512` | EU |

Responses were generated in July 2026 at temperature 1.0 with `max_tokens` 4096 and
no system prompt, and judged by `google/gemini-2.5-flash` at temperature 0.

## Reproducing the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python evaluation/scripts/06_analysis/descriptives.py
python evaluation/scripts/06_analysis/rq1.py
python evaluation/scripts/06_analysis/rq2_language_main.py
python evaluation/scripts/06_analysis/rq2_limrottger_replication.py
python evaluation/scripts/06_analysis/rq2_toward_china_figure.py  # after rq1.py
python evaluation/scripts/06_analysis/rq3.py
python evaluation/scripts/06_analysis/rq3_figures.py              # after rq3.py
python evaluation/scripts/06_analysis/rq4.py
```

RQ5 has two entry points: `rq5_framing.py` (keyword screens and the ordered framing
pipeline) and `rq5_text.py` (embeddings, stylometry, or the B7 repertoire audit).
Both need the response files above.

Scripts write regenerable tables, figures, and LaTeX exports to
`evaluation/analysis/outputs/`, which is intentionally excluded from version control.

Stance is defined relative to each issue's phrasing (1 = only pro, 3 = neutral,
5 = only con), so raw scores are not averaged across issues. Refusals are excluded
from stance means and reported separately.

## Acknowledgements

The template corpus and writing-assistance methodology build on
[IssueBench](https://github.com/paul-rottger/issuebench) (Röttger et al., 2025).
The bilingual design and stance rubric follow [Lim and Röttger
(2025)](https://github.com/yingslim/political_bias).

## License

See [LICENSE](LICENSE). Templates derived from IssueBench remain under CC-BY-4.0;
quoted source documents remain under their original licenses.
