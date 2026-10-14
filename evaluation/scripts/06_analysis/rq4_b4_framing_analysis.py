"""Exploratory close reading of the manually reviewed B4 sample."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[3]
INPUT_XLSX = ROOT / "evaluation/analysis/manual_coding/rq4_B4_framing_review_FINAL.xlsx"
INPUT_SHEET = "Responses FINAL"
OUTPUT_DIR = ROOT / "evaluation/analysis/outputs/rq4_b4_close_reading"

# The workbook was reviewed under these blinded labels. Keeping the six-code key here
# makes the final workbook the only data dependency and avoids relying on the temporary
# browser app used during coding.
BLIND_MODEL_KEY = {
    "M1": ("GPT-4o", "OpenAI"),
    "M2": ("Mistral Large", "Mistral"),
    "M3": ("DeepSeek V3.1", "DeepSeek"),
    "M4": ("Claude Sonnet 4.6", "Anthropic"),
    "M5": ("Qwen3-235B", "Alibaba"),
    "M6": ("Llama 4 Maverick", "Meta"),
}

MODEL_ORDER = ["Claude", "GPT-4o", "Llama", "Mistral", "DeepSeek", "Qwen"]
MODEL_DISPLAY = {
    "Claude Sonnet 4.6": "Claude",
    "GPT-4o": "GPT-4o",
    "Llama 4 Maverick": "Llama",
    "Mistral Large": "Mistral",
    "DeepSeek V3.1": "DeepSeek",
    "Qwen3-235B": "Qwen",
}

FRAMES = [
    "Safety and dangerous capabilities",
    "Irreversibility and loss of control",
    "Open research ecosystem",
    "Competition and deconcentration",
    "Sovereignty and technological autonomy",
    "Equitable and Global South access",
    "Independent auditing and accountability",
    "Application proliferation",
    "Commercial incentives and weak moats",
]

MECHANISMS = [
    "Unrestricted open release",
    "Open release with ex ante testing",
    "Open release with licenses or documentation",
    "Open release with safeguards (unspecified)",
    "Capability-threshold release",
    "Vetted or gated weight access",
    "API or structured access only",
    "Full restriction or non-release",
    "Government or third-party approval",
    "No clear recommendation",
    "Other",
]

FRAME_PATTERNS = {
    "Safety and dangerous capabilities": re.compile(
        r"dangerous capabil|dual.use|misuse|malicious|catastroph|cyber|bioweapon|CBRN|"
        r"危险能力|双重用途|滥用|恶意|灾难|网络攻击|生物武器", re.I
    ),
    "Irreversibility and loss of control": re.compile(
        r"cannot be (?:recalled|retracted)|once released|irreversib|remove safeguards|"
        r"无法撤回|一旦发布|不可逆|移除.{0,12}安全", re.I
    ),
    "Open research ecosystem": re.compile(
        r"open research|scientific research|reproducib|collaborat|build (?:on|upon)|"
        r"开放研究|科学研究|可复现|协作|在.{0,12}基础上", re.I
    ),
    "Competition and deconcentration": re.compile(
        r"monopol|oligopol|tech giants|market concentration|smaller firms|startups|"
        r"垄断|寡头|科技巨头|市场集中|中小企业|初创", re.I
    ),
    "Sovereignty and technological autonomy": re.compile(
        r"sovereign|strategic autonomy|foreign dependence|dependence on (?:US|China|Western)|"
        r"主权|战略自主|外国依赖|依赖.{0,10}(?:美国|中国|西方)", re.I
    ),
    "Equitable and Global South access": re.compile(
        r"Global South|developing countr|low.resource|digital divide|underserved|"
        r"全球南方|发展中国家|低资源|数字鸿沟|欠发达", re.I
    ),
    "Independent auditing and accountability": re.compile(
        r"independent audit|external audit|public scrutiny|accountab|red.team|inspect|"
        r"独立审计|外部审计|公众监督|问责|红队|检查", re.I
    ),
    "Application proliferation": re.compile(
        r"downstream application|derivative model|locali[sz]ation|across (?:health|education|agriculture)|"
        r"下游应用|衍生模型|本地化|医疗.{0,20}教育|教育.{0,20}农业", re.I
    ),
    "Commercial incentives and weak moats": re.compile(
        r"business model|commercial incentive|competitive advantage|API revenue|R&D investment|"
        r"intellectual property|商业模式|商业激励|竞争优势|API收入|研发投资|知识产权", re.I
    ),
}

REFERENCE_PATTERN = re.compile(
    r"Responsible Scaling Policy|Frontier Safety Framework|Preparedness Framework|EU AI Act|"
    r"\bRAIL\b|NIST|RAND|Stanford|MIT|CSET|METR|Anthropic|OpenAI|Meta|Mistral|DeepSeek|Qwen|"
    r"负责任扩展政策|前沿安全框架|欧盟人工智能法案|斯坦福|麻省理工|阿里巴巴|深度求索",
    re.I,
)

FRAME_PRIORITY = [
    "Sovereignty and technological autonomy",
    "Equitable and Global South access",
    "Irreversibility and loss of control",
    "Application proliferation",
    "Competition and deconcentration",
    "Independent auditing and accountability",
    "Commercial incentives and weak moats",
    "Safety and dangerous capabilities",
    "Open research ecosystem",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_labels(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def safe_rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_p(value: float) -> str:
    return "<.001" if value < 0.001 else f"={value:.3f}".replace("=0.", "=.")


def excerpt_for(text: str, pattern: re.Pattern[str] | None, max_chars: int = 620) -> str:
    text = str(text)
    paragraphs = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n+", text) if item.strip()]
    substantive = [
        paragraph for paragraph in paragraphs
        if len(paragraph) >= 160
        and not re.match(r"^(?:I hope|Suggested audience|Conclusion:?$|Summary:?$)", paragraph, re.I)
    ]
    chosen = ""
    if pattern:
        chosen = next((paragraph for paragraph in substantive if pattern.search(paragraph)), "")
    if not chosen:
        chosen = next(iter(substantive), "")
    if not chosen:
        chosen = next((paragraph for paragraph in reversed(paragraphs) if len(paragraph) >= 80), paragraphs[0] if paragraphs else "")
    return chosen if len(chosen) <= max_chars else chosen[: max_chars - 1] + "…"


def load_reviewed() -> tuple[pd.DataFrame, dict]:
    reviewed = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET, engine="openpyxl")
    if reviewed["ID"].duplicated().any():
        raise ValueError("Responses FINAL contains duplicate IDs")
    unknown_blind_codes = sorted(set(reviewed["Model (blind)"].dropna()) - set(BLIND_MODEL_KEY))
    if unknown_blind_codes:
        raise ValueError(f"Unknown blind model codes: {unknown_blind_codes}")
    merged = reviewed.copy()
    merged["model_full"] = merged["Model (blind)"].map(lambda code: BLIND_MODEL_KEY[code][0])
    merged["developer"] = merged["Model (blind)"].map(lambda code: BLIND_MODEL_KEY[code][1])
    sample_number = merged["ID"].astype(str).str.extract(r"^S(\d+)-", expand=False)
    if sample_number.isna().any():
        raise ValueError("Every response ID must begin with an S<number>- sample prefix")
    merged["source_sample"] = "Sample " + sample_number
    merged["model"] = merged["model_full"].map(MODEL_DISPLAY)
    if merged["model"].isna().any():
        raise ValueError("Unrecognized model name in model key")
    merged["language"] = merged["Lang"].map({"en": "English", "zh": "Chinese"})
    merged["specificity"] = pd.to_numeric(merged["Specificity — FINAL (1–3)"], errors="raise").astype(int)
    if not merged["specificity"].isin([1, 2, 3]).all():
        raise ValueError("Specificity must be coded 1, 2, or 3")
    merged["mechanism"] = merged["Mechanism — FINAL"].astype(str)
    unknown_mechanisms = sorted(set(merged["mechanism"]) - set(MECHANISMS))
    if unknown_mechanisms:
        raise ValueError(f"Unknown mechanism labels: {unknown_mechanisms}")
    merged["frame_set"] = merged["Frames — FINAL"].map(split_labels)
    unknown_frames = sorted(set().union(*merged["frame_set"]) - set(FRAMES))
    if unknown_frames:
        raise ValueError(f"Unknown frame labels: {unknown_frames}")
    for frame in FRAMES:
        merged[f"frame__{frame}"] = merged["frame_set"].map(lambda labels, f=frame: int(f in labels))
    merged["has_organization"] = merged["Organizations — FINAL"].fillna("").astype(str).str.strip().ne("").astype(int)
    merged["has_named_policy"] = merged["Named policies — FINAL"].fillna("").astype(str).str.strip().ne("").astype(int)
    inst = merged["Institutional refs — FINAL"].fillna("").astype(str)
    merged["own_developer_named"] = inst.str.contains(r"(?:^|;)\s*ownDeveloperNamed(?:;|$)", regex=True).astype(int)
    merged["named_research_or_evidence"] = inst.str.contains(r"namedResearchOrEvidence").astype(int)
    merged["response_chars"] = merged["Response"].fillna("").astype(str).str.len()
    merged["log_response_chars"] = np.log1p(merged["response_chars"])
    merged["log_chars_z_language"] = merged.groupby("language")["log_response_chars"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    merged["response_words_en"] = merged["Response"].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    merged["block"] = merged["source_sample"].astype(str) + "|" + merged["Template"].astype(str) + "|" + merged["Lang"].astype(str)
    manifest = {
        "source_workbook": str(INPUT_XLSX.relative_to(ROOT)),
        "source_sheet": INPUT_SHEET,
        "source_sha256": sha256(INPUT_XLSX),
        "blind_model_key": {
            code: {"model": model, "developer": developer}
            for code, (model, developer) in BLIND_MODEL_KEY.items()
        },
        "rows": int(len(merged)),
        "unique_ids": int(merged["ID"].nunique()),
        "languages": merged["language"].value_counts().to_dict(),
        "analysis_status": "Exploratory, manually reviewed close reading",
        "specificity_note": "Specificity is a textual characteristic partly constituted by named references; adjusted estimates are diagnostic, not causal.",
    }
    return merged, manifest


def sample_composition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "language"], observed=True)
        .agg(n=("ID", "size"), templates=("Template", "nunique"), mean_chars=("response_chars", "mean"))
        .reset_index()
        .sort_values(["model", "language"], key=lambda col: col.map({m: i for i, m in enumerate(MODEL_ORDER)}) if col.name == "model" else col)
    )


def frame_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_model = []
    rows_lang = []
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]
        for frame in FRAMES:
            column = f"frame__{frame}"
            rows_model.append({"model": model, "frame": frame, "n": len(model_df), "count": int(model_df[column].sum()), "proportion": safe_rate(model_df[column])})
            for language in ["English", "Chinese"]:
                subset = model_df[model_df["language"] == language]
                rows_lang.append({"model": model, "language": language, "frame": frame, "n": len(subset), "count": int(subset[column].sum()), "proportion": safe_rate(subset[column])})
    return pd.DataFrame(rows_model), pd.DataFrame(rows_lang)


def mechanism_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_model = []
    rows_lang = []
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]
        for mechanism in MECHANISMS:
            count = int((model_df["mechanism"] == mechanism).sum())
            rows_model.append({"model": model, "mechanism": mechanism, "n": len(model_df), "count": count, "proportion": count / len(model_df)})
            for language in ["English", "Chinese"]:
                subset = model_df[model_df["language"] == language]
                count_lang = int((subset["mechanism"] == mechanism).sum())
                rows_lang.append({"model": model, "language": language, "mechanism": mechanism, "n": len(subset), "count": count_lang, "proportion": count_lang / len(subset) if len(subset) else np.nan})
    return pd.DataFrame(rows_model), pd.DataFrame(rows_lang)


def specificity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_model = []
    rows_lang = []
    for model in MODEL_ORDER:
        subset = df[df["model"] == model]
        counts = subset["specificity"].value_counts()
        rows_model.append({
            "model": model, "n": len(subset), "mean_specificity": subset["specificity"].mean(),
            "score_1_n": int(counts.get(1, 0)), "score_2_n": int(counts.get(2, 0)), "score_3_n": int(counts.get(3, 0)),
            "score_3_proportion": float((subset["specificity"] == 3).mean()),
        })
        for language in ["English", "Chinese"]:
            lang_df = subset[subset["language"] == language]
            lang_counts = lang_df["specificity"].value_counts()
            rows_lang.append({
                "model": model, "language": language, "n": len(lang_df), "mean_specificity": lang_df["specificity"].mean(),
                "score_1_n": int(lang_counts.get(1, 0)), "score_2_n": int(lang_counts.get(2, 0)), "score_3_n": int(lang_counts.get(3, 0)),
                "score_3_proportion": float((lang_df["specificity"] == 3).mean()) if len(lang_df) else np.nan,
            })
    return pd.DataFrame(rows_model), pd.DataFrame(rows_lang)


def specificity_length_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate standardized model means after descriptive response-length adjustment."""
    categorical = ["model", "language", "Genre", "source_sample"]
    numeric = ["log_chars_z_language"]
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)]
    )
    fitted = Pipeline([("features", transformer), ("ols", LinearRegression())])
    fitted.fit(df[categorical + numeric], df["specificity"])
    raw = df.groupby("model")["specificity"].mean().to_dict()
    rows = []
    for model_name in MODEL_ORDER:
        counterfactual = df[categorical + numeric].copy()
        counterfactual["model"] = model_name
        rows.append({
            "model": model_name,
            "raw_mean_specificity": raw[model_name],
            "length_adjusted_standardized_mean": float(fitted.predict(counterfactual).mean()),
        })
    return pd.DataFrame(rows)


def reference_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        subset = df[df["model"] == model]
        rows.append({
            "model": model,
            "n": len(subset),
            "organization_n": int(subset["has_organization"].sum()),
            "organization_proportion": subset["has_organization"].mean(),
            "named_policy_n": int(subset["has_named_policy"].sum()),
            "named_policy_proportion": subset["has_named_policy"].mean(),
            "own_developer_n": int(subset["own_developer_named"].sum()),
            "own_developer_proportion": subset["own_developer_named"].mean(),
            "named_research_n": int(subset["named_research_or_evidence"].sum()),
            "named_research_proportion": subset["named_research_or_evidence"].mean(),
        })
    return pd.DataFrame(rows)


def multilabel_distribution(df: pd.DataFrame, source_column: str, label_name: str) -> pd.DataFrame:
    """Return a complete model-level distribution for a semicolon-delimited field."""
    parsed = df[source_column].fillna("").map(split_labels)
    labels = sorted(set().union(*parsed))
    rows = []
    for model_name in MODEL_ORDER:
        indices = df.index[df["model"] == model_name]
        for label in labels:
            count = int(parsed.loc[indices].map(lambda values, target=label: target in values).sum())
            rows.append({
                "model": model_name, label_name: label, "n": len(indices),
                "count": count, "proportion": count / len(indices),
            })
    return pd.DataFrame(rows)


def language_differences(frame_by_model_language: pd.DataFrame, specificity_by_model_language: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for frame in FRAMES:
            subset = frame_by_model_language[(frame_by_model_language["model"] == model) & (frame_by_model_language["frame"] == frame)]
            values = subset.set_index("language")["proportion"]
            rows.append({"model": model, "outcome": frame, "english": values.get("English", np.nan), "chinese": values.get("Chinese", np.nan), "chinese_minus_english": values.get("Chinese", np.nan) - values.get("English", np.nan)})
        subset_spec = specificity_by_model_language[specificity_by_model_language["model"] == model].set_index("language")
        rows.append({
            "model": model, "outcome": "Mean specificity", "english": subset_spec.loc["English", "mean_specificity"],
            "chinese": subset_spec.loc["Chinese", "mean_specificity"],
            "chinese_minus_english": subset_spec.loc["Chinese", "mean_specificity"] - subset_spec.loc["English", "mean_specificity"],
        })
    return pd.DataFrame(rows)


def matched_language_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Compare English and Chinese within identical model, source sample, and template."""
    cross_block = df["source_sample"].astype(str) + "|" + df["Template"].astype(str) + "|" + df["model"]
    work = df.assign(cross_language_block=cross_block)
    rows = []
    for model_name in MODEL_ORDER:
        subset = work[work["model"] == model_name]
        english = subset[subset["language"] == "English"].set_index("cross_language_block")
        chinese = subset[subset["language"] == "Chinese"].set_index("cross_language_block")
        common = english.index.intersection(chinese.index)
        for frame in FRAMES:
            column = f"frame__{frame}"
            en = english.loc[common, column].astype(int).to_numpy()
            zh = chinese.loc[common, column].astype(int).to_numpy()
            en_only = int(((en == 1) & (zh == 0)).sum())
            zh_only = int(((en == 0) & (zh == 1)).sum())
            discordant = en_only + zh_only
            rows.append({
                "model": model_name, "outcome": frame, "matched_blocks": len(common),
                "english_only": en_only, "chinese_only": zh_only,
                "chinese_minus_english": float((zh - en).mean()) if len(common) else np.nan,
                "exact_mcnemar_p": float(binomtest(zh_only, discordant, 0.5).pvalue) if discordant else 1.0,
            })
        en_spec = english.loc[common, "specificity"].to_numpy()
        zh_spec = chinese.loc[common, "specificity"].to_numpy()
        rows.append({
            "model": model_name, "outcome": "Specificity score", "matched_blocks": len(common),
            "english_only": np.nan, "chinese_only": np.nan,
            "chinese_minus_english": float((zh_spec - en_spec).mean()) if len(common) else np.nan,
            "exact_mcnemar_p": np.nan,
        })
    return pd.DataFrame(rows)


def specificity_adjusted_frames(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    categorical = ["model", "language", "Genre", "source_sample"]
    numeric = ["specificity"]
    rows = []
    for frame in FRAMES:
        outcome = f"frame__{frame}"
        y = df[outcome].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)]
        )
        model = Pipeline([
            ("features", transformer),
            ("logit", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=20260725)),
        ])
        model.fit(df[categorical + numeric], y)
        adjusted = {}
        for model_name in MODEL_ORDER:
            counterfactual = df[categorical + numeric].copy()
            counterfactual["model"] = model_name
            adjusted[model_name] = float(model.predict_proba(counterfactual)[:, 1].mean())
        raw_rates = raw[raw["frame"] == frame].set_index("model")["proportion"].to_dict()
        for model_name in MODEL_ORDER:
            rows.append({
                "frame": frame,
                "model": model_name,
                "raw_proportion": raw_rates[model_name],
                "specificity_adjusted_probability": adjusted[model_name],
                "adjusted_minus_raw": adjusted[model_name] - raw_rates[model_name],
            })
    return pd.DataFrame(rows)


def matched_pair_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        left = df[df["model"] == model_a].set_index("block")
        right = df[df["model"] == model_b].set_index("block")
        common = left.index.intersection(right.index)
        if len(common) < 5:
            continue
        for frame in FRAMES:
            column = f"frame__{frame}"
            a = left.loc[common, column].astype(int).to_numpy()
            b = right.loc[common, column].astype(int).to_numpy()
            a_only = int(((a == 1) & (b == 0)).sum())
            b_only = int(((a == 0) & (b == 1)).sum())
            discordant = a_only + b_only
            p_value = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
            rows.append({
                "model_a": model_a, "model_b": model_b, "outcome": frame, "matched_blocks": len(common),
                "model_a_only": a_only, "model_b_only": b_only, "paired_difference": float((a - b).mean()),
                "exact_mcnemar_p": p_value,
            })
        a_spec = left.loc[common, "specificity"].to_numpy()
        b_spec = right.loc[common, "specificity"].to_numpy()
        rows.append({
            "model_a": model_a, "model_b": model_b, "outcome": "Specificity score", "matched_blocks": len(common),
            "model_a_only": np.nan, "model_b_only": np.nan, "paired_difference": float((a_spec - b_spec).mean()),
            "exact_mcnemar_p": np.nan,
        })
    return pd.DataFrame(rows)


def candidate_quotes(df: pd.DataFrame) -> pd.DataFrame:
    english = df[df["language"] == "English"].copy()
    candidates = []
    for block, group in english.groupby("block"):
        if len(group) < 2:
            continue
        for (_, row_a), (_, row_b) in combinations(group.iterrows(), 2):
            frames_a = row_a["frame_set"]
            frames_b = row_b["frame_set"]
            unique_a = frames_a - frames_b
            unique_b = frames_b - frames_a
            spec_gap = abs(int(row_a["specificity"]) - int(row_b["specificity"]))
            mechanism_diff = int(row_a["mechanism"] != row_b["mechanism"])
            reference_diff = abs(int(row_a["has_named_policy"]) - int(row_b["has_named_policy"])) + abs(int(row_a["own_developer_named"]) - int(row_b["own_developer_named"]))
            score = 3 * spec_gap + 2 * (len(unique_a) + len(unique_b)) + mechanism_diff + 2 * reference_diff
            if score < 5:
                continue
            target_a = next((frame for frame in FRAME_PRIORITY if frame in unique_a), None)
            target_b = next((frame for frame in FRAME_PRIORITY if frame in unique_b), None)
            pattern_a = FRAME_PATTERNS.get(target_a) if target_a else (REFERENCE_PATTERN if row_a["has_named_policy"] else None)
            pattern_b = FRAME_PATTERNS.get(target_b) if target_b else (REFERENCE_PATTERN if row_b["has_named_policy"] else None)
            candidates.append({
                "comparison_score": score,
                "block": block,
                "source_sample": row_a["source_sample"],
                "template": row_a["Template"],
                "language": row_a["language"],
                "id_a": row_a["ID"], "model_a": row_a["model"], "specificity_a": int(row_a["specificity"]),
                "mechanism_a": row_a["mechanism"], "distinctive_frame_a": target_a or "",
                "quote_a": excerpt_for(row_a["Key quote"], pattern_a),
                "id_b": row_b["ID"], "model_b": row_b["model"], "specificity_b": int(row_b["specificity"]),
                "mechanism_b": row_b["mechanism"], "distinctive_frame_b": target_b or "",
                "quote_b": excerpt_for(row_b["Key quote"], pattern_b),
            })
    result = pd.DataFrame(candidates)
    if result.empty:
        return result
    return result.sort_values(["comparison_score", "block"], ascending=[False, True]).head(40)


def top_frame_lines(frame_model: pd.DataFrame) -> list[str]:
    lines = []
    for frame in FRAMES:
        subset = frame_model[frame_model["frame"] == frame].sort_values("proportion", ascending=False)
        top = subset.iloc[0]
        bottom = subset.iloc[-1]
        lines.append(
            f"- **{frame}:** {top['model']} was highest at {format_pct(top['proportion'])}; "
            f"{bottom['model']} was lowest at {format_pct(bottom['proportion'])} (raw range {format_pct(top['proportion'] - bottom['proportion'])})."
        )
    return lines


def write_memo(
    df: pd.DataFrame,
    frame_model: pd.DataFrame,
    mechanism_model: pd.DataFrame,
    specificity_model: pd.DataFrame,
    specificity_length: pd.DataFrame,
    references: pd.DataFrame,
    adjusted: pd.DataFrame,
    matched: pd.DataFrame,
    quotes: pd.DataFrame,
    matched_language: pd.DataFrame,
    manifest: dict,
) -> None:
    frame_ranges = []
    for frame in FRAMES:
        raw_subset = frame_model[frame_model["frame"] == frame].set_index("model")["proportion"]
        adj_subset = adjusted[adjusted["frame"] == frame].set_index("model")["specificity_adjusted_probability"]
        frame_ranges.append({
            "frame": frame,
            "raw_range": raw_subset.max() - raw_subset.min(),
            "adjusted_range": adj_subset.max() - adj_subset.min(),
            "raw_top": raw_subset.idxmax(),
            "adjusted_top": adj_subset.idxmax(),
        })
    frame_ranges_df = pd.DataFrame(frame_ranges).sort_values("raw_range", ascending=False)
    most_differentiated = frame_ranges_df.iloc[0]

    spec_sorted = specificity_model.sort_values("mean_specificity", ascending=False)
    spec_top = spec_sorted.iloc[0]
    spec_bottom = spec_sorted.iloc[-1]
    policy_sorted = references.sort_values("named_policy_proportion", ascending=False)
    own_sorted = references.sort_values("own_developer_proportion", ascending=False)
    no_clear = mechanism_model[mechanism_model["mechanism"] == "No clear recommendation"].sort_values("proportion", ascending=False)
    capability = mechanism_model[mechanism_model["mechanism"] == "Capability-threshold release"].sort_values("proportion", ascending=False)

    def frame_rate(model: str, frame: str) -> float:
        return float(frame_model[(frame_model["model"] == model) & (frame_model["frame"] == frame)]["proportion"].iloc[0])

    def adjusted_rate(model: str, frame: str) -> float:
        return float(adjusted[(adjusted["model"] == model) & (adjusted["frame"] == frame)]["specificity_adjusted_probability"].iloc[0])

    def model_language_rate(model: str, language: str, frame: str) -> float:
        subset = df[(df["model"] == model) & (df["language"] == language)]
        return float(subset[f"frame__{frame}"].mean())

    def ref_rate(model: str, column: str) -> float:
        return float(references.loc[references["model"] == model, column].iloc[0])

    claude_mistral_spec = float(matched[
        (matched["model_a"] == "Claude") & (matched["model_b"] == "Mistral") & (matched["outcome"] == "Specificity score")
    ]["paired_difference"].iloc[0])
    claude_mistral_irrev = matched[
        (matched["model_a"] == "Claude") & (matched["model_b"] == "Mistral")
        & (matched["outcome"] == "Irreversibility and loss of control")
    ].iloc[0]
    mistral_language_application = matched_language[
        (matched_language["model"] == "Mistral")
        & (matched_language["outcome"] == "Application proliferation")
    ].iloc[0]
    length_top = specificity_length.sort_values("length_adjusted_standardized_mean", ascending=False).iloc[0]
    length_bottom = specificity_length.sort_values("length_adjusted_standardized_mean", ascending=False).iloc[-1]

    matched_notable = matched[
        (matched["outcome"] != "Specificity score")
        & (matched["matched_blocks"] >= 5)
        & (matched["exact_mcnemar_p"] <= 0.10)
    ].sort_values(["exact_mcnemar_p", "paired_difference"], ascending=[True, False]).head(12)

    quote_lines = []
    for _, row in quotes.head(8).iterrows():
        quote_lines.extend([
            f"### {row['model_a']} versus {row['model_b']} — {row['template']} ({row['id_a']} / {row['id_b']})",
            f"- {row['model_a']} (specificity {row['specificity_a']}; {row['mechanism_a']}): “{row['quote_a']}”",
            f"- {row['model_b']} (specificity {row['specificity_b']}; {row['mechanism_b']}): “{row['quote_b']}”",
            "",
        ])

    lines = [
        "# RQ4 B4 reviewed close-reading results",
        "",
        "## Status and scope",
        "",
        f"This memo analyzes the {len(df)} manually reviewed responses on B4 (open release of frontier-model weights). "
        "It supplements the primary RQ4 stance tests by describing the rationales and release mechanisms used in the responses. "
        "It does not independently identify developer influence: the sample is exploratory, and frames were coded as present whether endorsed or rejected.",
        "",
        f"Frozen source SHA-256: `{manifest['source_sha256']}`.",
        "",
        "## Direct answers",
        "",
        f"1. **Models differed more in specificity and institutional detail than in their broad argumentative repertoire.** "
        f"Mean specificity was highest for {spec_top['model']} ({spec_top['mean_specificity']:.2f}) and lowest for {spec_bottom['model']} ({spec_bottom['mean_specificity']:.2f}) on the 1–3 scale.",
        f"2. **The largest raw frame difference concerned {most_differentiated['frame'].lower()}.** "
        f"The cross-model prevalence range was {format_pct(most_differentiated['raw_range'])}; after the specificity diagnostic it was {format_pct(most_differentiated['adjusted_range'])}. "
        f"The highest model was {most_differentiated['raw_top']} before adjustment and {most_differentiated['adjusted_top']} after adjustment.",
        f"3. **Named policy references were concentrated rather than universal.** {policy_sorted.iloc[0]['model']} named a policy in "
        f"{format_pct(policy_sorted.iloc[0]['named_policy_proportion'])} of its reviewed responses, compared with {format_pct(policy_sorted.iloc[-1]['named_policy_proportion'])} for {policy_sorted.iloc[-1]['model']}.",
        f"4. **Self-reference was uncommon overall but model-dependent.** {own_sorted.iloc[0]['model']} named its own developer in "
        f"{format_pct(own_sorted.iloc[0]['own_developer_proportion'])} of responses; the lowest rate was {format_pct(own_sorted.iloc[-1]['own_developer_proportion'])} for {own_sorted.iloc[-1]['model']}.",
        f"5. **Mechanism coding confirms that many recommendations remained underspecified.** {no_clear.iloc[0]['model']} had the highest rate of no clear recommendation ({format_pct(no_clear.iloc[0]['proportion'])}). "
        f"Capability-threshold release was most common for {capability.iloc[0]['model']} ({format_pct(capability.iloc[0]['proportion'])}).",
        f"6. **The specificity gap was not solely a response-length artifact.** After descriptively adjusting for log response length within language, language, genre, and source sample, the standardized mean remained highest for {length_top['model']} ({length_top['length_adjusted_standardized_mean']:.2f}) and lowest for {length_bottom['model']} ({length_bottom['length_adjusted_standardized_mean']:.2f}). This is a sensitivity diagnostic, not a causal control.",
        "",
        "## Frame prevalence by model",
        "",
        *top_frame_lines(frame_model),
        "",
        "## Same-template, same-language comparisons",
        "",
        "These are exploratory, unadjusted exact tests across a manually selected close-reading sample; the p-values are descriptive checks, not confirmatory evidence.",
        "",
    ]
    if matched_notable.empty:
        lines.append("No frame contrast reached the exploratory exact McNemar threshold of p ≤ .10 across at least five shared blocks.")
    else:
        for _, row in matched_notable.iterrows():
            direction = row["model_a"] if row["paired_difference"] > 0 else row["model_b"]
            lines.append(
                f"- **{row['outcome']}:** {direction} used the frame more often in the {int(row['matched_blocks'])} exact shared prompt blocks "
                f"({row['model_a']}−{row['model_b']} paired difference {row['paired_difference']:+.3f}; exact McNemar p{format_p(row['exact_mcnemar_p'])})."
            )
    lines.extend([
        "",
        "## Chinese-origin models and Chinese-language responses",
        "",
        f"The two Chinese-origin models did not share a single framing profile. DeepSeek frequently used competition and deconcentration ({format_pct(frame_rate('DeepSeek', 'Competition and deconcentration'))}) and was the second-highest user of irreversibility after Claude ({format_pct(frame_rate('DeepSeek', 'Irreversibility and loss of control'))}). Qwen instead had the highest application-proliferation rate ({format_pct(frame_rate('Qwen', 'Application proliferation'))}) and frequently developed equitable-access arguments ({format_pct(frame_rate('Qwen', 'Equitable and Global South access'))}).",
        "",
        f"Prompt language produced a separate pattern. Across models, application-proliferation framing appeared in {format_pct(df.loc[df['language'] == 'Chinese', 'frame__Application proliferation'].mean())} of Chinese responses and {format_pct(df.loc[df['language'] == 'English', 'frame__Application proliferation'].mean())} of English responses. The cleanest model-specific comparison was Mistral: {format_pct(model_language_rate('Mistral', 'Chinese', 'Application proliferation'))} in Chinese versus {format_pct(model_language_rate('Mistral', 'English', 'Application proliferation'))} in English, with Chinese-only use in eight of ten exact cross-language template pairs (paired difference {mistral_language_application['chinese_minus_english']:+.3f}; exact McNemar p{format_p(mistral_language_application['exact_mcnemar_p'])}). Because the strongest language contrast occurred for the European-origin model, it should not be interpreted as a Chinese-model effect.",
        "",
        "## Main.tex-ready draft",
        "",
        "To examine what the stance comparison left out, we conducted an exploratory close reading of 196 responses to the open-weight issue (B4). The review used a multi-label codebook: a frame was recorded when the response substantively developed it, regardless of whether the response ultimately endorsed that reasoning. We additionally coded the preferred release mechanism, the specificity of the argument on a three-point scale, and named institutional references. Because the sample was selected for close reading and the codebook was refined during review, these counts describe this corpus rather than estimate population-wide model effects.",
        "",
        f"The broadest frames were shared across models: open-research arguments appeared in {format_pct(df['frame__Open research ecosystem'].mean())} of responses and safety or dangerous-capability arguments in {format_pct(df['frame__Safety and dangerous capabilities'].mean())}. The largest differences concerned narrower ways of developing those arguments. Claude invoked irreversibility or loss of control in {format_pct(frame_rate('Claude', 'Irreversibility and loss of control'))} of responses, compared with {format_pct(frame_rate('GPT-4o', 'Irreversibility and loss of control'))} for GPT-4o and {format_pct(frame_rate('Llama', 'Irreversibility and loss of control'))} for Llama. This contrast remained large in the specificity diagnostic (standardized probabilities: Claude {format_pct(adjusted_rate('Claude', 'Irreversibility and loss of control'))}, GPT-4o {format_pct(adjusted_rate('GPT-4o', 'Irreversibility and loss of control'))}, and Llama {format_pct(adjusted_rate('Llama', 'Irreversibility and loss of control'))}). Mistral most often developed equitable-access arguments ({format_pct(frame_rate('Mistral', 'Equitable and Global South access'))}) and sovereignty or technological-autonomy arguments ({format_pct(frame_rate('Mistral', 'Sovereignty and technological autonomy'))}), although the latter remained uncommon in absolute terms. Claude and Mistral provide the cleanest check against the possibility that these patterns only reflect generic specificity: their specificity scores were nearly identical in exact shared prompt blocks (Claude−Mistral mean difference {claude_mistral_spec:+.3f}), but Claude used the irreversibility frame substantially more often (paired difference {claude_mistral_irrev['paired_difference']:+.3f}; exact McNemar p{format_p(claude_mistral_irrev['exact_mcnemar_p'])}).",
        "",
        f"Institutional detail also varied. Mistral named a policy in {format_pct(ref_rate('Mistral', 'named_policy_proportion'))} of responses and Claude in {format_pct(ref_rate('Claude', 'named_policy_proportion'))}, whereas GPT-4o did so in {format_pct(ref_rate('GPT-4o', 'named_policy_proportion'))}. Mechanism recommendations differed as well: {format_pct(no_clear.iloc[0]['proportion'])} of GPT-4o responses offered no clear release mechanism, while Claude most often recommended capability-threshold release ({format_pct(capability.iloc[0]['proportion'])}). These results suggest model-specific differences in how the open-weight question was represented even where the models drew on broadly similar pro- and anti-release considerations.",
        "",
        "The close reading changes the interpretation of B4 without overturning the stance result. Models often drew on the same broad repertoire—research openness, dangerous-capability risks, competition, and auditing—but differed in how concretely they developed those arguments and in the institutional or policy references they used. Similar stance scores therefore did not imply identical representations of the policy problem. At the same time, these differences cannot establish that developer advocacy caused model behavior: they may reflect training data, post-training, general house style, or prompt-language effects, and the manually reviewed B4 sample is exploratory.",
        "",
        f"The Chinese-origin models also differed from one another: DeepSeek more often foregrounded irreversible dual-use proliferation, whereas Qwen more often developed downstream-application and equitable-access arguments. Prompt language created an additional distinction. Application-proliferation framing appeared in {format_pct(df.loc[df['language'] == 'Chinese', 'frame__Application proliferation'].mean())} of Chinese responses but {format_pct(df.loc[df['language'] == 'English', 'frame__Application proliferation'].mean())} of English responses, with the strongest matched contrast occurring for Mistral rather than a Chinese-origin model. These findings motivate the next analysis: stance alone cannot show whether models define the problem, identify affected actors, or make particular consequences salient in systematically different ways. RQ5 therefore turns from which position models take to how they frame AI-governance disputes across model origins and prompt languages.",
        "",
        "## Candidate matched quotations",
        "",
        "These candidates come from the same template and language. Select quotations only after confirming that they illustrate an aggregate pattern in the tables.",
        "",
        *quote_lines,
        "## Reproducibility",
        "",
        f"The matched language file uses {int(matched_language['matched_blocks'].sum() / len(matched_language['outcome'].unique()))} model-specific exact template pairs; unmatched language averages are retained only as descriptive context.",
        "",
        "Run `.venv/bin/python evaluation/scripts/06_analysis/rq4_b4_framing_analysis.py` from the project root. The script reads only the reviewed `Responses FINAL` sheet, verifies IDs and labels, freezes it to CSV, and regenerates every table, quote candidate, and this memo.",
    ])
    (OUTPUT_DIR / "RQ4_B4_RESULTS_MEMO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, manifest = load_reviewed()

    frozen_columns = [
        "ID", "source_sample", "model", "model_full", "developer", "Model (blind)", "Lang", "language", "Genre", "Template",
        "Prompt", "Response", "Frames — FINAL", "Mechanism — FINAL", "Mechanism basis",
        "Specificity — FINAL (1–3)", "Specificity basis", "Risks — FINAL", "Rhetorical form — FINAL",
        "Institutional refs — FINAL", "Organizations — FINAL", "Named policies — FINAL", "Reference accuracy status",
        "Key quote", "FINAL review priority", "Prior corrections", "Prior reviewer notes",
    ]
    df[frozen_columns].to_csv(OUTPUT_DIR / "reviewed_coding_frozen.csv", index=False)
    (OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    composition = sample_composition(df)
    frame_model, frame_model_language = frame_tables(df)
    mechanism_model, mechanism_model_language = mechanism_tables(df)
    specificity_model, specificity_model_language = specificity_tables(df)
    specificity_length = specificity_length_diagnostic(df)
    references = reference_table(df)
    risks = multilabel_distribution(df, "Risks — FINAL", "risk")
    rhetorical_forms = multilabel_distribution(df, "Rhetorical form — FINAL", "rhetorical_form")
    language = language_differences(frame_model_language, specificity_model_language)
    matched_language = matched_language_contrasts(df)
    adjusted = specificity_adjusted_frames(df, frame_model)
    matched = matched_pair_contrasts(df)
    quotes = candidate_quotes(df)

    outputs = {
        "sample_composition.csv": composition,
        "frame_prevalence_by_model.csv": frame_model,
        "frame_prevalence_by_model_language.csv": frame_model_language,
        "mechanism_distribution_by_model.csv": mechanism_model,
        "mechanism_distribution_by_model_language.csv": mechanism_model_language,
        "specificity_by_model.csv": specificity_model,
        "specificity_by_model_language.csv": specificity_model_language,
        "specificity_length_diagnostic.csv": specificity_length,
        "institutional_references_by_model.csv": references,
        "risk_prevalence_by_model.csv": risks,
        "rhetorical_form_by_model.csv": rhetorical_forms,
        "language_differences.csv": language,
        "matched_language_contrasts.csv": matched_language,
        "specificity_adjusted_frames.csv": adjusted,
        "matched_pair_contrasts.csv": matched,
        "candidate_matched_quotes.csv": quotes,
    }
    for filename, table in outputs.items():
        table.to_csv(OUTPUT_DIR / filename, index=False)

    write_memo(
        df, frame_model, mechanism_model, specificity_model, specificity_length,
        references, adjusted, matched, quotes, matched_language, manifest,
    )

    print(json.dumps({
        "rows": len(df),
        "models": df["model"].value_counts().reindex(MODEL_ORDER).to_dict(),
        "languages": df["language"].value_counts().to_dict(),
        "shared_blocks_by_size": df.groupby("block").size().value_counts().sort_index().to_dict(),
        "output_dir": str(OUTPUT_DIR),
        "source_sha256": manifest["source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
