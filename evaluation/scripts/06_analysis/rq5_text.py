"""RQ5 text analyses: embeddings, similarity, stylometry, and the B7 repertoire audit. Requires raw responses."""

import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master, load_responses  # noqa: E402
from viz_style import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    MODEL_MARKER,
    MODEL_ORDER,
    MODEL_ORIGIN,
    ORIGIN_COLOR,
    SEQUENTIAL,
    SLOT,
    apply_style,
    save,
)

EMB = Path("evaluation/analysis/outputs/embeddings")  # written by embed, read by similarity
FIG = Path("evaluation/analysis/outputs/figures")
TAB = Path("evaluation/analysis/outputs/tables")


# ============================================================================
# embed  (nb5a) -- embed every response into a shared multilingual vector space
# ============================================================================
#
# One-time build step for the response-similarity analysis (similarity, below). Runs
# entirely locally on the Mac GPU (MPS) with open weights -- no API, no cost.
#
# Default model: intfloat/multilingual-e5-base (278M, 768-dim, 512-token window).
# Chosen because EN and ZH translations land close together in its space, which is
# what lets us compare an English response to a Chinese one at all.
#
# Two alternate encoders are available via --model, as a robustness check on whether
# similarity's findings are a property of the models or an artifact of this one encoder:
#     labse    Google LaBSE -- trained for cross-lingual BITEXT RETRIEVAL, a different
#              objective from e5's contrastive retrieval training. Short 256-token
#              window, so it leans harder on the chunk-and-pool step below.
#     bge-m3   BAAI BGE-M3 -- modern, strong on multilingual MTEB, and natively
#              handles ~8k tokens, so most responses fit in a SINGLE window with no
#              chunking at all. A check on whether chunk-and-pool itself distorts
#              anything relative to a native long-context encoder.
#
# Responses are essays (EN median ~5k chars), which can overrun a small window.
# Each response is cut into windows sized to the chosen encoder, every window is
# encoded, and the windows are mean-pooled back into one vector per response.
# Truncating to the front of the window instead would embed the introduction, not
# the argument.
#
# Output (evaluation/analysis/outputs/embeddings/<sample|full>/<model_slug>/):
#     index.csv       one row per response: rep, model, prompt_id, issue_id, ...
#     emb.npy         float16 [n_responses, dim], L2-normalised, row-aligned to index.csv
#
# Written in shards so an interrupted run resumes instead of restarting -- close the
# laptop mid-run and the next invocation picks up where it left off.
#
# --sample keeps every issue, model, language and repetition, and thins only the
# template dimension (one template per template category). So the pilot can still
# estimate the sampling noise floor and the model/issue/language effects; the only
# component it is thin on is template-to-template variance.

SAMPLE_TEMPLATES_PER_CATEGORY = 1
BATCH = 64
SHARD_SIZE = 2000  # responses per shard

# repo id, "query: "-style prefix (None if the encoder wants raw text), window size
# in characters, batch size. e5 needs the prefix per its model card; LaBSE/BGE-M3
# do not. BGE-M3's window is capped well below its native 8192 tokens and its batch
# kept small -- long sequences at batch 64 blow through MPS's memory limit.
MODEL_CONFIGS = {
    "e5": ("intfloat/multilingual-e5-base", "query: ", 1400, 64),
    "labse": ("sentence-transformers/LaBSE", None, 700, 64),  # 256-token window
    "bge-m3": ("BAAI/bge-m3", None, 3000, 8),  # long-context, but memory-hungry per batch
}


def chunk(text: str, chunk_chars: int) -> list[str]:
    """Cut a response into encoder-sized windows."""
    text = text.strip()
    if not text:
        return [""]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def sample_templates(n_per_category: int) -> list[str]:
    """n templates per template category, so the sample spans all four writing styles."""
    m = load_master()[["template_id", "template_category"]].drop_duplicates()
    picked = (
        m.sort_values("template_id")
        .groupby("template_category")["template_id"]
        .head(n_per_category)
    )
    return sorted(picked)


def run_embed(args: argparse.Namespace) -> None:
    repo_id, prefix, chunk_chars, batch_size = MODEL_CONFIGS[args.model]

    # sample-size/encoder combos must each get their own dir: shards are
    # position-indexed against the response table, so mixing them would silently
    # misalign vectors. "sample" (n=1) keeps its original bare name so the existing
    # 3-encoder pilot comparison stays untouched; other sizes get a suffix.
    n = args.templates_per_category
    tag = "sample" if n == 1 else f"sample{n}"
    out = EMB / (tag if args.sample else "full") / args.model
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    df = load_responses()
    if args.sample:
        keep = sample_templates(n)
        df = df[df["template_id"].isin(keep)]
        print(f"sample mode: {n}/category, templates {keep}")

    # stable order, so shard k always covers the same responses across runs
    df = df.sort_values(["rep", "model", "prompt_id"]).reset_index(drop=True)
    print(f"{len(df):,} responses to embed")

    index_cols = ["rep", "model", "prompt_id", "issue_id", "template_id", "lang"]
    df[index_cols].to_csv(out / "index.csv", index=False)

    from sentence_transformers import SentenceTransformer

    print(f"encoder: {repo_id}")
    model = SentenceTransformer(repo_id, device="mps")

    n_shards = (len(df) + SHARD_SIZE - 1) // SHARD_SIZE
    t_start = time.time()

    for s in range(n_shards):
        shard_path = shards / f"{s:04d}.npy"
        if shard_path.exists():
            continue

        block = df.iloc[s * SHARD_SIZE : (s + 1) * SHARD_SIZE]

        # flatten every response in the shard into windows, remembering which
        # response each window came from, so we can pool them back afterwards
        windows: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(block["response"]):
            for w in chunk(str(text), chunk_chars):
                windows.append((prefix + w) if prefix else w)
                owner.append(i)

        vecs = model.encode(
            windows,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # mean-pool windows -> one vector per response, then re-normalise so that
        # a dot product between any two rows is a cosine similarity
        pooled = np.zeros((len(block), vecs.shape[1]), dtype=np.float32)
        counts = np.zeros(len(block), dtype=np.float32)
        np.add.at(pooled, owner, vecs)
        np.add.at(counts, owner, 1.0)
        pooled /= np.maximum(counts, 1)[:, None]
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

        np.save(shard_path, pooled.astype(np.float16))

        done = s + 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (n_shards - done)
        print(
            f"shard {done}/{n_shards}  ({done * SHARD_SIZE:,} responses)  "
            f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m",
            flush=True,
        )

    emb = np.concatenate([np.load(shards / f"{s:04d}.npy") for s in range(n_shards)])
    assert len(emb) == len(df), f"{len(emb)} embeddings vs {len(df)} responses"
    np.save(out / "emb.npy", emb)
    print(f"\nwrote {out / 'emb.npy'}  shape={emb.shape}  ({emb.nbytes / 1e6:.0f} MB)")


# ============================================================================
# similarity  (nb5b) -- how similar are the responses themselves, not their stances
# ============================================================================
#
# Reads the embeddings built by `embed` and asks four questions:
#
# 1. NOISE FLOOR. The same model, given the same prompt, three times at temperature 1.0.
#    Nothing differs but sampling, so this is as similar as two responses can get. Every
#    other number below is read against it -- a raw cosine of 0.72 means nothing on its
#    own, but "0.72 against a 0.89 ceiling and a 0.41 floor" means something.
#
# 2. VARIANCE DECOMPOSITION. Of all the variation in what gets written, how much is
#    explained by the issue, the template, and the model? Computed as a distance-based
#    eta-squared. On L2-normalised vectors, cosine distance is proportional to squared
#    Euclidean, so the between/within sums of squares decompose from group centroids
#    directly -- no N x N distance matrix is ever needed.
#
# 3. MODEL GEOMETRY. Centroid per model, cosine between them, projected to 2-D. Do the
#    US / CN / EU models separate on content the way they do on stance?
#
# 4. DISCRIMINABILITY. After subtracting each prompt's own cell mean -- i.e. removing
#    the topic and the template, leaving only how the model wrote it -- can a classifier
#    still recover the model's origin? Tested on held-out ISSUES, so it cannot win by
#    memorising topics.
#
# Language is handled carefully. Multilingual encoders leave a residual same-script
# clustering effect, so a headline "language explains X%" would partly measure the
# encoder rather than the models. Everything here is therefore computed WITHIN language,
# and the EN/ZH comparison is reported separately and labelled as what it is.


def mean_pairwise_cos(X: np.ndarray) -> float:
    """Mean cosine over all distinct pairs of rows in X (rows are L2-normalised).

    Uses the identity ||sum_i x_i||^2 = n + sum_{i != j} <x_i, x_j>, which gives the
    mean pairwise similarity in O(n*d) instead of building an n x n matrix.
    """
    n = len(X)
    if n < 2:
        return np.nan
    total = float(np.dot(X.sum(0), X.sum(0)))
    return (total - n) / (n * (n - 1))


def _groups_as_index(df: pd.DataFrame, keys: list[str]):
    return df.groupby(keys, observed=True).indices.items()


def tier_similarity(df: pd.DataFrame, X: np.ndarray, keys: list[str]) -> float:
    """Mean within-group pairwise cosine, averaged over groups of the given keys."""
    vals = []
    for _, idx in _groups_as_index(df, keys):
        if len(idx) > 1:
            v = mean_pairwise_cos(X[idx])
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def eta_squared(df: pd.DataFrame, X: np.ndarray, factor: str) -> float:
    """Share of total squared-distance variation that lies BETWEEN levels of `factor`.

    SS_total = sum_i ||x_i - grand_mean||^2
    SS_within = sum_g sum_{i in g} ||x_i - centroid_g||^2
    eta^2 = 1 - SS_within / SS_total

    This is the Euclidean case of distance-based ANOVA (PERMANOVA), computed from
    centroids rather than from a pairwise distance matrix.
    """
    grand = X.mean(0)
    ss_total = float(((X - grand) ** 2).sum())
    ss_within = 0.0
    for _, idx in _groups_as_index(df, [factor]):
        Xg = X[idx]
        ss_within += float(((Xg - Xg.mean(0)) ** 2).sum())
    return 1.0 - ss_within / ss_total


def permutation_pvalue(
    R: np.ndarray, y: np.ndarray, groups: np.ndarray, observed_bal: float,
    n_perm: int = 1000, seed: int = 0,
) -> float:
    """One-sided permutation p-value for the balanced-accuracy discriminability score.

    Shuffles the origin labels (breaking any true style<->origin link, but leaving
    the CV group structure and class sizes untouched) and refits the same GroupKFold
    logistic regression each time. The p-value is the share of shuffles whose
    balanced accuracy matches or beats the real one -- i.e. how often this level of
    "discriminability" would arise from a style vector that carries no origin signal.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    null = np.empty(n_perm)
    for p in range(n_perm):
        y_perm = rng.permutation(y)
        preds = np.empty(len(y_perm), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(R, y_perm, groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(R[tr], y_perm[tr])
            preds[te] = clf.predict(R[te])
        null[p] = np.mean([(preds[y_perm == o] == o).mean() for o in classes])
    return float((1 + np.sum(null >= observed_bal)) / (1 + n_perm))


def residualise(df: pd.DataFrame, X: np.ndarray, keys: list[str]) -> np.ndarray:
    """Subtract each prompt-cell's mean vector, leaving only model-specific style.

    After this, two responses to the same prompt differ only in how the model wrote
    it -- the topic and the template have been projected out.
    """
    R = X.copy()
    for _, idx in _groups_as_index(df, keys):
        R[idx] -= R[idx].mean(0)
    return R


def fig_tiers(tiers: pd.DataFrame, lang: str, suffix: str = "") -> None:
    """A dot plot, not bars: every value sits between 0.92 and 0.98, so bars anchored
    at zero would render the differences invisible. Cosines in this encoder occupy a
    narrow cone, so what carries meaning is the ORDER of the tiers and their spacing
    between the two reference lines -- not the absolute numbers."""
    d = tiers[tiers["lang"] == lang].sort_values("cosine")
    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    ceiling = float(d.loc[d["tier"].str.startswith("same model, same prompt"), "cosine"].iloc[0])
    floor = float(d["cosine"].min())

    for ref, label in [(ceiling, "ceiling: same model, same prompt"),
                       (floor, "floor: same model, unrelated prompt")]:
        ax.axvline(ref, color=INK_MUTED, lw=0.9, alpha=0.45, zorder=2)

    ax.hlines(d["tier"], floor, d["cosine"], color=GRID, lw=1.4, zorder=2)
    ax.scatter(d["cosine"], d["tier"], s=64, color=SLOT[0], zorder=3,
               edgecolor="#fcfcfb", linewidth=1.4)
    for y, v in zip(d["tier"], d["cosine"]):
        ax.text(v + 0.0022, y, f"{v:.3f}", va="center", ha="left", fontsize=8, color=INK)

    pad = (ceiling - floor) * 0.28
    ax.set_xlim(floor - pad * 0.5, ceiling + pad)
    ax.set_xlabel("mean cosine similarity between responses")
    ax.set_title(
        f"Two different models on the same prompt agree more than\n"
        f"one model does with itself on a different template ({lang.upper()})"
    )
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    save(fig, FIG / f"nb5b_similarity_tiers_{lang}{suffix}")


def fig_variance(var: pd.DataFrame, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    langs = ["en", "zh"]
    factors = var["factor"].unique().tolist()
    w = 0.38
    x = np.arange(len(factors))
    for i, lang in enumerate(langs):
        d = var[var["lang"] == lang].set_index("factor").loc[factors]
        ax.bar(
            x + (i - 0.5) * w,
            d["eta_sq"],
            width=w - 0.03,
            label=lang.upper(),
            color=SLOT[i],
            zorder=3,
        )
        for xi, v in zip(x + (i - 0.5) * w, d["eta_sq"]):
            ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylabel("share of content variation explained ($\\eta^2$)")
    ax.set_title("What decides what a model writes?")
    ax.legend(title=None, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    save(fig, FIG / f"nb5b_variance_decomposition{suffix}")


def fig_model_heatmap(sim: dict[str, pd.DataFrame], suffix: str = "") -> None:
    """Style similarity is signed -- two models can lean the same way or opposite ways
    relative to the average answer -- so this is a diverging scale with grey at zero."""
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#c33534", "#ef9d9c", "#d8d7d3", "#86b6ef", "#1c5cab"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    off = np.concatenate([S.values[~np.eye(len(S), dtype=bool)] for S in sim.values()])
    lim = float(np.abs(off).max())
    for ax, (lang, S) in zip(axes, sim.items()):
        M = S.values.copy()
        np.fill_diagonal(M, np.nan)  # the diagonal is 1.0 by construction; it says nothing
        im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(S)), S.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(S)), S.index)
        for i in range(len(S)):
            for j in range(len(S)):
                if i == j:
                    continue
                v = M[i, j]
                ax.text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.66 * lim else INK,
                )
        ax.set_title(lang.upper())
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        "Given the same prompt, which models depart from the average answer the same way?",
        y=1.02, fontsize=10, fontweight="bold",
    )
    fig.colorbar(im, ax=axes, shrink=0.72, label="cosine between model style centroids")
    save(fig, FIG / f"nb5b_model_similarity_heatmap{suffix}")


def fig_mds(coords: pd.DataFrame, suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for ax, lang in zip(axes, ["en", "zh"]):
        d = coords[coords["lang"] == lang]
        for _, r in d.iterrows():
            ax.scatter(
                r["x"],
                r["y"],
                s=90,
                color=ORIGIN_COLOR[r["origin"]],
                marker=MODEL_MARKER[r["model"]],
                edgecolor="#fcfcfb",
                linewidth=1.6,
                zorder=3,
            )
            ax.annotate(
                r["model"],
                (r["x"], r["y"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=INK,
            )
        ax.set_title(lang.upper())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color(GRID)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=o, markersize=8)
        for o, c in ORIGIN_COLOR.items()
    ]
    axes[1].legend(handles=handles, title="origin", loc="best")
    fig.suptitle(
        "Where the models sit once the prompt is projected out (MDS on style centroids)",
        fontsize=10, fontweight="bold",
    )
    save(fig, FIG / f"nb5b_model_mds{suffix}")


def run_similarity(args: argparse.Namespace) -> None:
    # e5 + the default 4-template pilot keep the original bare filenames so existing
    # figures/tables are untouched; any other combo gets a suffix so all can coexist
    suffix = "" if (args.model == "e5" and args.sample_tag == "sample") else (
        f"_{args.model}" + ("" if args.sample_tag == "sample" else f"_{args.sample_tag}")
    )

    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    src = EMB / (("full" if args.full else args.sample_tag)) / args.model
    idx = pd.read_csv(src / "index.csv")
    X = np.load(src / "emb.npy").astype(np.float32)
    assert len(idx) == len(X)
    idx["origin"] = idx["model"].map(MODEL_ORIGIN)
    print(f"{len(idx):,} responses | {idx['issue_id'].nunique()} issues | "
          f"{idx['template_id'].nunique()} templates | {idx['rep'].nunique()} reps\n")

    # ---- 1. similarity tiers, within language
    tiers = []
    for lang in ["en", "zh"]:
        m = idx["lang"] == lang
        d = idx[m].reset_index(drop=True)
        Xl = X[m.to_numpy()]

        spec = {
            "same model, same prompt\n(different sample)": ["model", "issue_id", "template_id"],
            "same model, same issue\n(different template)": ["model", "issue_id", "rep"],
            "same model, same template\n(different issue)": ["model", "template_id", "rep"],
            "different model, same prompt": ["issue_id", "template_id", "rep"],
            "same model, anything\n(different issue+template)": ["model"],
        }
        for name, keys in spec.items():
            tiers.append(
                {"lang": lang, "tier": name, "cosine": tier_similarity(d, Xl, keys)}
            )
    tiers = pd.DataFrame(tiers)
    tiers.to_csv(TAB / f"nb5b_similarity_tiers{suffix}.csv", index=False)
    print("SIMILARITY TIERS (mean pairwise cosine)")
    print(tiers.pivot(index="tier", columns="lang", values="cosine").round(3), "\n")

    # ---- 2. variance decomposition, within language
    var = []
    for lang in ["en", "zh"]:
        m = idx["lang"] == lang
        d = idx[m].reset_index(drop=True)
        Xl = X[m.to_numpy()]
        for f in ["issue_id", "template_id", "model", "origin"]:
            var.append(
                {
                    "lang": lang,
                    "factor": {"issue_id": "issue", "template_id": "template"}.get(f, f),
                    "eta_sq": eta_squared(d, Xl, f),
                }
            )
    var = pd.DataFrame(var)
    var.to_csv(TAB / f"nb5b_variance_decomposition{suffix}.csv", index=False)
    print("VARIANCE DECOMPOSITION (eta^2 on response content)")
    print(var.pivot(index="factor", columns="lang", values="eta_sq").round(3), "\n")

    # ---- 3. model geometry, on STYLE rather than topic
    #
    # A raw centroid-vs-centroid cosine is uninformative here: sentence encoders pack
    # every vector into a narrow cone (anisotropy), so all six models come out at 0.99+
    # and the ranking among them is noise. Two corrections, both standard:
    #   (a) subtract each prompt cell's mean, so the topic -- which we already know
    #       dominates -- is projected out and only the model's own contribution remains;
    #   (b) the residuals are then mean-centred by construction, which removes the cone.
    # What the heatmap shows afterwards is: given the same prompt, do these two models
    # depart from the average answer in the same direction?
    sim, coords = {}, []
    for lang in ["en", "zh"]:
        m = idx["lang"] == lang
        d = idx[m].reset_index(drop=True)
        R = residualise(d, X[m.to_numpy()], ["issue_id", "template_id", "rep"])

        C = np.vstack([R[d.index[d["model"] == mdl].to_numpy()].mean(0) for mdl in MODEL_ORDER])
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        S = pd.DataFrame(C @ C.T, index=MODEL_ORDER, columns=MODEL_ORDER)
        sim[lang] = S

        D = np.clip(1 - S.values, 0, None)
        np.fill_diagonal(D, 0)
        xy = MDS(
            n_components=2, dissimilarity="precomputed", random_state=0,
            normalized_stress="auto", n_init=4,
        ).fit_transform(D)
        for mdl, (a, b) in zip(MODEL_ORDER, xy):
            coords.append({"lang": lang, "model": mdl, "origin": MODEL_ORIGIN[mdl],
                           "x": a, "y": b})
    coords = pd.DataFrame(coords)
    coords.to_csv(TAB / f"nb5b_model_mds_coords{suffix}.csv", index=False)
    pd.concat(sim, names=["lang", "model"]).to_csv(TAB / f"nb5b_model_similarity{suffix}.csv")
    print("MODEL STYLE SIMILARITY (prompt projected out; EN)")
    print(sim["en"].round(3))
    print(
        "  NB: residualising forces the six style vectors to sum to ~0, so the mean\n"
        "  off-diagonal is pinned near -1/(k-1) = -0.20. Read WHO IS NEAR WHOM, not the\n"
        "  absolute sign: a value above -0.20 is 'more alike than average', below it\n"
        "  'less alike'.\n"
    )

    # ---- 4. cross-lingual alignment, against a proper baseline
    #
    # "EN and ZH centroids for the same issue are 0.94 apart" is meaningless alone --
    # everything is ~0.94 apart in this space. The test that means something: is a
    # model's ZH answer on issue i closer to its own EN answer on issue i than to its
    # EN answer on some OTHER issue j? If yes, the space really is aligned across
    # languages and an EN/ZH content comparison is licensed.
    rows = []
    for mdl in MODEL_ORDER:
        g = idx[idx["model"] == mdl]
        issues = sorted(g["issue_id"].unique())
        cen = {}
        for lang in ["en", "zh"]:
            for iss in issues:
                sel = g.index[(g["lang"] == lang) & (g["issue_id"] == iss)].to_numpy()
                v = X[sel].mean(0)
                cen[(lang, iss)] = v / np.linalg.norm(v)
        E = np.vstack([cen[("en", i)] for i in issues])
        Z = np.vstack([cen[("zh", i)] for i in issues])
        M = Z @ E.T  # rows: ZH issue, cols: EN issue
        same = float(np.mean(np.diag(M)))
        off = float((M.sum() - np.trace(M)) / (M.size - len(issues)))
        # does the ZH vector retrieve its own EN issue out of all 30?
        top1 = float(np.mean(np.argmax(M, axis=1) == np.arange(len(issues))))
        rows.append({"model": mdl, "cos_same_issue": same, "cos_different_issue": off,
                     "gap": same - off, "retrieval_top1": top1,
                     "retrieval_chance": 1 / len(issues)})
    xl = pd.DataFrame(rows)
    xl.to_csv(TAB / f"nb5b_cross_lingual{suffix}.csv", index=False)
    print("CROSS-LINGUAL ALIGNMENT (ZH centroid vs EN centroid, per model)")
    print(xl.round(3).to_string(index=False), "\n")

    # ---- 5. origin discriminability from style alone
    disc = []
    for lang in ["en", "zh"]:
        m = idx["lang"] == lang
        d = idx[m].reset_index(drop=True)
        Xl = X[m.to_numpy()]
        # project out the prompt: what remains is how the model wrote it, not what about
        R = residualise(d, Xl, ["issue_id", "template_id", "rep"])

        y = d["origin"].to_numpy()
        groups = d["issue_id"].to_numpy()
        majority = pd.Series(y).value_counts(normalize=True).max()

        preds = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(R, y, groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(R[tr], y[tr])
            preds[te] = clf.predict(R[te])
        acc = float((preds == y).mean())
        # balanced accuracy: the origins are unbalanced (3 US / 2 CN / 1 EU models)
        bal = float(
            np.mean([ (preds[y == o] == o).mean() for o in np.unique(y) ])
        )
        pval = permutation_pvalue(R, y, groups, bal)
        disc.append({"lang": lang, "accuracy": acc, "balanced_accuracy": bal,
                     "majority_baseline": majority, "chance_balanced": 1 / len(np.unique(y)),
                     "perm_pvalue": pval})
    disc = pd.DataFrame(disc)
    disc.to_csv(TAB / f"nb5b_origin_discriminability{suffix}.csv", index=False)
    print("ORIGIN DISCRIMINABILITY (held-out issues; prompt content projected out)")
    print(disc.round(3).to_string(index=False), "\n")

    # ---- 5b. origin discriminability that must GENERALISE to an unseen model
    #
    # Section 5 holds out issues but keeps all six models in train and test, so a
    # classifier can win by memorising one model's personal fingerprint rather than by
    # learning what US or Chinese models share. The honest generalisation test holds out
    # a whole MODEL: train origin on the other five, predict the origin of a model the
    # classifier has never seen. Only a signal common to same-origin models can transfer.
    #
    # Caveat that shapes the design: the sample has one EU model (Mistral), so when it is
    # held out there is no EU example left to train on and EU is unpredictable by
    # construction. The three-class table therefore reports each held-out model on its own,
    # and the formal headline is the binary US-vs-CN test, where 3 US + 2 CN models let the
    # split be permuted at the model level for an exact null.
    per_model, binary = [], []
    for lang in ["en", "zh"]:
        m = idx["lang"] == lang
        d = idx[m].reset_index(drop=True)
        R = residualise(d, X[m.to_numpy()], ["issue_id", "template_id", "rep"])
        y = d["origin"].to_numpy()
        mdl_id = d["model"].to_numpy()

        # three-class leave-one-model-out, scored per held-out model
        preds = np.empty(len(y), dtype=object)
        for tr, te in LeaveOneGroupOut().split(R, y, groups=mdl_id):
            preds[te] = LogisticRegression(max_iter=2000, C=1.0).fit(R[tr], y[tr]).predict(R[te])
        for mdl in MODEL_ORDER:
            sel = mdl_id == mdl
            per_model.append({
                "lang": lang, "held_out_model": mdl, "origin": MODEL_ORIGIN[mdl],
                "n": int(sel.sum()), "correct_frac": float((preds[sel] == y[sel]).mean()),
                "note": "EU untrainable when held out (only 1 EU model)"
                        if MODEL_ORIGIN[mdl] == "EU" else "",
            })

        # binary US vs CN: drop the EU model, leave-one-model-out, model-level permutation
        keep = y != "EU"
        Rb, yb, gb = R[keep], y[keep], mdl_id[keep]
        us_cn_models = [mm for mm in MODEL_ORDER if MODEL_ORIGIN[mm] in ("US", "CN")]

        def lomo_balanced(labels: np.ndarray) -> float:
            """Balanced accuracy of a binary leave-one-model-out origin classifier."""
            p = np.empty(len(labels), dtype=object)
            for tr, te in LeaveOneGroupOut().split(Rb, labels, groups=gb):
                if len(np.unique(labels[tr])) < 2:      # held-out split leaves one class
                    p[te] = labels[tr][0]
                else:
                    p[te] = LogisticRegression(max_iter=2000, C=1.0).fit(Rb[tr], labels[tr]).predict(Rb[te])
            return float(np.mean([(p[labels == o] == o).mean() for o in np.unique(labels)]))

        obs = lomo_balanced(yb)
        # exact null: reassign the observed origin counts across the five US/CN models
        truth = {mm: MODEL_ORIGIN[mm] for mm in us_cn_models}
        n_cn = sum(v == "CN" for v in truth.values())
        perm_scores = []
        for cn_set in itertools.combinations(us_cn_models, n_cn):
            relabel = {mm: ("CN" if mm in cn_set else "US") for mm in us_cn_models}
            perm_scores.append(lomo_balanced(np.array([relabel[mm] for mm in gb], dtype=object)))
        perm_scores = np.array(perm_scores)
        binary.append({
            "lang": lang, "balanced_accuracy": obs, "chance": 0.5,
            "n_model_assignments": len(perm_scores),
            "perm_pvalue": float(np.mean(perm_scores >= obs)),
        })

    per_model = pd.DataFrame(per_model)
    binary = pd.DataFrame(binary)
    per_model.to_csv(TAB / f"nb5b_origin_generalization_by_model{suffix}.csv", index=False)
    binary.to_csv(TAB / f"nb5b_origin_generalization_us_cn{suffix}.csv", index=False)
    print("ORIGIN GENERALISATION (leave-one-MODEL-out; can origin be read on an unseen model?)")
    print("  per held-out model, share of its responses whose origin is predicted correctly:")
    print(per_model.pivot(index="held_out_model", columns="lang", values="correct_frac").round(3))
    print("  binary US-vs-CN (EU dropped), balanced accuracy vs 0.5 chance, model-level exact p:")
    print(binary.round(3).to_string(index=False), "\n")

    # ---- figures
    for lang in ["en", "zh"]:
        fig_tiers(tiers, lang, suffix)
    fig_variance(var, suffix)
    fig_model_heatmap(sim, suffix)
    fig_mds(coords, suffix)
    print(f"figures -> {FIG}/nb5b_*.pdf|png")
    print(f"tables  -> {TAB}/nb5b_*.csv")


# ============================================================================
# stylometry  (nb5h) -- WHY are the per-model voices so distinct, in plain features?
# ============================================================================
#
# similarity shows models are ~0.90-0.96 separable on embeddings but not what the
# signal is. This measures INTERPRETABLE surface style -- formatting, punctuation,
# sentence shape, hedging, function words -- and asks:
#
#   1. How much of the model-distinctiveness is explainable? A probe on these plain
#      features vs. the black-box embedding number.
#   2. What is each model's signature? (per-model z-scored feature means -> heatmap)
#   3. What do GPT-4o and Llama share? They are the one same-origin pair that clustered
#      in similarity; naming their shared habits explains the *only* origin-consistent
#      style.
#
# Verbosity itself is owned by nb6; length is kept here only as a control feature.
# Comparisons are WITHIN language (punctuation/word units are not commensurable across
# scripts). Refusals are dropped -- a refusal has its own boilerplate style.

# hedging / modal lexicons (word-boundary for EN; substring for ZH)
HEDGE_EN = (r"\b(may|might|could|should|would|can|possibly|perhaps|arguably|likely|"
            r"seems?|appears?|suggests?|tends? to|potentially|relatively|somewhat)\b")
HEDGE_ZH = ["可能", "也许", "或许", "应该", "往往", "通常", "似乎", "倾向", "大概", "在一定程度"]
# classic style function words (content-free); ZH particles/connectives via jieba
FUNC_EN = ["the", "of", "to", "and", "a", "in", "that", "is", "it", "for", "as",
           "this", "however", "therefore", "while", "although", "thus", "moreover"]
FUNC_ZH = ["的", "了", "是", "和", "在", "也", "都", "而", "但", "因此", "然而", "并",
           "等", "这", "对", "为", "与", "及"]

# the curated interpretable core shown in the heatmap (language-common, no function words)
CORE = ["chars", "words_per_sent", "n_paragraphs", "bullet_line_frac", "has_headers",
        "bold_per_1k", "emdash_per_1k", "semicolon_per_1k", "colon_per_1k",
        "question_per_1k", "paren_per_1k", "hedge_rate"]
CORE_LABEL = {
    "chars": "length (chars)", "words_per_sent": "words/sentence",
    "n_paragraphs": "paragraphs", "bullet_line_frac": "bullet lines",
    "has_headers": "uses headers", "bold_per_1k": "bold /1k",
    "emdash_per_1k": "em-dash /1k", "semicolon_per_1k": "semicolon /1k",
    "colon_per_1k": "colon /1k", "question_per_1k": "question /1k",
    "paren_per_1k": "parens /1k", "hedge_rate": "hedging",
}

SENT_SPLIT = re.compile(r"[.!?。！？\n]+")
BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M)
HEADER = re.compile(r"^\s*(#{1,6}\s|\*\*[^*]+\*\*\s*:?\s*$)", re.M)
BOLD = re.compile(r"\*\*[^*]+\*\*")


def features(text: str, lang: str) -> dict:
    t = str(text)
    n_chars = max(len(t), 1)
    if lang == "en":
        words = re.findall(r"[A-Za-z']+", t.lower())
    else:
        import jieba
        words = [w for w in jieba.cut(t) if w.strip()]
    nw = max(len(words), 1)
    ns = max(len([s for s in SENT_SPLIT.split(t) if s.strip()]), 1)
    nl = max(len([ln for ln in t.split("\n") if ln.strip()]), 1)
    per1k = 1000.0 / n_chars

    f = {
        "chars": n_chars,
        "words_per_sent": nw / ns,
        "bullet_line_frac": len(BULLET.findall(t)) / nl,
        "has_headers": int(bool(HEADER.search(t))),
        "bold_per_1k": len(BOLD.findall(t)) * per1k,
        "n_paragraphs": t.count("\n\n") + 1,
        "comma_per_1k": (t.count(",") + t.count("，")) * per1k,
        "semicolon_per_1k": (t.count(";") + t.count("；")) * per1k,
        "colon_per_1k": (t.count(":") + t.count("：")) * per1k,
        "emdash_per_1k": (t.count("—") + t.count("--")) * per1k,
        "question_per_1k": (t.count("?") + t.count("？")) * per1k,
        "paren_per_1k": (t.count("(") + t.count("（")) * per1k,
        "hedge_rate": (len(re.findall(HEDGE_EN, t.lower())) if lang == "en"
                       else sum(t.count(h) for h in HEDGE_ZH)) / nw,
    }
    fw = FUNC_EN if lang == "en" else FUNC_ZH
    if lang == "en":
        wc = pd.Series(words).value_counts()
        for w in fw:
            f[f"fw_{w}"] = wc.get(w, 0) / nw
    else:
        for w in fw:
            f[f"fw_{w}"] = t.count(w) / nw
    return f


def fig_signatures(z_by_lang: dict, suffix: str = "") -> None:
    """Heatmap: per-model z-scored feature means, models x curated core features.

    Diverging scale centered at the cross-model mean, so + = high for this model,
    - = low. Rows ordered by origin so any origin block would be visible if it existed.
    """
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#c33534", "#ef9d9c", "#f4f4f2", "#86b6ef", "#1c5cab"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    lim = max(np.nanmax(np.abs(z[CORE].values)) for z in z_by_lang.values())
    lim = min(lim, 3.0)
    labels = [CORE_LABEL[c] for c in CORE]
    for ax, lang in zip(axes, ["en", "zh"]):
        Z = z_by_lang[lang].loc[MODEL_ORDER, CORE]
        im = ax.imshow(Z.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(CORE)), labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(MODEL_ORDER)),
                      [f"{m} ({MODEL_ORIGIN[m]})" for m in MODEL_ORDER], fontsize=8)
        for i in range(len(MODEL_ORDER)):
            for j in range(len(CORE)):
                v = Z.values[i, j]
                if abs(v) > 0.9:
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > 0.62 * lim else INK)
        ax.set_title(lang.upper(), color=INK)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Each model has a distinctive house style; it does not line up with origin",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.7, label="z vs. cross-model mean (+ high / − low)")
    save(fig, FIG / f"nb5h_style_signatures{suffix}")


def run_stylometry() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    resp = load_responses()
    resp["origin"] = resp["model"].map(MODEL_ORIGIN)
    # drop refusals: refusal boilerplate has its own style and would leak in
    master = load_master()
    refused = set(map(tuple,
                      master.loc[master["is_refusal"] == 1,
                                 ["rep", "model", "prompt_id"]].values))
    keys = list(map(tuple, resp[["rep", "model", "prompt_id"]].values))
    resp = resp[[k not in refused for k in keys]].reset_index(drop=True)
    print(f"{len(resp):,} non-refusal responses\n")

    z_by_lang, probe_rows = {}, []
    for lang in ["en", "zh"]:
        d = resp[resp["lang"] == lang].reset_index(drop=True)
        F = pd.DataFrame([features(t, lang) for t in d["response"]]).fillna(0.0)
        cols = list(F.columns)
        print(f"===== {lang.upper()}  (n={len(d)}, {len(cols)} features) =====")

        # (1) probe: model identity from interpretable features, held-out issues
        y, groups = d["model"].values, d["issue_id"].values
        pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(5).split(F.values, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            clf.fit(F.values[tr], y[tr])
            pred[te] = clf.predict(F.values[te])
        bal = float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))
        print(f"[probe] model from interpretable features: balanced acc {bal:.3f} "
              f"(chance {1/6:.3f})")
        probe_rows.append({"lang": lang, "n": len(d), "n_features": len(cols),
                           "balanced_acc": bal, "chance": 1 / 6})

        # (2) per-model signature: z-scored feature means across models
        g = F.assign(model=d["model"].values).groupby("model")[cols].mean()
        z = ((g - g.mean()) / g.std(ddof=0)).loc[MODEL_ORDER]
        z_by_lang[lang] = z
        g.loc[MODEL_ORDER].to_csv(TAB / f"nb5h_style_means_{lang}.csv")
        z.to_csv(TAB / f"nb5h_style_z_{lang}.csv")

        # (3) GPT-4o + Llama shared style vs the other four
        rest = [m for m in MODEL_ORDER if m not in ("gpt4o", "llama")]
        gap = (z.loc[["gpt4o", "llama"]].mean() - z.loc[rest].mean()).sort_values()
        contrast = pd.DataFrame({"z_gap_pair_minus_rest": gap})
        contrast.to_csv(TAB / f"nb5h_gpt4o_llama_shared_{lang}.csv")
        print("  GPT-4o & Llama share (biggest z-gaps vs the other four):")
        for c in list(gap.index[:3]) + list(gap.index[-3:]):
            print(f"    {c:18s} gap {gap[c]:+.2f}")
        print()

    pd.DataFrame(probe_rows).to_csv(TAB / "nb5h_style_probe.csv", index=False)
    fig_signatures(z_by_lang)
    print(f"figure -> {FIG}/nb5h_style_signatures.pdf|png")
    print(f"tables -> {TAB}/nb5h_style_*.csv, nb5h_gpt4o_llama_shared_*.csv")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RQ5 response embeddings: build (embed), probe (similarity), "
                     "or a text-only style probe (stylometry).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser(
        "embed", help="nb5a: embed every response into a shared vector space",
    )
    p_embed.add_argument(
        "--sample",
        action="store_true",
        help="sampled run: N templates per category, all issues/models/langs/reps",
    )
    p_embed.add_argument(
        "--templates-per-category", type=int, default=SAMPLE_TEMPLATES_PER_CATEGORY,
        help="how many templates per category to keep in --sample mode (default: 1)",
    )
    p_embed.add_argument(
        "--model", choices=sorted(MODEL_CONFIGS), default="e5",
        help="encoder to embed with (default: e5)",
    )

    p_sim = sub.add_parser(
        "similarity", help="nb5b: similarity tiers / variance / geometry / discriminability",
    )
    p_sim.add_argument("--full", action="store_true", help="use the full-run embeddings")
    p_sim.add_argument(
        "--model", default="e5",
        help="which embed encoder's embeddings to read (e5 | labse | bge-m3)",
    )
    p_sim.add_argument(
        "--sample-tag", default="sample",
        help="sample-size dir written by embed, e.g. 'sample' (n=1/category) or "
             "'sample20' (n=5/category) -- must match what embed was run with",
    )

    sub.add_parser("stylometry", help="nb5h: interpretable surface-style features")

    return ap


def run_embeddings() -> None:
    args = build_parser().parse_args()
    if args.command == "embed":
        run_embed(args)
    elif args.command == "similarity":
        run_similarity(args)
    elif args.command == "stylometry":
        run_stylometry()





import itertools
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master, load_responses  # noqa: E402


TAB = Path("evaluation/analysis/outputs/tables")
MODELS = ["claude", "gpt4o", "llama", "mistral", "deepseek", "qwen"]

# A response counts once per construct, regardless of repeated mentions. The separate
# occurrence count is retained as a diagnostic only because regex alternatives can overlap
# conceptually even when the literal matches do not overlap.
PATTERNS = [
    # Institutions, standards, and legal references
    ("institution/legal", "NIST", r"\bNIST\b|National Institute of Standards and Technology"),
    ("institution/legal", "MIT Media Lab", r"\bMIT Media Lab\b"),
    ("institution/legal", "ACLU", r"\bACLU\b|American Civil Liberties Union"),
    ("institution/legal", "Clearview AI", r"\bClearview AI\b"),
    ("institution/legal", "Carpenter case", r"\bCarpenter(?: v\. United States)?\b"),
    ("institution/legal", "Fourth Amendment", r"\bFourth Amendment\b"),
    ("institution/legal", "First Amendment", r"\bFirst Amendment\b"),
    ("institution/legal", "EU AI Act", r"\b(?:EU|European Union) AI Act\b|\bArtificial Intelligence Act\b"),
    ("institution/legal", "GDPR", r"\bGDPR\b|General Data Protection Regulation"),
    ("institution/legal", "FBI", r"\bFBI\b|Federal Bureau of Investigation"),
    ("institution/legal", "NSA", r"\bNSA\b|National Security Agency"),
    ("institution/legal", "Georgetown Center on Privacy & Technology", r"Georgetown Law(?:'s)? (?:Center|Centre) on Privacy(?:\s*(?:&|and)\s*Technology)?|Center on Privacy\s*(?:&|and)\s*Technology|Perpetual Line[- ]Up"),
    ("institution/legal", "Electronic Frontier Foundation", r"\bEFF\b|Electronic Frontier Foundation"),
    ("institution/legal", "Amnesty International", r"\bAmnesty International\b"),
    ("institution/legal", "Human Rights Watch", r"\bHuman Rights Watch\b"),
    ("institution/legal", "Big Brother Watch", r"\bBig Brother Watch\b"),
    ("institution/legal", "Algorithmic Justice League", r"\bAlgorithmic Justice League\b"),
    ("institution/legal", "STOP", r"Surveillance Technology Oversight Project|(?<![A-Z])STOP(?![A-Z])"),
    ("institution/legal", "GAO", r"\bGAO\b|Government Accountability Office"),
    ("institution/legal", "DHS", r"\bDHS\b|Department of Homeland Security"),
    ("institution/legal", "CBP", r"\bCBP\b|Customs and Border Protection"),
    ("institution/legal", "ICE", r"\bICE\b|Immigration and Customs Enforcement"),
    ("institution/legal", "NYPD", r"\bNYPD\b|New York Police Department"),
    ("institution/legal", "LAPD", r"\bLAPD\b|Los Angeles Police Department"),
    ("institution/legal", "Metropolitan Police", r"\bMetropolitan Police(?: Service)?\b|\bLondon Met\b"),
    ("institution/legal", "South Wales Police", r"\bSouth Wales Police\b"),
    ("institution/legal", "Delhi Police", r"\bDelhi Police\b"),
    ("institution/legal", "FRVT", r"\bFRVT\b|Face Recognition Vendor Test(?:ing)?"),
    ("institution/legal", "ECHR", r"\bECHR\b|European Convention on Human Rights"),
    ("institution/legal", "ICO", r"\bICO\b|Information Commissioner(?:’s|'s)? Office"),
    ("institution/legal", "Pew Research", r"\bPew Research(?: Center)?\b"),
    ("institution/legal", "BIPA", r"\bBIPA\b|Biometric Information Privacy Act"),
    ("institution/legal", "Facial Recognition Moratorium Act", r"Facial Recognition and Biometric Technology Moratorium Act|Recognition and Biometric Technology Moratorium Act"),
    ("institution/legal", "US Constitution", r"\bU\.?S\.? Constitution\b|\bUnited States Constitution\b|\bthe Constitution\b|\bconstitutional rights?\b"),
    ("institution/legal", "Bill of Rights", r"\bBill of Rights\b"),
    ("institution/legal", "Fourteenth Amendment", r"\bFourteenth Amendment\b|\b14th Amendment\b"),
    ("institution/legal", "Supreme Court", r"\b(?:U\.?S\.? )?Supreme Court\b"),
    ("institution/legal", "due process", r"\bdue process\b"),
    ("institution/legal", "judicial warrant", r"\bjudicial warrants?\b|\bsearch warrants?\b|\bwarrant requirement\b|\brequir(?:e|ing) (?:a )?warrant\b"),
    ("institution/legal", "Katz case", r"\bKatz v\.? (?:the )?United States\b"),
    ("institution/legal", "United States v. Jones", r"\bUnited States v\.? Jones\b|\bU\.?S\.? v\.? Jones\b"),

    # Named people and cases
    ("person/case", "Robert Williams", r"\bRobert(?: Julian-Borchak)? Williams\b"),
    ("person/case", "Nijeer Parks", r"\bNijeer Parks\b"),
    ("person/case", "Michael Oliver", r"\bMichael Oliver\b"),
    ("person/case", "Joy Buolamwini", r"\bJoy Buolamwini\b"),
    ("person/case", "Timnit Gebru", r"\bTimnit Gebru\b"),
    ("person/case", "Edward Snowden", r"\b(?:Edward )?Snowden\b"),
    ("person/case", "Porcha Woodruff", r"\bPorcha Woodruff\b"),
    ("person/case", "Randal Reid", r"\bRand(?:a|o)l(?: Quran)? Reid\b"),
    ("person/case", "Alonzo Sawyer", r"\bAlonzo Sawyer\b"),
    ("person/case", "Clare Garvie", r"\bClare Garvie\b"),
    ("person/case", "Albert Fox Cahn", r"\bAlbert Fox Cahn\b"),

    # Cultural shorthand
    ("cultural reference", "Big Brother / Orwell / 1984", r"\bBig Brother\b|(?:George )?Orwell(?:ian)?|\b1984\b"),
    ("cultural reference", "Minority Report", r"\bMinority Report\b"),
    ("cultural reference", "Gender Shades", r"\bGender Shades\b"),
    ("cultural reference", "panopticon", r"\bpanopticon\b|\bpanoptic(?:on| gaze)?\b"),

    # China-specific sensitive-topic repertoire
    ("China-sensitive", "Xinjiang", r"\bXinjiang\b"),
    ("China-sensitive", "Uyghur/Uighur", r"\bU(?:y|i)ghurs?\b|\bUygurs?\b"),
    ("China-sensitive", "Skynet/Sharp Eyes", r"\bSkynet\b|\bSharp Eyes\b"),
    ("China-sensitive", "social credit system", r"\bsocial credit system\b"),
    ("China-sensitive", "Hong Kong", r"\bHong Kong\b"),

    # Vendors and products
    ("vendor/product", "Amazon Rekognition", r"\bAmazon(?:'s)? Rekognition\b|\bRekognition\b"),
    ("vendor/product", "NEC", r"\bNEC\b"),
    ("vendor/product", "IBM", r"\bIBM\b"),
    ("vendor/product", "Palantir", r"\bPalantir\b"),
    ("technical detail", "Haar cascades", r"\bHaar cascades?\b"),
    ("technical detail", "HOG", r"\bHOG\b|Histogram of Oriented Gradients"),
    ("technical detail", "SSD detector", r"\bSSD\b|Single Shot MultiBox Detector"),

    # Events and recurring exemplars
    ("event/example", "Boston Marathon", r"\bBoston Marathon\b"),
    ("event/example", "Manchester Arena", r"\bManchester Arena\b"),
    ("event/example", "Notting Hill Carnival", r"\bNotting Hill Carnival\b"),
    ("event/example", "Black Lives Matter", r"\bBlack Lives Matter\b|\bBLM\b"),
    ("event/example", "AMBER Alert", r"\bAMBER Alerts?\b"),

    # U.S.-political voice, especially relevant to politician/speech templates
    ("US political voice", "senator/US Senate persona", r"\bU\.?S\.? Senators?\b|\bUnited States Senators?\b|\bSenator [A-Z][a-z]+\b|\bCandidate for (?:the )?U\.?S\.? Senate\b"),
    ("US political voice", "Congress", r"\bCongress\b|\bcongressional\b|\bmembers? of Congress\b"),
    ("US political voice", "patriot/patriotic", r"\bpatriots?\b|\bpatriotic\b"),
    ("US political voice", "God bless", r"\bGod bless\b"),
    ("US political voice", "not the America", r"\bnot (?:the )?America\b|\bThis is not America\b"),

    # Jurisdictional reference: broad mention indicators, not endorsement measures
    ("jurisdiction", "United States reference", r"\bUnited States\b|\bU\.S\.?(?=\s|,|\.|$)|\bAmerican\b|\bAmerica\b"),
    ("jurisdiction", "United Kingdom reference", r"\bUnited Kingdom\b|\bU\.K\.?(?=\s|,|\.|$)|\bBritain\b|\bBritish\b"),
    ("jurisdiction", "China reference", r"\bChina\b|\bChinese\b|\bXinjiang\b|\bUyghurs?\b|\bUighurs?\b"),
    ("jurisdiction", "EU/Europe reference", r"\bEuropean Union\b|\bEU\b|\bEuropean\b"),

    # Recurrent substantive arguments
    ("argument/theme", "racial bias/discrimination", r"\bracial(?:ly)?\b|\brace\b|\bpeople of colo[u]?r\b|\bBlack (?:people|men|women|individuals|faces|Americans?)\b|\bdiscriminat(?:e|ion|ory)\b"),
    ("argument/theme", "gender bias", r"\bgender (?:bias|disparit|inequal)|\bwomen\b.{0,80}\b(?:error|misidentif|accurac|bias)"),
    ("argument/theme", "wrongful arrest/misidentification", r"\bwrongful(?:ly)? arrest|\bfalse (?:match|positive)|\bmisidentif(?:y|ied|ication)"),
    ("argument/theme", "civil liberties", r"\bcivil libert(?:y|ies)\b"),
    ("argument/theme", "mass surveillance", r"\bmass surveillance\b"),
    ("argument/theme", "chilling effect", r"\bchilling effect\b|\bchill(?:s|ed|ing) (?:speech|protest|expression|participation)"),
    ("argument/theme", "public safety", r"\bpublic safety\b"),
    ("argument/theme", "democracy/democratic society", r"\bdemocratic societ(?:y|ies)\b|\bdemocrac(?:y|ies)\b"),
    ("argument/theme", "authoritarianism", r"\bauthoritarian(?:ism| regime| government| state| use| tendencies)?\b"),
    ("argument/theme", "protest/dissent", r"\bprotests?\b|\bdemonstrations?\b|\bdissent(?:ers?)?\b"),
]

# Exact acronym tokens for the technology itself. Negative lookarounds prevent FRT from
# being counted inside RT-FRT. These are case-sensitive because ordinary "fr" and "rfr"
# character sequences are not acronyms.
ACRONYM_PATTERNS = [
    ("RT-FRT", r"(?<![A-Z-])RT-FRT(?![A-Z-])"),
    ("RTFR", r"(?<![A-Z-])RTFR(?![A-Z-])"),
    ("RFRT", r"(?<![A-Z-])RFRT(?![A-Z-])"),
    ("RFR", r"(?<![A-Z-])RFR(?![A-Z-])"),
    ("FRT", r"(?<![A-Z-])FRT(?![A-Z-])"),
    ("LFR", r"(?<![A-Z-])LFR(?![A-Z-])"),
    ("AFR", r"(?<![A-Z-])AFR(?![A-Z-])"),
    ("FR", r"(?<![A-Z-])FR(?![A-Z-])"),
]

CITIES = [
    "San Francisco", "Boston", "Portland", "Detroit", "London", "New Orleans",
    "Baltimore", "New York", "Chicago", "Washington DC", "Los Angeles", "Seattle",
]


def matches(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, str(text), flags=re.IGNORECASE | re.DOTALL)


def shannon_entropy(counts: list[int], normalized: bool = False) -> float:
    values = np.asarray([x for x in counts if x > 0], dtype=float)
    if len(values) <= 1:
        return 0.0
    p = values / values.sum()
    entropy = float(-(p * np.log(p)).sum())
    return entropy / np.log(len(values)) if normalized else entropy


def run_b7() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    responses = load_responses()
    sample = responses[
        responses["rep"].eq("real_r2")
        & responses["issue_id"].eq("B7")
        & responses["lang"].eq("en")
    ].copy()
    if len(sample) != 600 or sample.groupby("model").size().ne(100).any():
        raise AssertionError("Expected 100 matched B7 English responses for each model")

    count_rows = []
    for model in MODELS:
        group = sample[sample["model"].eq(model)]
        for category, label, pattern in PATTERNS:
            counts = group["response"].map(lambda text: len(matches(text, pattern)))
            count_rows.append({
                "category": category,
                "construct": label,
                "pattern": pattern,
                "model": model,
                "n_responses": len(group),
                "responses_with_mention": int(counts.gt(0).sum()),
                "response_share": float(counts.gt(0).mean()),
                "literal_occurrences": int(counts.sum()),
            })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(TAB / "nb5l_b7_repertoire_counts.csv", index=False)

    # Technology-acronym repertoire and consistency.
    acronym_rows, acronym_stability_rows = [], []
    for model in MODELS:
        group = sample[sample["model"].eq(model)]
        dominant = Counter()
        multiple_variants = 0
        using_any = 0
        for response_row in group.itertuples(index=False):
            text = response_row.response
            per_response = {}
            for label, pattern in ACRONYM_PATTERNS:
                n = len(re.findall(pattern, str(text)))
                per_response[label] = n
                if n:
                    acronym_rows.append({
                        "model": model, "template_id": response_row.template_id,
                        "acronym": label, "present": 1, "literal_occurrences": n,
                    })
            present = [label for label, n in per_response.items() if n]
            if present:
                using_any += 1
                if len(present) > 1:
                    multiple_variants += 1
                # Stable tie-breaking follows the declared acronym order.
                top = max(ACRONYM_PATTERNS, key=lambda item: per_response[item[0]])[0]
                dominant[top] += 1
        dominant_counts = list(dominant.values())
        acronym_stability_rows.append({
            "model": model,
            "n_responses": len(group),
            "responses_using_any_acronym": using_any,
            "responses_using_multiple_variants": multiple_variants,
            "multiple_variant_share_of_acronym_responses": multiple_variants / using_any if using_any else np.nan,
            "distinct_dominant_variants": len(dominant),
            "dominant_variant": dominant.most_common(1)[0][0] if dominant else None,
            "dominant_variant_share": dominant.most_common(1)[0][1] / using_any if dominant else np.nan,
            "dominant_variant_entropy_normalized": shannon_entropy(dominant_counts, normalized=True),
        })
    acronym_raw = pd.DataFrame(acronym_rows)
    if not acronym_raw.empty:
        acronym_counts = acronym_raw.groupby(["model", "acronym"], as_index=False).agg(
            responses_with_mention=("present", "sum"),
            literal_occurrences=("literal_occurrences", "sum"),
        )
        acronym_counts["response_share"] = acronym_counts["responses_with_mention"] / 100
    else:
        acronym_counts = pd.DataFrame()
    acronym_counts.to_csv(TAB / "nb5l_b7_acronym_counts.csv", index=False)
    pd.DataFrame(acronym_stability_rows).to_csv(
        TAB / "nb5l_b7_acronym_stability.csv", index=False)

    # Stability of the curated named-reference repertoire. This is a transparent
    # dictionary-based measure, not open-ended named-entity recognition.
    reference_patterns = [
        (category, label, pattern) for category, label, pattern in PATTERNS
        if category in {
            "institution/legal", "person/case", "cultural reference",
            "China-sensitive", "vendor/product", "technical detail", "event/example", "US political voice",
        }
    ]
    repertoire_rows = []
    for model in MODELS:
        group = sample[sample["model"].eq(model)]
        response_sets = []
        construct_df = Counter()
        total_words = int(group["response"].str.split().str.len().sum())
        for text in group["response"]:
            present = {
                label for _, label, pattern in reference_patterns if matches(text, pattern)
            }
            response_sets.append(present)
            construct_df.update(present)
        pairwise = []
        for left, right in itertools.combinations(response_sets, 2):
            union = left | right
            if union:
                pairwise.append(len(left & right) / len(union))
        total_mentions = sum(construct_df.values())
        top5_mentions = sum(n for _, n in construct_df.most_common(5))
        repertoire_rows.append({
            "model": model,
            "n_responses": len(group),
            "total_words": total_words,
            "curated_response_construct_mentions": total_mentions,
            "construct_mentions_per_1000_words": total_mentions / total_words * 1000,
            "mean_constructs_per_response": total_mentions / len(group),
            "distinct_constructs_used": len(construct_df),
            "constructs_used_once": sum(n == 1 for n in construct_df.values()),
            "constructs_used_in_5plus_responses": sum(n >= 5 for n in construct_df.values()),
            "top5_construct_concentration": top5_mentions / total_mentions if total_mentions else np.nan,
            "construct_distribution_entropy_normalized": shannon_entropy(list(construct_df.values()), normalized=True),
            "mean_pairwise_reference_jaccard": float(np.mean(pairwise)) if pairwise else np.nan,
            "top5_constructs": " | ".join(f"{label} ({n})" for label, n in construct_df.most_common(5)),
        })
    pd.DataFrame(repertoire_rows).to_csv(
        TAB / "nb5l_b7_repertoire_stability.csv", index=False)

    city_rows, combo_rows = [], []
    for model in MODELS:
        group = sample[sample["model"].eq(model)]
        combos = Counter()
        for text in group["response"]:
            present = []
            for city in CITIES:
                if city == "Washington DC":
                    pattern = r"\bWashington,?\s+D\.?C\.?\b|\bDistrict of Columbia\b"
                else:
                    pattern = r"\b" + re.escape(city) + r"(?: City)?\b"
                n = len(matches(text, pattern))
                if n:
                    present.append(city)
                city_rows.append({
                    "model": model, "city": city, "present": int(n > 0),
                    "literal_occurrences": n,
                })
            if len(present) >= 2:
                combos[" | ".join(present)] += 1
        for combination, n in combos.most_common():
            combo_rows.append({
                "model": model, "city_combination": combination,
                "responses_with_combination": n,
            })

    city_raw = pd.DataFrame(city_rows)
    city_counts = city_raw.groupby(["model", "city"], as_index=False).agg(
        responses_with_mention=("present", "sum"),
        literal_occurrences=("literal_occurrences", "sum"),
    )
    city_counts["response_share"] = city_counts["responses_with_mention"] / 100
    city_counts.to_csv(TAB / "nb5l_b7_city_counts.csv", index=False)
    pd.DataFrame(combo_rows).to_csv(
        TAB / "nb5l_b7_city_combinations.csv", index=False)

    # Stance variation within this diagnostic repetition and exact-template stability
    # across all three repetitions. These distinguish lexical repertoire changes from
    # positional instability.
    master = load_master()
    b7 = master[
        master["issue_id"].eq("B7") & master["lang"].eq("en")
        & master["model"].isin(MODELS) & master["stance_num"].notna()
    ].copy()
    stance_rows = []
    for model in MODELS:
        values = b7[b7["model"].eq(model) & b7["rep"].eq("real_r2")]["stance_num"]
        counts5 = [int(values.eq(score).sum()) for score in range(1, 6)]
        stance_rows.append({
            "model": model, "n": len(values), "mean_stance": values.mean(),
            "stance_sd": values.std(ddof=0),
            "stance_entropy_normalized": shannon_entropy(counts5, normalized=True),
            **{f"stance_{score}": counts5[score - 1] for score in range(1, 6)},
        })
    pd.DataFrame(stance_rows).to_csv(
        TAB / "nb5l_b7_stance_distribution.csv", index=False)

    rep_rows = []
    for model in MODELS:
        per_template = []
        for _, group in b7[b7["model"].eq(model)].groupby("template_id"):
            values = group.set_index("rep")["stance_num"].reindex(["real_r1", "real_r2", "real_r3"]).dropna()
            if len(values) < 2:
                continue
            per_template.append({
                "all_same": values.nunique() == 1,
                "range": values.max() - values.min(),
                "sd": values.std(ddof=0),
            })
        t = pd.DataFrame(per_template)
        rep_rows.append({
            "model": model, "templates_with_2plus_reps": len(t),
            "all_available_reps_same_share": t["all_same"].mean(),
            "mean_within_template_range": t["range"].mean(),
            "range_2plus_share": t["range"].ge(2).mean(),
            "range_3plus_share": t["range"].ge(3).mean(),
            "mean_within_template_sd": t["sd"].mean(),
        })
    pd.DataFrame(rep_rows).to_csv(
        TAB / "nb5l_b7_stance_rep_stability.csv", index=False)

    print("Recurring-reference counts ->", TAB / "nb5l_b7_repertoire_counts.csv")
    print("City counts ->", TAB / "nb5l_b7_city_counts.csv")
    print("City combinations ->", TAB / "nb5l_b7_city_combinations.csv")
    print("Acronym counts/stability ->", TAB / "nb5l_b7_acronym_*.csv")
    print("Repertoire stability ->", TAB / "nb5l_b7_repertoire_stability.csv")
    print("Stance stability ->", TAB / "nb5l_b7_stance_*.csv")




if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "b7":
        sys.argv = [sys.argv[0]]
        run_b7()
    else:
        run_embeddings()
