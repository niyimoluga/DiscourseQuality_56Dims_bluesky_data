"""
Phase 1 — Data Analysis for Bluesky Discourse Quality Dataset

Produces:
  phase1_artifacts/
    ├── 01_dataset_fingerprint.json
    ├── 02_per_variable_zero_inflation.csv
    ├── 03_per_labeller_stats.csv
    ├── 04_krippendorff_alpha.csv
    ├── 05_correlation_matrix_avg.npy
    ├── 05_dendrogram_by_group.png
    ├── 05_dendrogram_by_polarity.png
    ├── 05_dendrogram_per_labeller.png
    ├── 05_macro_groups.json
    ├── 06_author_distribution.csv
    ├── 06_author_zipf.png
    ├── 07_topic_clusters.csv
    ├── 07_topic_umap.png
    ├── 08_length_distributions.csv
    ├── 08_length_distributions.png
    ├── 09_context_availability.csv
    ├── 10_missing_score_cells.csv
    └── phase1_report.md

Runtime: ~1-2 hours CPU
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# Optional heavy imports — only used if available
try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    print("⚠️  krippendorff not installed — α computation will be skipped. "
          "Install with: pip install krippendorff")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import HDBSCAN
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn / scipy missing — dendrogram and clustering skipped.")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  umap-learn not installed — UMAP plots will be skipped. "
          "Install with: pip install umap-learn")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("⚠️  sentence-transformers not installed — embedding skipped. "
          "Install with: pip install sentence-transformers")


# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
ARTIFACTS = Path("./phase1_artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)

LABELLERS = ["qwen3.5", "gemma4", "mistral3.2", "gptoss", "granite4.2"]

# Codebook variable IDs — 56 total
VAR_IDS = [
    # Reasoning and argumentation (17)
    "D1a", "D1b", "D3+", "D3\u2212", "D4+", "D4\u2212", "D5+", "D5\u2212",
    "D6+", "D6\u2212", "D7+", "D7\u2212", "F7+", "F7\u2212", "D8", "D9", "D11",
    # Conflict and identity (12)
    "A1", "A2", "A3+", "A3\u2212", "A4", "A5+", "A5\u2212", "A6", "A11",
    "A12", "A13", "F9",
    # Evidence and epistemic stance (7)
    "E1+", "E1\u2212", "E2", "E3", "E4+", "E4\u2212", "E7",
    # Interaction and inclusion (8)
    "I1+", "I1\u2212", "I3+", "I3\u2212", "I4", "I8", "I9", "F13",
    # Democratic preconditions (12)
    "P1", "P2", "P8", "P3", "P4", "P5+", "P5\u2212", "P6+", "P6\u2212",
    "P7", "P10", "P11",
]
assert len(VAR_IDS) == 56, f"Expected 56 vars, got {len(VAR_IDS)}"

# Theoretical grouping (for dendrogram coloring)
VAR_TO_GROUP = {}
for v in ["D1a","D1b","D3+","D3\u2212","D4+","D4\u2212","D5+","D5\u2212",
          "D6+","D6\u2212","D7+","D7\u2212","F7+","F7\u2212","D8","D9","D11"]:
    VAR_TO_GROUP[v] = "Reasoning"
for v in ["A1","A2","A3+","A3\u2212","A4","A5+","A5\u2212","A6","A11","A12","A13","F9"]:
    VAR_TO_GROUP[v] = "Conflict"
for v in ["E1+","E1\u2212","E2","E3","E4+","E4\u2212","E7"]:
    VAR_TO_GROUP[v] = "Epistemic"
for v in ["I1+","I1\u2212","I3+","I3\u2212","I4","I8","I9","F13"]:
    VAR_TO_GROUP[v] = "Interaction"
for v in ["P1","P2","P8","P3","P4","P5+","P5\u2212","P6+","P6\u2212","P7","P10","P11"]:
    VAR_TO_GROUP[v] = "Preconditions"

GROUP_COLORS = {
    "Reasoning":     "#1f77b4",
    "Conflict":      "#d62728",
    "Epistemic":     "#2ca02c",
    "Interaction":   "#ff7f0e",
    "Preconditions": "#9467bd",
}

# Polarity
POLARITY = {v: ("negative" if v.endswith("\u2212") else "positive")
            for v in VAR_IDS}

POLARITY_COLORS = {"positive": "#2ca02c", "negative": "#d62728"}


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def log(msg):
    print(f"[Phase 1] {msg}", flush=True)


def extract_author_did(comment_id):
    """comment_id = 'at://did:plc:XXXX/app.bsky.feed.post/YYYY' → 'did:plc:XXXX'."""
    try:
        return comment_id.replace("at://", "").split("/")[0]
    except Exception:
        return None


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════
# 1.0  LOAD AND VERIFY
# ══════════════════════════════════════════════════════════════════════════
def load_data():
    log("Loading dataset from HuggingFace...")
    ds = load_dataset(
        "idealab-cs2/bluesky-discourse-quality",
        data_files="bluesky-discourse-quality.parquet",
        token=True,
    )
    df = ds["train"].to_pandas()
    log(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def verify_dataset(df):
    log("Verifying dataset structure...")
    report = {}

    report["n_rows"] = int(len(df))
    report["n_columns"] = int(len(df.columns))
    report["expected_n_rows"] = 467125

    # Context columns
    context_cols = ["comment_id", "created_utc", "root_text",
                    "parent_text", "comment_text", "comment_len"]
    report["context_cols_present"] = all(c in df.columns for c in context_cols)

    # Check for expected nulls
    report["null_comment_id"] = int(df["comment_id"].isna().sum())
    report["null_comment_text"] = int(df["comment_text"].isna().sum())
    report["null_parent_text"] = int(df["parent_text"].isna().sum())
    report["null_root_text"] = int(df["root_text"].isna().sum())

    # Time range
    report["created_utc_min"] = int(df["created_utc"].min())
    report["created_utc_max"] = int(df["created_utc"].max())
    report["time_range_hours"] = (report["created_utc_max"] - report["created_utc_min"]) / 3600

    # Score column integrity
    expected_score_cols = set()
    for var in VAR_IDS:
        for lab in LABELLERS:
            expected_score_cols.add(f"{var}__{lab}")
        expected_score_cols.update({f"{var}__mean", f"{var}__disagreement", f"{var}__n_models"})
    missing = expected_score_cols - set(df.columns)
    report["missing_score_columns"] = sorted(list(missing))
    report["missing_score_columns_count"] = len(missing)

    save_json(report, ARTIFACTS / "01_dataset_fingerprint.json")
    log(f"   Rows: {report['n_rows']:,} (expected {report['expected_n_rows']:,})")
    log(f"   Time span: {report['time_range_hours']:.1f} hours")
    log(f"   Missing score columns: {report['missing_score_columns_count']}")
    return report


# ══════════════════════════════════════════════════════════════════════════
# 1.2  PER-VARIABLE ZERO-INFLATION
# ══════════════════════════════════════════════════════════════════════════
def analyze_zero_inflation(df):
    log("Computing per-variable zero-inflation...")
    rows = []
    for var in VAR_IDS:
        lab_cols = [f"{var}__{lab}" for lab in LABELLERS]
        sub = df[lab_cols]

        # Cells where all 5 labellers gave 0 (ignore NaN)
        all_zero = ((sub.fillna(1.0) == 0.0).all(axis=1)).sum()

        # Cells where at least one labeller gave non-zero
        any_nonzero = (sub.fillna(0.0) > 0.0).any(axis=1).sum()

        # Cells where all 5 labellers gave non-zero
        all_nonzero = ((sub.fillna(0.0) > 0.0).all(axis=1)).sum()

        # Mean across all cells
        mean_val = float(sub.stack().mean())

        # Fraction of NaN cells
        n_nan = int(sub.isna().sum().sum())
        n_total = sub.size
        pct_nan = 100.0 * n_nan / n_total

        rows.append({
            "variable": var,
            "group": VAR_TO_GROUP[var],
            "polarity": POLARITY[var],
            "pct_all_zero": 100.0 * all_zero / len(sub),
            "pct_any_nonzero": 100.0 * any_nonzero / len(sub),
            "pct_all_nonzero": 100.0 * all_nonzero / len(sub),
            "mean_score": mean_val,
            "pct_nan_cells": pct_nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "02_per_variable_zero_inflation.csv", index=False)
    log(f"   Sparse variables (>95% all-zero): {(out['pct_all_zero'] > 95).sum()}")
    log(f"   Active variables (<80% all-zero): {(out['pct_all_zero'] < 80).sum()}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1.3  PER-LABELLER STATISTICS
# ══════════════════════════════════════════════════════════════════════════
def analyze_per_labeller(df):
    log("Computing per-labeller statistics...")
    rows = []
    for lab in LABELLERS:
        cols = [f"{var}__{lab}" for var in VAR_IDS]
        sub = df[cols]

        # All cells flattened
        vals = sub.stack()
        n_total = len(vals)
        n_nan = vals.isna().sum()
        vals_nn = vals.dropna()

        row = {
            "labeller": lab,
            "n_cells": n_total,
            "n_nan": int(n_nan),
            "pct_nan": 100.0 * n_nan / n_total,
            "mean_score": float(vals_nn.mean()),
            "pct_zero": 100.0 * (vals_nn == 0.0).mean(),
            "pct_0.25": 100.0 * (vals_nn == 0.25).mean(),
            "pct_0.50": 100.0 * (vals_nn == 0.50).mean(),
            "pct_0.75": 100.0 * (vals_nn == 0.75).mean(),
            "pct_1.00": 100.0 * (vals_nn == 1.00).mean(),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "03_per_labeller_stats.csv", index=False)
    log(f"   Labeller mean range: [{out['mean_score'].min():.3f}, {out['mean_score'].max():.3f}]")
    log(f"   Labeller NaN range:  [{out['pct_nan'].min():.3f}%, {out['pct_nan'].max():.3f}%]")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1.4  KRIPPENDORFF'S ALPHA
# ══════════════════════════════════════════════════════════════════════════
def analyze_alpha(df):
    log("Computing Krippendorff's α per variable...")
    if not HAS_KRIPPENDORFF:
        log("   Skipped (krippendorff not installed)")
        return None

    rows = []
    for var in VAR_IDS:
        cols = [f"{var}__{lab}" for lab in LABELLERS]
        # krippendorff expects shape (raters, items) — transpose to that
        ratings = df[cols].dropna().values.T  # (5, N)
        # Need at least 2 raters and enough items
        if ratings.shape[1] < 10:
            alpha = np.nan
        else:
            try:
                alpha = krippendorff.alpha(
                    reliability_data=ratings,
                    level_of_measurement="ordinal",
                )
            except Exception as e:
                alpha = np.nan
        rows.append({
            "variable": var,
            "group": VAR_TO_GROUP[var],
            "polarity": POLARITY[var],
            "krippendorff_alpha": alpha,
            "n_items": ratings.shape[1],
        })

    out = pd.DataFrame(rows).sort_values("krippendorff_alpha", ascending=False)
    out.to_csv(ARTIFACTS / "04_krippendorff_alpha.csv", index=False)
    log(f"   α mean: {out['krippendorff_alpha'].mean():.4f}")
    log(f"   α min:  {out['krippendorff_alpha'].min():.4f} ({out.iloc[-1]['variable']})")
    log(f"   α max:  {out['krippendorff_alpha'].max():.4f} ({out.iloc[0]['variable']})")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1.5  CORRELATION + DENDROGRAM
# ══════════════════════════════════════════════════════════════════════════
def analyze_correlation_and_dendrogram(df):
    log("Computing per-labeller correlation matrices and dendrogram...")
    if not HAS_SKLEARN:
        log("   Skipped (scipy/sklearn not installed)")
        return None

    # Per-labeller correlation matrices
    corr_per_labeller = []
    for lab in LABELLERS:
        cols = [f"{var}__{lab}" for var in VAR_IDS]
        sub = df[cols].dropna()
        corr = sub.corr(method="spearman").values  # (56, 56)
        # Replace NaN (from constant columns) with 0
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr_per_labeller.append(corr)

    corr_avg = np.mean(corr_per_labeller, axis=0)
    np.save(ARTIFACTS / "05_correlation_matrix_avg.npy", corr_avg)

    # Distance: 1 - |corr| (matches Reddit approach)
    dist = 1.0 - np.abs(corr_avg)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # symmetrize
    dist = np.clip(dist, 0.0, None)

    # Ward linkage
    Z = linkage(squareform(dist, checks=False), method="ward")
    np.save(ARTIFACTS / "05_linkage_Ward.npy", Z)

    # ── Plot 1: colored by theoretical group ─────────────────────────────
    _plot_dendrogram_colored(Z, VAR_IDS, VAR_TO_GROUP, GROUP_COLORS,
                             title="Hierarchical Correlation Dendrogram — Colored by Codebook Group",
                             path=ARTIFACTS / "05_dendrogram_by_group.png",
                             legend_title="Theoretical Group")

    # ── Plot 2: colored by polarity ───────────────────────────────────────
    _plot_dendrogram_colored(Z, VAR_IDS, POLARITY, POLARITY_COLORS,
                             title="Hierarchical Correlation Dendrogram — Colored by +/− Polarity",
                             path=ARTIFACTS / "05_dendrogram_by_polarity.png",
                             legend_title="Polarity")

    # ── Plot 3: per-labeller small multiples ──────────────────────────────
    _plot_per_labeller_dendrograms(corr_per_labeller, VAR_IDS,
                                    path=ARTIFACTS / "05_dendrogram_per_labeller.png")

    # ── Extract macro-groups from a 2-cluster cut ─────────────────────────
    labels_2 = fcluster(Z, t=2, criterion="maxclust")
    # labels_2 is arbitrary (1/2); we orient by average value so
    # cluster 1 = "positive-leaning", cluster 2 = "negative-leaning"
    means = df[[f"{var}__mean" for var in VAR_IDS]].mean().values
    cl1_mean = means[labels_2 == 1].mean()
    cl2_mean = means[labels_2 == 2].mean()
    if cl1_mean >= cl2_mean:
        positive_cluster = 1
    else:
        positive_cluster = 2

    macro_groups = {}
    for var, lab in zip(VAR_IDS, labels_2):
        macro_groups[var] = "Cluster_A" if lab == positive_cluster else "Cluster_B"

    save_json(macro_groups, ARTIFACTS / "05_macro_groups.json")

    # Compare to theoretical groups
    crosstab = pd.crosstab(
        pd.Series([VAR_TO_GROUP[v] for v in VAR_IDS], name="Theoretical"),
        pd.Series([macro_groups[v] for v in VAR_IDS], name="Empirical"),
    )
    crosstab.to_csv(ARTIFACTS / "05_macro_groups_vs_theoretical.csv")

    log(f"   Cluster A size: {sum(1 for v in macro_groups.values() if v == 'Cluster_A')}")
    log(f"   Cluster B size: {sum(1 for v in macro_groups.values() if v == 'Cluster_B')}")
    return corr_avg, Z, macro_groups


def _plot_dendrogram_colored(Z, labels, label_to_color, color_map, title, path, legend_title):
    fig, ax = plt.subplots(figsize=(18, 8))
    ddata = dendrogram(Z, labels=labels, ax=ax, color_threshold=0,
                       above_threshold_color="gray", leaf_rotation=90)
    # Re-color the branch labels using our grouping
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        text = lbl.get_text()
        group = label_to_color.get(text, "unknown")
        lbl.set_color(color_map.get(group, "black"))
        lbl.set_fontsize(9)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cluster Distance (Ward Linkage Space)", fontsize=11)
    ax.set_xlabel("Dataset Dimensions (Color-coded)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Legend
    handles = [plt.Line2D([0], [0], color=c, lw=4, label=g)
               for g, c in color_map.items()]
    ax.legend(handles=handles, title=legend_title, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_labeller_dendrograms(corr_per_labeller, labels, path):
    fig, axes = plt.subplots(1, 5, figsize=(28, 7), sharey=True)
    for i, (lab, corr) in enumerate(zip(LABELLERS, corr_per_labeller)):
        dist = 1.0 - np.abs(corr)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        dist = np.clip(dist, 0.0, None)
        Z = linkage(squareform(dist, checks=False), method="ward")
        dendrogram(Z, labels=labels, ax=axes[i], no_labels=(i > 0),
                   color_threshold=0, above_threshold_color="gray")
        axes[i].set_title(lab, fontsize=12)
        axes[i].grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Per-Labeller Dendrograms (Ward, Spearman, distance = 1 − |corr|)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 1.6  AUTHOR DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
def analyze_authors(df):
    log("Analyzing author distribution...")
    df = df.copy()
    df["author_did"] = df["comment_id"].apply(extract_author_did)

    counts = df["author_did"].value_counts()

    stats = {
        "n_unique_dids": int(counts.shape[0]),
        "comments_per_did_min": int(counts.min()),
        "comments_per_did_25": float(counts.quantile(0.25)),
        "comments_per_did_50": float(counts.quantile(0.50)),
        "comments_per_did_75": float(counts.quantile(0.75)),
        "comments_per_did_95": float(counts.quantile(0.95)),
        "comments_per_did_max": int(counts.max()),
        "pct_dids_with_1_comment": float(100.0 * (counts == 1).mean()),
        "pct_comments_from_top_1pct_dids": float(
            100.0 * counts.head(max(1, int(0.01 * len(counts)))).sum() / counts.sum()
        ),
    }
    save_json(stats, ARTIFACTS / "06_author_distribution_stats.json")

    # Save the full count distribution
    counts.rename("n_comments").to_frame().to_csv(
        ARTIFACTS / "06_author_distribution.csv")

    # Zipf plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ranks = np.arange(1, len(counts) + 1)
    ax.loglog(ranks, counts.values, ".", markersize=2, alpha=0.5)
    ax.set_xlabel("Author rank (log)")
    ax.set_ylabel("Comment count (log)")
    ax.set_title("Author Comment Count Distribution (Zipf-like)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "06_author_zipf.png", dpi=150)
    plt.close(fig)

    log(f"   Unique DIDs: {stats['n_unique_dids']:,}")
    log(f"   Median comments/DID: {stats['comments_per_did_50']:.1f}")
    log(f"   Max comments/DID: {stats['comments_per_did_max']:,}")
    log(f"   Top-1% DIDs account for {stats['pct_comments_from_top_1pct_dids']:.1f}% of comments")
    return df, stats


# ══════════════════════════════════════════════════════════════════════════
# 1.7  TOPIC CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
def analyze_topics(df):
    log("Running topic clustering on 20k sample...")
    if not (HAS_ST and HAS_SKLEARN):
        log("   Skipped (sentence-transformers or sklearn unavailable)")
        return None

    sample = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)

    log("   Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample["comment_text"].tolist(),
                              batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)

    log("   Clustering with HDBSCAN...")
    # Try progressively looser HDBSCAN settings; fall back to KMeans if all fail
    labels = None
    method_used = None

    for mcs, ms in [(100, 10), (50, 10), (30, 5), (20, 5)]:
        clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean")
        candidate = clusterer.fit_predict(embeddings)
        n_c = len(set(candidate)) - (1 if -1 in candidate else 0)
        n_noise = (candidate == -1).sum()
        log(f"   HDBSCAN(min_cluster_size={mcs}, min_samples={ms}): "
            f"{n_c} clusters, {n_noise} noise")
        if n_c >= 3:
            labels = candidate
            method_used = f"HDBSCAN(mcs={mcs}, ms={ms})"
            break

    if labels is None:
        log("   HDBSCAN found no usable clusters — falling back to KMeans (K=20)")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        method_used = "KMeans(K=20)"

    sample["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log(f"   Final method: {method_used}")
    log(f"   Found {n_clusters} clusters, {n_noise} noise points")

    # Top keywords per cluster via TF-IDF on cluster members
    cluster_summaries = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        members = sample[sample["cluster"] == c]
        vec = TfidfVectorizer(max_features=2000, stop_words="english",
                              ngram_range=(1, 2), min_df=3)
        try:
            X = vec.fit_transform(members["comment_text"])
            vocab = np.array(vec.get_feature_names_out())
            scores = np.asarray(X.sum(axis=0)).flatten()
            top_idx = scores.argsort()[::-1][:15]
            top_terms = vocab[top_idx].tolist()
        except ValueError:
            top_terms = []
        cluster_summaries.append({
            "cluster": int(c),
            "size": int(len(members)),
            "top_terms": "; ".join(top_terms),
            "example_comment": members["comment_text"].iloc[0][:300],
        })

    pd.DataFrame(cluster_summaries).to_csv(
        ARTIFACTS / "07_topic_clusters.csv", index=False)

    # Save cluster assignments for later phases
    sample[["comment_id", "cluster"]].to_csv(
        ARTIFACTS / "07_topic_cluster_assignments.csv", index=False)

    # UMAP plot
    if HAS_UMAP:
        log("   Running UMAP for visualization...")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15,
                       min_dist=0.1, verbose=False)
        emb_2d = reducer.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                             c=labels, cmap="tab20", s=2, alpha=0.6)
        ax.set_title(f"UMAP of {len(sample):,} comments — {n_clusters} HDBSCAN clusters")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "07_topic_umap.png", dpi=150)
        plt.close(fig)

    return cluster_summaries


# ══════════════════════════════════════════════════════════════════════════
# 1.8  LENGTH DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════
def analyze_lengths(df):
    log("Computing length distributions...")

    # Char lengths
    comment_len_chars = df["comment_text"].str.len()
    root_len_chars = df["root_text"].str.len()
    parent_len_chars = df["parent_text"].dropna().str.len()

    # Approximate token lengths (rough heuristic: chars / 4)
    # Exact would require the DeBERTa tokenizer which is slow on 467k rows
    def pct(series):
        return {
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }

    stats = {
        "comment_chars": pct(comment_len_chars),
        "root_chars": pct(root_len_chars),
        "parent_chars": pct(parent_len_chars),
    }
    save_json(stats, ARTIFACTS / "08_length_distributions.json")

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(comment_len_chars, bins=100, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Comment text length (chars)")
    axes[1].hist(root_len_chars, bins=100, color="#2ca02c", alpha=0.8)
    axes[1].set_title("Root text length (chars)")
    axes[2].hist(parent_len_chars, bins=100, color="#ff7f0e", alpha=0.8)
    axes[2].set_title("Parent text length (chars, non-null only)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log)")
        ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "08_length_distributions.png", dpi=150)
    plt.close(fig)

    log(f"   Comment chars p50/p95/p99: {stats['comment_chars']['p50']:.0f} / "
        f"{stats['comment_chars']['p95']:.0f} / {stats['comment_chars']['p99']:.0f}")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# 1.9  CONTEXT AVAILABILITY
# ══════════════════════════════════════════════════════════════════════════
def analyze_context(df):
    log("Analyzing context availability...")

    n_rows = len(df)
    has_parent = df["parent_text"].notna().sum()
    has_root = df["root_text"].notna().sum()
    has_both = (df["parent_text"].notna() & df["root_text"].notna()).sum()

    # Estimated full input length (with special tokens)
    full_input = (
        "[THREAD ROOT] " + df["root_text"].fillna("") +
        " [PARENT] " + df["parent_text"].fillna("") +
        " [TARGET POST] " + df["comment_text"]
    )
    full_len_chars = full_input.str.len()

    stats = {
        "n_rows": int(n_rows),
        "pct_has_parent": float(100.0 * has_parent / n_rows),
        "pct_has_root": float(100.0 * has_root / n_rows),
        "pct_has_both": float(100.0 * has_both / n_rows),
        "pct_missing_parent": float(100.0 * (n_rows - has_parent) / n_rows),
        "full_input_chars_p50": float(full_len_chars.quantile(0.50)),
        "full_input_chars_p95": float(full_len_chars.quantile(0.95)),
        "full_input_chars_p99": float(full_len_chars.quantile(0.99)),
        "full_input_chars_max": float(full_len_chars.max()),
        # Rough token estimates (chars / 4)
        "full_input_tokens_p50_approx": float(full_len_chars.quantile(0.50) / 4),
        "full_input_tokens_p95_approx": float(full_len_chars.quantile(0.95) / 4),
        "full_input_tokens_p99_approx": float(full_len_chars.quantile(0.99) / 4),
        "pct_exceed_512_tokens_approx": float(
            100.0 * (full_len_chars / 4 > 512).mean()),
        "pct_exceed_384_tokens_approx": float(
            100.0 * (full_len_chars / 4 > 384).mean()),
        "pct_exceed_256_tokens_approx": float(
            100.0 * (full_len_chars / 4 > 256).mean()),
    }
    save_json(stats, ARTIFACTS / "09_context_availability.json")
    log(f"   % rows with parent_text: {stats['pct_has_parent']:.2f}%")
    log(f"   Full input p95 chars: {stats['full_input_chars_p95']:.0f} "
        f"(≈ {stats['full_input_tokens_p95_approx']:.0f} tokens)")
    log(f"   % exceeding 512 tokens (approx): {stats['pct_exceed_512_tokens_approx']:.2f}%")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# 1.10  MISSING SCORE CELLS
# ══════════════════════════════════════════════════════════════════════════
def analyze_missing_scores(df):
    log("Analyzing missing score cells per labeller...")

    rows = []
    for lab in LABELLERS:
        cols = [f"{var}__{lab}" for var in VAR_IDS]
        n_total = df[cols].size
        n_nan = int(df[cols].isna().sum().sum())
        rows.append({
            "labeller": lab,
            "total_cells": n_total,
            "nan_cells": n_nan,
            "pct_nan": 100.0 * n_nan / n_total,
            "rows_with_any_nan": int(df[cols].isna().any(axis=1).sum()),
            "pct_rows_with_any_nan": 100.0 * df[cols].isna().any(axis=1).mean(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "10_missing_score_cells.csv", index=False)
    log(f"   NaN rate range: [{out['pct_nan'].min():.4f}%, {out['pct_nan'].max():.4f}%]")
    return out


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════
def write_report(results):
    log("Writing phase1_report.md...")
    lines = []
    lines.append("# Phase 1 — Bluesky Discourse Quality Data Analysis\n")
    lines.append(f"Generated on {pd.Timestamp.now().isoformat()}\n")

    # 1.1
    r = results["verify"]
    lines.append("\n## 1.1 Dataset Fingerprint\n")
    lines.append(f"- Rows: **{r['n_rows']:,}** (expected {r['expected_n_rows']:,})")
    lines.append(f"- Columns: **{r['n_columns']}**")
    lines.append(f"- Null `comment_id`: {r['null_comment_id']}")
    lines.append(f"- Null `comment_text`: {r['null_comment_text']}")
    lines.append(f"- Null `parent_text`: {r['null_parent_text']} "
                 f"({100.0 * r['null_parent_text'] / r['n_rows']:.2f}%)")
    lines.append(f"- Time span: **{r['time_range_hours']:.1f} hours**")
    lines.append(f"- Missing score columns: {r['missing_score_columns_count']}")

    # 1.2
    z = results["zero_inflation"]
    lines.append("\n## 1.2 Per-Variable Zero-Inflation\n")
    lines.append(f"- Sparse variables (>95% all-zero): **{(z['pct_all_zero'] > 95).sum()}**")
    lines.append(f"- Active variables (<80% all-zero): **{(z['pct_all_zero'] < 80).sum()}**")
    lines.append("\n### Top-15 most active variables (lowest all-zero rate)\n")
    top = z.nsmallest(15, "pct_all_zero")[["variable", "group", "pct_all_zero", "mean_score"]]
    lines.append(top.to_markdown(index=False))
    lines.append("\n### Top-15 sparsest variables\n")
    sparsest = z.nlargest(15, "pct_all_zero")[["variable", "group", "pct_all_zero", "mean_score"]]
    lines.append(sparsest.to_markdown(index=False))

    # 1.3
    pl = results["per_labeller"]
    lines.append("\n## 1.3 Per-Labeller Statistics\n")
    lines.append(pl.to_markdown(index=False))

    # 1.4
    if results.get("alpha") is not None:
        a = results["alpha"]
        lines.append("\n## 1.4 Krippendorff's Alpha\n")
        lines.append(f"- Mean α: **{a['krippendorff_alpha'].mean():.4f}**")
        lines.append(f"- Variables with α ≥ 0.4: **{(a['krippendorff_alpha'] >= 0.4).sum()}**")
        lines.append(f"- Variables with α < 0.2: **{(a['krippendorff_alpha'] < 0.2).sum()}**")
        lines.append("\n### Top-10 highest-α variables\n")
        lines.append(a.head(10).to_markdown(index=False))
        lines.append("\n### Top-10 lowest-α variables\n")
        lines.append(a.tail(10).to_markdown(index=False))

    # 1.5
    if results.get("dendrogram") is not None:
        lines.append("\n## 1.5 Correlation Dendrogram\n")
        mg = results["dendrogram"]
        a_count = sum(1 for v in mg.values() if v == "Cluster_A")
        b_count = sum(1 for v in mg.values() if v == "Cluster_B")
        lines.append(f"- Cluster A size: **{a_count}**")
        lines.append(f"- Cluster B size: **{b_count}**")
        lines.append("- See `05_dendrogram_*.png` for visualizations")
        lines.append("- See `05_macro_groups_vs_theoretical.csv` for crosstab")

    # 1.6
    a = results["authors"]
    lines.append("\n## 1.6 Author Distribution\n")
    for k, v in a.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")

    # 1.7
    tc = results.get("topics")
    lines.append("\n## 1.7 Topic Clustering\n")
    if tc:
        lines.append(f"- Found **{len(tc)}** clusters on a 20k sample")
        for c in tc[:10]:
            lines.append(f"  - Cluster {c['cluster']} (n={c['size']}): {c['top_terms'][:120]}")
    else:
        lines.append("Skipped (dependencies missing)")

    # 1.8
    le = results["lengths"]
    lines.append("\n## 1.8 Length Distributions\n")
    for k, v in le.items():
        lines.append(f"### {k}")
        for kk, vv in v.items():
            lines.append(f"  - {kk}: {vv:.1f}")
    lines.append("- See `08_length_distributions.png`")

    # 1.9
    cx = results["context"]
    lines.append("\n## 1.9 Context Availability\n")
    for k, v in cx.items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

    # 1.10
    ms = results["missing_scores"]
    lines.append("\n## 1.10 Missing Score Cells\n")
    lines.append(ms.to_markdown(index=False))

    # Recommendations
    lines.append("\n## Recommendations for Downstream Phases\n")
    lines.append("- **max_length**: see §1.9 for p95/p99 token estimates")
    lines.append("- **Sparse variables to exclude**: see §1.2 top-15 sparsest")
    lines.append("- **Split strategy**: see §1.6 — if median comments/DID is low "
                 "and top-1% is not dominant, random split ≈ author-DID split; "
                 "otherwise author-DID is essential.")
    lines.append("- **Task C routing**: if §1.5 dendrogram shows clean macro-groups, "
                 "consider PLE; otherwise use MMoE over labellers.")

    with open(ARTIFACTS / "phase1_report.md", "w") as f:
        f.write("\n".join(lines))
    log(f"   Report written to {ARTIFACTS / 'phase1_report.md'}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    df = load_data()

    results = {}
    results["verify"]          = verify_dataset(df)
    results["zero_inflation"]  = analyze_zero_inflation(df)
    results["per_labeller"]    = analyze_per_labeller(df)
    results["alpha"]           = analyze_alpha(df)

    dendro = analyze_correlation_and_dendrogram(df)
    results["dendrogram"]      = dendro[2] if dendro else None

    df, author_stats           = analyze_authors(df)
    results["authors"]         = author_stats
    results["topics"]          = analyze_topics(df)
    results["lengths"]         = analyze_lengths(df)
    results["context"]         = analyze_context(df)
    results["missing_scores"]  = analyze_missing_scores(df)

    write_report(results)
    log("Done. All artifacts in ./phase1_artifacts/")


if __name__ == "__main__":
    main()