"""
Shape feature extraction and clustering filter module.

Pipeline:
  1. Extract shape features via AptShape Python API (predictor.predictBatch)
  2. Combine reference + generated sequences into a feature matrix
  3. Cluster with K-Means, Hierarchical, and GMM
  4. Output sequences that fall in the same cluster as the reference in ALL three methods
  5. Save per-method UMAP plots, heatmaps, and the final consensus sequences
"""

import sys
import numpy as np
from pathlib import Path

DEFAULT_FEATURES = ["MGW", "Roll", "HelT", "ProT", "Shift", "Slide", "Rise", "Tilt"]


def _get_predictor():
    """Load AptShape predictor (embedded subpackage)."""
    from .aptshape.predictor import predictor
    return predictor(mode="cpu")


def extract_shape_features(
    sequences: list,
    features: list = None,
    layer: int = 7,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract shape features in batch. Returns [n_seqs, n_features] matrix.
    Each feature value is the mean across all positions in the sequence.
    """
    if features is None:
        features = DEFAULT_FEATURES

    pred = _get_predictor()
    feature_cols = []

    for feat in features:
        print(f"  Extracting feature: {feat} ...", flush=True)
        col_vals = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i: i + batch_size]
            preds = pred.predictBatch(feat, batch, layer=layer)
            for row in preds:
                col_vals.append(float(np.mean(row)))
        feature_cols.append(np.array(col_vals))

    return np.column_stack(feature_cols)


def _cluster_kmeans(X: np.ndarray, n_clusters: int):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def _cluster_hierarchical(X: np.ndarray, n_clusters: int):
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
    return labels, centers


def _cluster_gmm(X: np.ndarray, n_clusters: int):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    labels = gmm.fit_predict(X)
    return labels, gmm.means_


def cluster_and_filter(
    ref_features: np.ndarray,
    gen_features: np.ndarray,
    gen_sequences: list,
    feature_names: list,
    output_dir: Path,
    n_clusters: int = 5,
    prefix: str = "result",
    methods: list = None,
) -> dict:
    """
    Cluster reference + generated sequences and filter those in the same cluster
    as the reference across all three methods (intersection).

    Returns:
        dict with keys:
          - "kmeans": sequences selected by K-Means
          - "hierarchical": sequences selected by hierarchical clustering
          - "gmm": sequences selected by GMM
          - "consensus": intersection of all three methods (main output)
    """
    from sklearn.preprocessing import StandardScaler
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if methods is None:
        methods = ["kmeans", "hierarchical", "gmm"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_ref = len(ref_features)
    n_gen = len(gen_features)

    all_features = np.vstack([ref_features, gen_features])

    scaler = StandardScaler()
    X = scaler.fit_transform(all_features)

    print("  Running UMAP dimensionality reduction ...", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    cluster_funcs = {
        "kmeans": _cluster_kmeans,
        "hierarchical": _cluster_hierarchical,
        "gmm": _cluster_gmm,
    }

    method_selected_indices = {}
    filtered_results = {}

    for method in methods:
        if method not in cluster_funcs:
            print(f"[WARNING] Unknown clustering method: {method}, skipping", file=sys.stderr)
            continue

        print(f"  [{method}] Clustering (k={n_clusters}) ...", flush=True)
        labels, centers = cluster_funcs[method](X, n_clusters)

        ref_labels = labels[:n_ref]
        gen_labels = labels[n_ref:]

        ref_clusters = set(ref_labels.tolist())

        selected_idx = set(
            i for i, lbl in enumerate(gen_labels) if lbl in ref_clusters
        )
        method_selected_indices[method] = selected_idx

        selected = [gen_sequences[i] for i in sorted(selected_idx)]
        filtered_results[method] = selected

        print(f"    Reference cluster(s): {sorted(ref_clusters)}")
        print(f"    Selected: {len(selected)}/{n_gen} sequences")

        mask = np.array([i in selected_idx for i in range(n_gen)])

        # --- UMAP scatter plot ---
        fig, ax = plt.subplots(figsize=(9, 6))
        sc_gen = ax.scatter(
            X_umap[n_ref:, 0], X_umap[n_ref:, 1],
            c=gen_labels, cmap="tab10", alpha=0.4, s=8, label="Generated"
        )
        ax.scatter(
            X_umap[:n_ref, 0], X_umap[:n_ref, 1],
            c="red", marker="*", s=200, zorder=5, label="Reference (PDB)"
        )
        if mask.any():
            ax.scatter(
                X_umap[n_ref:][mask, 0], X_umap[n_ref:][mask, 1],
                facecolors="none", edgecolors="lime", s=25, linewidths=0.8,
                zorder=4, label=f"Selected ({len(selected)})"
            )
        plt.colorbar(sc_gen, ax=ax, label="Cluster")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"Shape Clustering — {method.upper()} (K={n_clusters})")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        umap_path = output_dir / f"{prefix}_{method}_umap.png"
        fig.savefig(umap_path, dpi=150)
        plt.close(fig)
        print(f"    ✓ UMAP plot saved → {umap_path}")

        # --- Cluster center heatmap ---
        centers_orig = scaler.inverse_transform(centers)
        df_centers = pd.DataFrame(
            centers_orig,
            columns=feature_names,
            index=[
                f"Cluster {i}" + (" ★ref" if i in ref_clusters else "")
                for i in range(n_clusters)
            ],
        )
        fig, ax = plt.subplots(figsize=(max(8, len(feature_names)), n_clusters + 1))
        sns.heatmap(df_centers, annot=True, fmt=".3f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title(f"Cluster Centers — {method.upper()} (★ = reference cluster)")
        plt.tight_layout()
        heatmap_path = output_dir / f"{prefix}_{method}_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"    ✓ Heatmap saved → {heatmap_path}")

    # --- Consensus: intersection of all methods ---
    if len(method_selected_indices) == len(methods):
        consensus_idx = set.intersection(*method_selected_indices.values())
    else:
        consensus_idx = set.union(*method_selected_indices.values()) if method_selected_indices else set()

    consensus_seqs = [gen_sequences[i] for i in sorted(consensus_idx)]
    filtered_results["consensus"] = consensus_seqs

    print(f"\n  [Consensus] Intersection of all methods: {len(consensus_seqs)}/{n_gen} sequences")

    # Save per-method selected sequences
    for method in methods:
        if method not in filtered_results:
            continue
        out_path = output_dir / f"{prefix}_{method}_selected.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for seq in filtered_results[method]:
                f.write(seq + "\n")
        print(f"  ✓ [{method}] selected sequences → {out_path}")

    # Save final consensus sequences
    consensus_path = output_dir / f"{prefix}_final.txt"
    with open(consensus_path, "w", encoding="utf-8") as f:
        for seq in consensus_seqs:
            f.write(seq + "\n")
    print(f"  ✓ [Final] consensus sequences → {consensus_path}")

    # --- Consensus UMAP plot ---
    consensus_mask = np.array([i in consensus_idx for i in range(n_gen)])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        X_umap[n_ref:, 0], X_umap[n_ref:, 1],
        c="lightgray", alpha=0.3, s=8, label="Generated (all)"
    )
    if consensus_mask.any():
        ax.scatter(
            X_umap[n_ref:][consensus_mask, 0], X_umap[n_ref:][consensus_mask, 1],
            c="steelblue", alpha=0.7, s=15, label=f"Consensus ({len(consensus_seqs)})"
        )
    ax.scatter(
        X_umap[:n_ref, 0], X_umap[:n_ref, 1],
        c="red", marker="*", s=200, zorder=5, label="Reference (PDB)"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Final Consensus Sequences (all 3 methods agree)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    consensus_umap_path = output_dir / f"{prefix}_consensus_umap.png"
    fig.savefig(consensus_umap_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ [Final] consensus UMAP plot → {consensus_umap_path}")

    return filtered_results
