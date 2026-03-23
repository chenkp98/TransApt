"""
DNA 形状特征提取与聚类筛选模块。

流程：
  1. 直接调用 deepDNAshape Python API（predictor.predictBatch）提取形状特征
  2. 对原始序列 + 生成序列拼合后提取特征矩阵
  3. 用 K-Means、层次聚类、GMM 三种方法分别聚类
  4. 取三种方法都与原始序列聚为同一类的生成序列作为最终输出
  5. 输出各方法的 PCA 散点图、热图，以及最终交集序列
"""

import sys
import numpy as np
from pathlib import Path

DEFAULT_FEATURES = ["MGW", "Roll", "HelT", "ProT", "Shift", "Slide", "Rise", "Tilt"]

def _get_predictor():
    """获取 deepDNAshape predictor 实例（内嵌子包）。"""
    from .deepdnashape.predictor import predictor
    return predictor(mode="cpu")


def extract_shape_features(
    sequences: list,
    features: list = None,
    layer: int = 7,
    batch_size: int = 256,
) -> np.ndarray:
    """
    批量提取 DNA 形状特征，返回 [n_seqs, n_features] 矩阵。
    每条序列每种特征取各位置均值作为标量。
    """
    if features is None:
        features = DEFAULT_FEATURES

    pred = _get_predictor()
    feature_cols = []

    for feat in features:
        print(f"  提取特征: {feat} ...", flush=True)
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
    联合聚类原始序列与生成序列，筛选三种方法都与原始序列同簇的生成序列。

    Returns:
        dict 包含：
          - "kmeans": K-Means 筛选的序列列表
          - "hierarchical": 层次聚类筛选的序列列表
          - "gmm": GMM 筛选的序列列表
          - "consensus": 三种方法取交集的最终序列列表（主要输出）
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
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

    # 合并：前 n_ref 行为参考序列，后 n_gen 行为生成序列
    all_features = np.vstack([ref_features, gen_features])

    scaler = StandardScaler()
    X = scaler.fit_transform(all_features)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    cluster_funcs = {
        "kmeans": _cluster_kmeans,
        "hierarchical": _cluster_hierarchical,
        "gmm": _cluster_gmm,
    }

    # 每种方法筛选出的序列索引集合（在 gen_sequences 中的下标）
    method_selected_indices = {}
    filtered_results = {}

    for method in methods:
        if method not in cluster_funcs:
            print(f"[WARNING] 未知聚类方法: {method}，跳过", file=sys.stderr)
            continue

        print(f"  [{method}] 聚类 (k={n_clusters}) ...", flush=True)
        labels, centers = cluster_funcs[method](X, n_clusters)

        ref_labels = labels[:n_ref]
        gen_labels = labels[n_ref:]

        # 参考序列所在的所有簇
        ref_clusters = set(ref_labels.tolist())

        # 与参考序列同簇的生成序列索引
        selected_idx = set(
            i for i, lbl in enumerate(gen_labels) if lbl in ref_clusters
        )
        method_selected_indices[method] = selected_idx

        selected = [gen_sequences[i] for i in sorted(selected_idx)]
        filtered_results[method] = selected

        print(f"    参考序列所在簇: {sorted(ref_clusters)}")
        print(f"    筛选出 {len(selected)}/{n_gen} 条序列")

        mask = np.array([i in selected_idx for i in range(n_gen)])

        # --- PCA 散点图 ---
        fig, ax = plt.subplots(figsize=(9, 6))
        sc_gen = ax.scatter(
            X_pca[n_ref:, 0], X_pca[n_ref:, 1],
            c=gen_labels, cmap="tab10", alpha=0.4, s=8, label="Generated"
        )
        ax.scatter(
            X_pca[:n_ref, 0], X_pca[:n_ref, 1],
            c="red", marker="*", s=200, zorder=5, label="Reference (PDB)"
        )
        if mask.any():
            ax.scatter(
                X_pca[n_ref:][mask, 0], X_pca[n_ref:][mask, 1],
                facecolors="none", edgecolors="lime", s=25, linewidths=0.8,
                zorder=4, label=f"Selected ({len(selected)})"
            )
        plt.colorbar(sc_gen, ax=ax, label="Cluster")
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.set_title(f"DNA Shape Clustering — {method.upper()} (K={n_clusters})")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        pca_path = output_dir / f"{prefix}_{method}_pca.png"
        fig.savefig(pca_path, dpi=150)
        plt.close(fig)
        print(f"    ✓ PCA 散点图 → {pca_path}")

        # --- 聚类中心热图 ---
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
        print(f"    ✓ 聚类热图 → {heatmap_path}")

    # ── 三种方法取交集 ──────────────────────────────────────────────
    if len(method_selected_indices) == len(methods):
        consensus_idx = set.intersection(*method_selected_indices.values())
    else:
        consensus_idx = set.union(*method_selected_indices.values()) if method_selected_indices else set()

    consensus_seqs = [gen_sequences[i] for i in sorted(consensus_idx)]
    filtered_results["consensus"] = consensus_seqs

    print(f"\n  [共识] 三种方法交集: {len(consensus_seqs)}/{n_gen} 条序列")

    # 保存各方法筛选序列
    for method in methods:
        if method not in filtered_results:
            continue
        out_path = output_dir / f"{prefix}_{method}_selected.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for seq in filtered_results[method]:
                f.write(seq + "\n")
        print(f"  ✓ [{method}] 筛选序列 → {out_path}")

    # 保存最终交集序列（主要输出）
    consensus_path = output_dir / f"{prefix}_final.txt"
    with open(consensus_path, "w", encoding="utf-8") as f:
        for seq in consensus_seqs:
            f.write(seq + "\n")
    print(f"  ✓ [最终输出] 交集序列 → {consensus_path}")

    # --- 交集序列在 PCA 图上的综合可视化 ---
    consensus_mask = np.array([i in consensus_idx for i in range(n_gen)])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        X_pca[n_ref:, 0], X_pca[n_ref:, 1],
        c="lightgray", alpha=0.3, s=8, label="Generated (all)"
    )
    if consensus_mask.any():
        ax.scatter(
            X_pca[n_ref:][consensus_mask, 0], X_pca[n_ref:][consensus_mask, 1],
            c="steelblue", alpha=0.7, s=15, label=f"Consensus ({len(consensus_seqs)})"
        )
    ax.scatter(
        X_pca[:n_ref, 0], X_pca[:n_ref, 1],
        c="red", marker="*", s=200, zorder=5, label="Reference (PDB)"
    )
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(f"Final Consensus Sequences (all 3 methods agree)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    consensus_pca_path = output_dir / f"{prefix}_consensus_pca.png"
    fig.savefig(consensus_pca_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ [最终输出] 综合 PCA 图 → {consensus_pca_path}")

    return filtered_results
