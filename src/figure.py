import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

BASE = os.path.join("..", "outputs")
DOC2VEC_DIR = os.path.join(BASE, "doc2vec")
W2V_DIR = os.path.join(BASE, "w2v_bins")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_metrics():
    p = os.path.join(BASE, "comparison_metrics.csv")
    return pd.read_csv(p)

def plot_silhouette(metrics: pd.DataFrame):
    # 按 method 分组画 bar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    metrics = metrics.copy()
    metrics["label"] = metrics["method"] + "\n" + metrics["config"]

    x = np.arange(len(metrics))
    ax.bar(x, metrics["silhouette_cosine_tfidf"].fillna(0.0).values)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Silhouette (cosine on TF-IDF space)")
    ax.set_title("Silhouette comparison across methods/configs")
    fig.tight_layout()

    out = os.path.join(FIG_DIR, "fig1_silhouette_comparison.png")
    fig.savefig(out, dpi=200)
    print("Saved:", out)

def plot_cluster_size_distributions():
    # 对每个 clusters.csv 画 cluster size 条形图
    cluster_files = []
    for d in [DOC2VEC_DIR, W2V_DIR]:
        for f in sorted(os.listdir(d)):
            if f.endswith("_clusters.csv"):
                cluster_files.append(os.path.join(d, f))

    for path in cluster_files:
        df = pd.read_csv(path)
        counts = df["cluster_id"].value_counts().sort_index()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(counts.index.astype(str), counts.values)

        ax.set_xlabel("cluster_id")
        ax.set_ylabel("num_docs")
        ax.set_title(f"Cluster size distribution: {os.path.basename(path)}")
        fig.tight_layout()

        out = os.path.join(
            FIG_DIR,
            "fig2_cluster_sizes_" + os.path.basename(path).replace(".csv", ".png")
        )
        fig.savefig(out, dpi=200)
        print("Saved:", out)

def pick_best_configs(metrics: pd.DataFrame):
    # 选每种方法 silhouette 最高的 config（用于 2D scatter）
    best = {}
    for method in metrics["method"].unique():
        sub = metrics[metrics["method"] == method].copy()
        sub = sub.dropna(subset=["silhouette_cosine_tfidf"])
        sub = sub.sort_values("silhouette_cosine_tfidf", ascending=False)
        best[method] = sub.iloc[0]["config"]
    return best

def find_cluster_csv(method: str, config: str):
    if method == "Doc2Vec":
        d = DOC2VEC_DIR
    else:
        d = W2V_DIR
    target = os.path.join(d, f"{config}_clusters.csv")
    return target

def plot_2d_scatter_tfidf(df: pd.DataFrame, title: str, out_name: str):
    # 用 TF-IDF -> TruncatedSVD(2D) 做可视化
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["cluster_id"].astype(int).tolist()

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=2, random_state=0)
    X2 = svd.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 不指定颜色，让 matplotlib 默认循环即可（符合你的要求）
    for cid in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == cid]
        ax.scatter(X2[idx, 0], X2[idx, 1], s=6, label=str(cid), alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("SVD-1 (TF-IDF)")
    ax.set_ylabel("SVD-2 (TF-IDF)")
    ax.legend(title="cluster_id", fontsize=7, ncol=2)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, out_name)
    fig.savefig(out, dpi=200)
    print("Saved:", out)

def main():
    metrics = load_metrics()

    # Fig 1: silhouette comparison
    plot_silhouette(metrics)

    # Fig 2: cluster size distributions
    plot_cluster_size_distributions()

    # Fig 3/4: 2D scatter for best config of each method
    best = pick_best_configs(metrics)
    print("Best configs:", best)

    # Doc2Vec best
    p1 = find_cluster_csv("Doc2Vec", best["Doc2Vec"])
    df1 = pd.read_csv(p1)
    plot_2d_scatter_tfidf(df1, f"Doc2Vec 2D view (best: {best['Doc2Vec']})",
                          f"fig3_2d_doc2vec_{best['Doc2Vec']}.png")

    # Word2Vec+Bins best
    p2 = find_cluster_csv("Word2Vec+Bins", best["Word2Vec+Bins"])
    df2 = pd.read_csv(p2)
    plot_2d_scatter_tfidf(df2, f"Word2Vec+Bins 2D view (best: {best['Word2Vec+Bins']})",
                          f"fig4_2d_w2v_bins_{best['Word2Vec+Bins']}.png")

if __name__ == "__main__":
    main()