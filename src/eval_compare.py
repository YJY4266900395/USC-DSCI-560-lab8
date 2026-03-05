import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

DOC2VEC_DIR = os.path.join("..", "outputs", "doc2vec")
W2V_DIR = os.path.join("..", "outputs", "w2v_bins")

def load_clusters(csv_path):
    df = pd.read_csv(csv_path)
    # basic cleaning
    df["text"] = df["text"].fillna("").astype(str)
    return df

def tfidf_matrix(texts, max_features=5000):
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )
    X = vec.fit_transform(texts)
    return X

def safe_silhouette(X, labels):

    if len(set(labels)) < 2:
        return None
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return None

def summarize_clusters(df, top_n=3):
    out = {}
    for cid in sorted(df["cluster_id"].unique()):
        sample = df[df["cluster_id"] == cid].head(top_n)["text"].tolist()
        out[int(cid)] = [s[:120].replace("\n", " ") for s in sample]
    return out

def main():
    results = []

    doc2vec_files = sorted([f for f in os.listdir(DOC2VEC_DIR) if f.endswith("_clusters.csv")])
    w2v_files = sorted([f for f in os.listdir(W2V_DIR) if f.endswith("_clusters.csv")])

    all_runs = []
    for f in doc2vec_files:
        all_runs.append(("Doc2Vec", os.path.join(DOC2VEC_DIR, f), f.replace("_clusters.csv","")))
    for f in w2v_files:
        all_runs.append(("Word2Vec+Bins", os.path.join(W2V_DIR, f), f.replace("_clusters.csv","")))

    for method, path, config in all_runs:
        df = load_clusters(path)

        texts = df["text"].tolist()
        labels = df["cluster_id"].tolist()

        X = tfidf_matrix(texts, max_features=5000)
        sil = safe_silhouette(X, labels)

        results.append({
            "method": method,
            "config": config,
            "n_docs": len(df),
            "n_clusters": len(set(labels)),
            "silhouette_cosine_tfidf": sil
        })


        samples = summarize_clusters(df, top_n=3)
        sample_out = os.path.join("..", "outputs", f"samples_{method}_{config}.txt").replace(" ", "")
        with open(sample_out, "w", encoding="utf-8") as w:
            w.write(f"{method} - {config}\n")
            w.write("="*60 + "\n")
            for cid, sents in samples.items():
                w.write(f"\nCluster {cid}\n")
                for s in sents:
                    w.write(f"- {s}\n")
        print("Saved samples:", sample_out)

    res_df = pd.DataFrame(results).sort_values(["method", "config"])
    out_csv = os.path.join("..", "outputs", "comparison_metrics.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\nSaved metrics:", out_csv)
    print(res_df)

if __name__ == "__main__":
    main()