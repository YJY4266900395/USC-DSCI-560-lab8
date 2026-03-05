"""
Shared utilities
"""

import json
import os
import re
import numpy as np
from collections import Counter
from datetime import datetime
from typing import Dict, List

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# -- I/O helpers (from Lab 5 embed_and_cluster.py) ------------------

def load_records(path: str):
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Input must be .json or .jsonl")


def safe_get(rec: dict, key: str, default: str = ""):
    v = rec.get(key, default)
    return v if v is not None else default


# -- Text tokenization (new for Lab 8) ------------------------------
# Doc2Vec and Word2Vec require explicit tokenization, unlike
# SentenceTransformer used in Lab 5 which tokenizes internally.

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "they", "we", "me", "him", "her", "them", "us",
    "my", "your", "our", "their", "from", "not", "no", "yes", "about", "into",
    "over", "under", "after", "before", "between", "during", "than", "too",
    "very", "can", "could", "should", "would", "will", "just", "also", "more",
    "most", "some", "any", "all", "do", "did", "does", "done", "has", "have",
    "had", "so", "what", "which", "who", "whom", "how", "when", "where", "why",
    "up", "out", "off", "down", "there", "here", "now", "only", "own", "same",
    "such", "other", "each", "every", "both", "few", "many", "much", "well",
    "back", "even", "still", "new", "old", "first", "last", "long", "great",
    "little", "right", "big", "high", "small", "large", "next", "early",
    "young", "important", "public", "good", "bad", "make", "like", "get",
    "got", "go", "went", "going", "come", "take", "see", "know", "think",
    "say", "said", "tell", "give", "use", "find", "want", "look", "put",
    "need", "try", "ask", "work", "seem", "feel", "leave", "call", "keep",
    "let", "begin", "show", "hear", "play", "run", "move", "live", "believe",
    "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose",
    "pay", "meet", "include", "continue", "set", "learn", "change", "lead",
    "understand", "watch", "follow", "stop", "create", "speak", "read",
    "allow", "add", "spend", "grow", "open", "walk", "win", "offer",
    "remember", "consider", "appear", "buy", "wait", "serve", "die", "send",
    "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain",
    "suggest", "raise", "pass", "sell", "require", "report", "decide", "pull",
    "http", "https", "www", "com", "org", "reddit", "amp",
    "removed", "deleted", "edit", "update",
}

_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def tokenize(text: str, min_len: int = 3) -> List[str]:
    """Tokenize a document into cleaned lowercased word tokens."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return [w for w in text.split() if len(w) >= min_len and w not in STOPWORDS]


def prepare_data(input_path: str, min_text_len: int = 30, min_tokens: int = 3):
    """
    Load records, build raw texts and tokenized docs, filter short ones.
    Returns: (records, texts, tokenized_docs)
    """
    records = load_records(input_path)
    texts = [(safe_get(r, "final_text") + " " + safe_get(r, "ocr_text")).strip() for r in records]

    keep = [i for i, t in enumerate(texts) if isinstance(t, str) and len(t.strip()) >= min_text_len]
    records = [records[i] for i in keep]
    texts = [texts[i] for i in keep]

    tokenized_docs = [tokenize(t) for t in texts]

    valid = [i for i, toks in enumerate(tokenized_docs) if len(toks) >= min_tokens]
    records = [records[i] for i in valid]
    texts = [texts[i] for i in valid]
    tokenized_docs = [tokenized_docs[i] for i in valid]

    print(f"[INFO] {len(records)} documents after filtering + tokenization")
    print(f"[INFO] Avg tokens/doc: {np.mean([len(t) for t in tokenized_docs]):.1f}")

    return records, texts, tokenized_docs


# -- Keyword extraction (from Lab 5) --------------------------------

def top_keywords_per_cluster(
    texts: List[str], labels: np.ndarray, top_n: int = 10, max_features: int = 20000
) -> Dict[int, List[str]]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords: Dict[int, List[str]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            cluster_keywords[int(c)] = []
            continue
        mean_scores = X[idx].mean(axis=0).A1
        top_idx = mean_scores.argsort()[::-1][:top_n]
        cluster_keywords[int(c)] = terms[top_idx].tolist()
    return cluster_keywords


# -- Representative posts (from Lab 5) ------------------------------

def representative_posts(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    records: List[dict],
    per_cluster: int = 3,
) -> Dict[int, List[dict]]:
    reps: Dict[int, List[dict]] = {}
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[int(c)] = []
            continue

        E = embeddings[idx]
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        sim = E_norm @ c_norm[int(c)]
        order = np.argsort(-sim)[:per_cluster]
        chosen = idx[order]

        out = []
        for j, i in enumerate(chosen):
            r = records[int(i)]
            preview = (safe_get(r, "final_text") + " " + safe_get(r, "ocr_text")).strip()[:220]
            out.append({
                "fullname": safe_get(r, "fullname"),
                "subreddit": safe_get(r, "subreddit"),
                "title": safe_get(r, "title"),
                "permalink": safe_get(r, "permalink"),
                "is_image": bool(r.get("is_image", False)),
                "image_url": safe_get(r, "image_url"),
                "score": r.get("score", 0),
                "num_comments": r.get("num_comments", 0),
                "created": safe_get(r, "created"),
                "final_text_preview": preview,
                "cosine_similarity_to_centroid": float(sim[order[j]]),
            })
        reps[int(c)] = out
    return reps


# -- Topic labeling (from Lab 5) ------------------------------------

def build_topic_labels(cluster_keywords: Dict[int, List[str]], top_m: int = 3) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c, kws in cluster_keywords.items():
        kws = kws or []
        topic = ", ".join([k for k in kws[:top_m] if isinstance(k, str) and k.strip()]) if kws else f"cluster_{c}"
        labels[int(c)] = topic if topic else f"cluster_{c}"
    return labels


# -- PCA plotting (from Lab 5, extended with title param) -----------

def maybe_plot(embeddings, labels, outpath, title="Clusters (PCA 2D)", max_points=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = embeddings.shape[0]
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        E, y = embeddings[idx], labels[idx]
    else:
        E, y = embeddings, labels

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(E)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], s=8, c=y, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -- Evaluation metrics (new for Lab 8) -----------------------------

def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    emb_norm = normalize(embeddings, norm="l2")
    n_labels = len(set(labels.tolist()))
    if n_labels < 2:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"), "davies_bouldin": float("nan")}
    return {
        "silhouette": float(round(silhouette_score(emb_norm, labels, metric="cosine"), 4)),
        "calinski_harabasz": float(round(calinski_harabasz_score(emb_norm, labels), 4)),
        "davies_bouldin": float(round(davies_bouldin_score(emb_norm, labels), 4)),
    }


# -- Run a single config (shared pipeline for Part 1 & Part 2) -----

def run_config(
    method: str,
    config: dict,
    embeddings: np.ndarray,
    records: List[dict],
    texts: List[str],
    k: int,
    out_dir: str,
    top_keywords_n: int = 10,
    rep_posts_n: int = 3,
    do_plot: bool = True,
) -> dict:
    """
    Cluster → evaluate → save artifacts.
    Same output format as Lab 5 embed_and_cluster.py.
    """
    name = config["name"]
    # Filter out all-zero vectors (cosine distance is undefined for them)
    row_norms = np.linalg.norm(embeddings, axis=1)
    nonzero_mask = row_norms > 0
    if not nonzero_mask.all():
        n_dropped = int((~nonzero_mask).sum())
        print(f"[WARN] Dropping {n_dropped} all-zero vectors before clustering")
        embeddings = embeddings[nonzero_mask]
        records = [r for r, m in zip(records, nonzero_mask) if m]
        texts = [t for t, m in zip(texts, nonzero_mask) if m]

    # Cluster with cosine distance using Agglomerative Clustering
    emb_norm = normalize(embeddings, norm="l2")
    agg = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    labels = agg.fit_predict(emb_norm).astype(int)
    print("[INFO] Cluster sizes:", dict(Counter(labels.tolist())))

    # Compute centroids as mean of normalized embeddings per cluster
    centroids = np.zeros((k, emb_norm.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = emb_norm[mask].mean(axis=0)

    # Evaluate
    metrics = evaluate_clustering(embeddings, labels)
    print(f"[INFO] Silhouette: {metrics['silhouette']}")
    print(f"[INFO] Calinski-Harabasz: {metrics['calinski_harabasz']}")
    print(f"[INFO] Davies-Bouldin: {metrics['davies_bouldin']}")

    # Keywords + topics + representative posts
    cluster_keywords = top_keywords_per_cluster(texts, labels, top_n=top_keywords_n)
    topic_labels = build_topic_labels(cluster_keywords)
    reps = representative_posts(embeddings, labels, centroids, records, per_cluster=rep_posts_n)

    # Save artifacts
    prefix = f"{method}_{name}"

    summary = []
    for c in sorted(set(labels.tolist())):
        summary.append({
            "cluster_id": int(c),
            "topic": topic_labels.get(int(c), f"cluster_{int(c)}"),
            "size": int((labels == c).sum()),
            "top_keywords": cluster_keywords.get(int(c), []),
            "representative_posts": reps.get(int(c), []),
        })
    summary_path = os.path.join(out_dir, f"{prefix}_clusters_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {summary_path}")

    meta = {
        "method": method,
        "config": config,
        "k": int(k),
        "n_docs": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "text_fields_used": ["final_text", "ocr_text"],
        "metrics": metrics,
    }
    meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {meta_path}")

    if do_plot:
        plot_path = os.path.join(out_dir, f"{prefix}_clusters_plot.png")
        maybe_plot(embeddings, labels, outpath=plot_path,
                   title=f"{method} [{name}] k={k} (PCA 2D)")
        print(f"[DONE] Wrote {plot_path}")

    dim = config.get("vector_size") or config.get("doc_dim")
    return {
        "method": method,
        "config_name": name,
        "dim": dim,
        **config,
        "k": k,
        "n_docs": int(embeddings.shape[0]),
        "cluster_sizes": dict(Counter(labels.tolist())),
        **metrics,
    }