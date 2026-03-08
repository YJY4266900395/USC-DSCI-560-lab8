# Standard library
import json
import re
import html
from pathlib import Path
from itertools import product
from collections import Counter

# Data processing
import numpy as np
import pandas as pd

# Embedding models (Gensim)
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Text processing / keyword extraction
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Clustering and evaluation
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path("data") / "posts_lab5_5000.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    posts = json.load(f)

assert isinstance(posts, list), "Expected a JSON list (array) of posts."
print("Total posts:", len(posts))


records = [
    p for p in posts
    if isinstance(p, dict)
    and isinstance(p.get("final_text"), str)
    and p.get("final_text").strip()
]

texts = [p["final_text"].strip() for p in records]

assert len(records) == len(texts), "records and texts are not aligned."

print("Non-empty texts:", len(texts))
print("Unique texts:", len(set(texts)), "/", len(texts))


sample = next((p for p in records if isinstance(p, dict)), None)
print("Sample keys:", list(sample.keys()) if sample else "EMPTY POSTS")


lengths = [len(t.split()) for t in texts]

if lengths:
    print("Min words:", min(lengths))
    print("Max words:", max(lengths))
    print("Average words:", sum(lengths) / len(lengths))
else:
    print("No non-empty texts found, cannot compute length stats.")

lengths = np.array([len(p.get("final_text","").split()) for p in posts])

print("p50:", np.percentile(lengths, 50))
print("p90:", np.percentile(lengths, 90))
print("p95:", np.percentile(lengths, 95))
print("p99:", np.percentile(lengths, 99))
print("max:", lengths.max())

TAG_RE = re.compile(r"<[^>]+>")
WS_RE  = re.compile(r"\s+")
URL_RE = re.compile(r"http\S+|www\.\S+")

def clean_text(text: str) -> str:
    """
    - html unescape
    - remove urls, html tags
    - remove zero-width/BOM chars
    - keep only a-z0-9 and whitespace
    - normalize whitespace
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

MIN_WORDS = 4
MAX_WORDS = 583

clean_records = []
tokenized_docs = []

dropped_short = 0
truncated_long = 0
empty_after_clean = 0

for p in posts:
    if not isinstance(p, dict):
        continue

    raw_text = p.get("final_text")
    if not raw_text:
        raw_text = (p.get("title", "") + " " + p.get("body", ""))

    text = clean_text(raw_text)

    if not text:
        empty_after_clean += 1
        continue

    toks = text.split()

    if len(toks) < MIN_WORDS:
        dropped_short += 1
        continue

    if len(toks) > MAX_WORDS:
        toks = toks[:MAX_WORDS]
        truncated_long += 1

    clean_records.append(p)
    tokenized_docs.append(toks)

assert len(clean_records) == len(tokenized_docs)

print("Documents:", len(tokenized_docs))
print("Dropped (too short):", dropped_short)
print("Empty after clean:", empty_after_clean)
print("Truncated (too long):", truncated_long)

lengths = [len(t) for t in tokenized_docs]

print("Min words:", min(lengths) if lengths else 0)
print("Max words:", max(lengths) if lengths else 0)
print("Average words:", (sum(lengths)/len(lengths)) if lengths else 0)

def evaluate_clustering(X_norm, labels):
    """X_norm must be L2-normalized."""
    n_clusters = len(set(labels))
    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

    return {
        "silhouette": float(silhouette_score(X_norm, labels, metric="cosine")),
        "calinski_harabasz": float(calinski_harabasz_score(X_norm, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_norm, labels)),
    }


def top_keywords_per_cluster(docs, labels, top_n=10, max_features=20000):
    """
    docs must be aligned with labels, e.g.:
    docs = [" ".join(doc) for doc in tokenized_docs]
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vectorizer.fit_transform(docs)
    terms = np.array(vectorizer.get_feature_names_out())

    out = {}
    labels = np.asarray(labels)

    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[int(c)] = []
            continue

        mean_scores = X[idx].mean(axis=0).A1
        top_idx = mean_scores.argsort()[::-1][:top_n]
        out[int(c)] = terms[top_idx].tolist()

    return out


def representative_docs(X_norm, labels, docs, per_cluster=3, records=None):
    """
    docs must be aligned with labels and records.
    Example:
    docs = [" ".join(doc) for doc in tokenized_docs]
    """
    labels = np.asarray(labels)
    reps = {}

    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[int(c)] = []
            continue

        centroid = X_norm[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        sims = X_norm[idx] @ centroid
        top_local = np.argsort(-sims)[:per_cluster]
        chosen = idx[top_local]

        out = []
        for j, i in enumerate(chosen):
            item = {
                "doc_index": int(i),
                "cosine_to_centroid": float(sims[top_local[j]]),
                "preview": " ".join(docs[i].split()[:40])
            }

            if records is not None:
                item["title"] = records[i].get("title", "")
                item["subreddit"] = records[i].get("subreddit", "")
                item["permalink"] = records[i].get("permalink", "")

            out.append(item)

        reps[int(c)] = out

    return reps


def save_pca_plot(X, labels, outpath, title):
    X = normalize(X, norm="l2")
    labels = np.asarray(labels)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X2[:, 0],
        X2[:, 1],
        s=8,
        c=labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.colorbar(sc, label="Cluster")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    plt.close()


def show_cluster_samples(labels, docs, tokenized_docs, n=3, max_len=140):
    """
    docs must be aligned with tokenized_docs and labels.
    Example:
    docs = [" ".join(doc) for doc in tokenized_docs]
    """
    labels = np.asarray(labels)
    buckets = {}
    BAD_TOKENS = {"s", "t", "m", "ve", "re", "d", "ll", "1", "2", "3", "4", "0"}

    for i, c in enumerate(labels):
        buckets.setdefault(int(c), []).append(i)

    for c in sorted(buckets.keys()):
        idxs = buckets[c]
        print(f"\n=== Cluster {c} (size={len(idxs)}) ===")

        words = []
        for idx in idxs:
            words.extend([
                w for w in tokenized_docs[idx]
                if w not in ENGLISH_STOP_WORDS and w not in BAD_TOKENS
            ])

        top_words = Counter(words).most_common(10)
        print("Top words:", top_words)

        for idx in idxs[:n]:
            print("-", docs[idx][:max_len])


def compute_cluster_balance(labels):
    cluster_sizes = Counter(labels)
    size_values = list(cluster_sizes.values())

    return {
        "cluster_sizes": dict(cluster_sizes),
        "largest_cluster_ratio": max(size_values) / len(labels),
        "smallest_cluster_size": min(size_values),
        "num_tiny_clusters": sum(1 for s in size_values if s < 20),
        "std_cluster_size": float(np.std(size_values)),
    }


clean_texts = [" ".join(doc) for doc in tokenized_docs]


# Task 1: Doc2Vec Document Embedding and Clustering

tagged_docs = [
    TaggedDocument(words=doc, tags=[i])
    for i, doc in enumerate(tokenized_docs)
]

print("Sample TaggedDocument:", tagged_docs[0])


def train_doc2vec_get_X(
    tagged_docs,
    vector_size,
    min_count,
    epochs,
    window=5,
    workers=1,
    dm=1,
    negative=5,
    hs=0,
    sample=1e-4,
    seed=42,
):
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        workers=workers,
        dm=dm,
        negative=negative,
        hs=hs,
        sample=sample,
        seed=seed
    )

    model.build_vocab(tagged_docs)

    model.train(
        tagged_docs,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    vectors = np.vstack([
        model.dv[i] for i in range(len(tagged_docs))
    ]).astype(np.float32)

    X = normalize(vectors, norm="l2")
    return model, vectors, X



K_FIXED = 8

doc2vec_configs = [
    {"name": "D1", "vector_size": 50,  "min_count": 2, "epochs": 20},
    {"name": "D2", "vector_size": 100, "min_count": 3, "epochs": 30},
    {"name": "D3", "vector_size": 200, "min_count": 5, "epochs": 40},
]

doc2vec_vs_results = []

for cfg in doc2vec_configs:

    print(f"\nRunning Doc2Vec {cfg['name']} | "
          f"vector_size={cfg['vector_size']}, "
          f"min_count={cfg['min_count']}, "
          f"epochs={cfg['epochs']}")

    model, vectors, X = train_doc2vec_get_X(
        tagged_docs,
        vector_size=cfg["vector_size"],
        min_count=cfg["min_count"],
        epochs=cfg["epochs"],
        dm=1,
        seed=42,
        workers=1
    )

    km = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    metrics = evaluate_clustering(X, labels)
    balance = compute_cluster_balance(labels)

    doc2vec_vs_results.append({
        "config": cfg["name"],
        "vector_size": cfg["vector_size"],
        "min_count": cfg["min_count"],
        "epochs": cfg["epochs"],
        "silhouette_cosine": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],
        "largest_cluster_ratio": balance["largest_cluster_ratio"],
        "smallest_cluster_size": balance["smallest_cluster_size"],
        "num_tiny_clusters": balance["num_tiny_clusters"],
        "std_cluster_size": balance["std_cluster_size"],
    })

df_doc2vec_vs = pd.DataFrame(doc2vec_vs_results).sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n=== Doc2Vec configuration comparison ===")
display(df_doc2vec_vs)


best_cfg = {"name": "D1", "vector_size": 50, "min_count": 2, "epochs": 20}

model, vectors, X = train_doc2vec_get_X(
    tagged_docs,
    vector_size=best_cfg["vector_size"],
    min_count=best_cfg["min_count"],
    epochs=best_cfg["epochs"],
    dm=1,
    seed=42,
    workers=1
)

kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

top_words = top_keywords_per_cluster(clean_texts, labels, top_n=10)
reps = representative_docs(X, labels, clean_texts, per_cluster=3, records=clean_records)

save_pca_plot(vectors, labels, "doc2vec_best_pca.png", "Best Doc2Vec Clusters")
show_cluster_samples(labels, clean_texts, tokenized_docs, n=3)


docs = [" ".join(toks) for toks in tokenized_docs]
records = clean_records

vector_size_grid = [50, 100, 200]
min_count_grid = [2, 3, 5]
epochs_grid = [10, 20, 30, 40]
dm_grid = [0, 1]

task1_search_results = []

for vs, mc, ep, dm in product(vector_size_grid, min_count_grid, epochs_grid, dm_grid):
    print(f"Running: vector_size={vs}, min_count={mc}, epochs={ep}, dm={dm}")

    model, vectors, X = train_doc2vec_get_X(
        tagged_docs,
        vector_size=vs,
        min_count=mc,
        epochs=ep,
        dm=dm,
        seed=42,
        workers=1
    )

    kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    metrics = evaluate_clustering(X, labels)
    balance = compute_cluster_balance(labels)
    top_words = top_keywords_per_cluster(docs, labels, top_n=10)
    reps = representative_docs(X, labels, docs, per_cluster=3, records=records)

    task1_search_results.append({
        "vector_size": vs,
        "min_count": mc,
        "epochs": ep,
        "dm": dm,

        "model": model,
        "vectors": vectors,
        "X": X,
        "labels": labels,

        "silhouette_cosine": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],

        "cluster_sizes": balance["cluster_sizes"],
        "largest_cluster_ratio": balance["largest_cluster_ratio"],
        "smallest_cluster_size": balance["smallest_cluster_size"],
        "num_tiny_clusters": balance["num_tiny_clusters"],
        "std_cluster_size": balance["std_cluster_size"],

        "top_words": top_words,
        "representative_docs": reps,
    })

df_task1_search = pd.DataFrame([
    {
        "vector_size": r["vector_size"],
        "min_count": r["min_count"],
        "epochs": r["epochs"],
        "dm": r["dm"],
        "silhouette_cosine": r["silhouette_cosine"],
        "calinski_harabasz": r["calinski_harabasz"],
        "davies_bouldin": r["davies_bouldin"],
        "largest_cluster_ratio": r["largest_cluster_ratio"],
        "smallest_cluster_size": r["smallest_cluster_size"],
        "num_tiny_clusters": r["num_tiny_clusters"],
        "std_cluster_size": r["std_cluster_size"],
    }
    for r in task1_search_results
])

df_task1_search = df_task1_search.sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

df_task1_search["is_degenerate"] = (
    (df_task1_search["largest_cluster_ratio"] > 0.7) |
    (df_task1_search["smallest_cluster_size"] < 20) |
    (df_task1_search["num_tiny_clusters"] > 0)
)

print("\n=== All Doc2Vec search results ===")
display(df_task1_search)


df_metric_ranked = df_task1_search.sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n=== Metric-best ranking ===")
display(df_metric_ranked.head(10))

best_metric_row = df_metric_ranked.iloc[0]

best_metric_result = next(
    r for r in task1_search_results
    if r["vector_size"] == best_metric_row["vector_size"]
    and r["min_count"] == best_metric_row["min_count"]
    and r["epochs"] == best_metric_row["epochs"]
    and r["dm"] == best_metric_row["dm"]
)

print("\n=== Best model by internal metrics ===")
print(best_metric_row)

print("\nCluster sizes (metric-best):")
print(best_metric_result["cluster_sizes"])

save_pca_plot(
    best_metric_result["X"],
    best_metric_result["labels"],
    "doc2vec_metric_best.png",
    "Doc2Vec Metric-Best Model"
)

print("\n=== Metric-best cluster samples ===")
show_cluster_samples(
    best_metric_result["labels"],
    docs,
    tokenized_docs,
    n=3
)

df_reasonable = df_task1_search[
    (df_task1_search["largest_cluster_ratio"] < 0.50) &
    (df_task1_search["num_tiny_clusters"] == 0) &
    (df_task1_search["smallest_cluster_size"] >= 30) &
    (df_task1_search["silhouette_cosine"] >= 0.20)
].copy()

df_semantic_ranked = df_reasonable.sort_values(
    by=[
        "silhouette_cosine",
        "calinski_harabasz",
        "davies_bouldin",
        "largest_cluster_ratio",
        "std_cluster_size"
    ],
    ascending=[False, False, True, True, True]
).reset_index(drop=True)

print("\n=== Semantic candidate ranking ===")
display(df_semantic_ranked.head(10))

top_semantic_candidates = df_semantic_ranked.head(3).to_dict("records")

semantic_candidates_results = []

for row in top_semantic_candidates:
    cfg = {
        "vector_size": int(row["vector_size"]),
        "min_count": int(row["min_count"]),
        "epochs": int(row["epochs"]),
        "dm": int(row["dm"]),
    }

    result = next(
        r for r in task1_search_results
        if r["vector_size"] == cfg["vector_size"]
        and r["min_count"] == cfg["min_count"]
        and r["epochs"] == cfg["epochs"]
        and r["dm"] == cfg["dm"]
    )

    semantic_candidates_results.append((cfg, result))

    print("\n" + "=" * 80)
    print("Semantic candidate:", cfg)
    print(
        f"silhouette={result['silhouette_cosine']:.4f}, "
        f"CH={result['calinski_harabasz']:.2f}, "
        f"DB={result['davies_bouldin']:.4f}"
    )
    print("Cluster sizes:", result["cluster_sizes"])

    print("\nTop keywords by cluster:")
    for c, kws in result["top_words"].items():
        print(f"Cluster {c}: {kws}")

    print("\nRepresentative docs by cluster:")
    for c, items in result["representative_docs"].items():
        print(f"\nCluster {c}")
        for item in items:
            title = item.get("title", "")
            preview = item.get("preview", "")
            print(f"- {title} | {preview}")

    save_pca_plot(
        result["X"],
        result["labels"],
        f"doc2vec_semantic_candidate_vs{cfg['vector_size']}_mc{cfg['min_count']}_ep{cfg['epochs']}_dm{cfg['dm']}.png",
        f"Doc2Vec Candidate vs={cfg['vector_size']} mc={cfg['min_count']} ep={cfg['epochs']} dm={cfg['dm']}"
    )

    print("\nCluster samples:")
    show_cluster_samples(
        result["labels"],
        docs,
        tokenized_docs,
        n=3
    )


semantic_choice_d2v = {
    "vector_size": 200,
    "min_count": 3,
    "epochs": 20,
    "dm": 1
}

best_semantic_d2v_result = next(
    r for r in task1_search_results
    if r["vector_size"] == semantic_choice_d2v["vector_size"]
    and r["min_count"] == semantic_choice_d2v["min_count"]
    and r["epochs"] == semantic_choice_d2v["epochs"]
    and r["dm"] == semantic_choice_d2v["dm"]
)

print("\n=== Final semantic-best Doc2Vec model ===")
print(semantic_choice_d2v)
print("Cluster sizes:", best_semantic_d2v_result["cluster_sizes"])


save_pca_plot(
    best_semantic_d2v_result["X"],
    best_semantic_d2v_result["labels"],
    "doc2vec_semantic_best.png",
    "Doc2Vec Semantic-Best Model"
)

show_cluster_samples(
    best_semantic_d2v_result["labels"],
    docs,
    tokenized_docs,
    n=3
)


# Task 2: Word2Vec-Based Document Representation and Clustering

sentences = tokenized_docs   # list[list[str]]
print("Number of sentences:", len(sentences))

def cluster_words_into_bins(word_vecs, k):
    word_vecs_norm = normalize(word_vecs, norm="l2")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    word_labels = km.fit_predict(word_vecs_norm)

    return word_labels


def build_word_to_bin(words, word_labels):
    assert len(words) == len(word_labels), "words and word_labels are not aligned."
    return {w: int(lbl) for w, lbl in zip(words, word_labels)}


def docs_to_bin_freq_vectors(tokenized_docs, word_to_bin, k):
    X = np.zeros((len(tokenized_docs), k), dtype=np.float32)
    zero_doc_count = 0

    for i, tokens in enumerate(tokenized_docs):
        count = 0

        for tok in tokens:
            if tok in word_to_bin:
                X[i, word_to_bin[tok]] += 1.0
                count += 1

        if count > 0:
            X[i] /= count
        else:
            zero_doc_count += 1

    return X, zero_doc_count


def train_word2vec_get_wordvecs(
    sentences,
    vector_size,
    min_count,
    epochs,
    window=5,
    workers=1,
    sg=0,
    negative=5,
    hs=0,
    sample=1e-4,
    seed=42,
):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        workers=workers,
        sg=sg,
        negative=negative,
        hs=hs,
        sample=sample,
        seed=seed,
        epochs=epochs
    )

    words = list(model.wv.index_to_key)
    word_vecs = np.vstack([model.wv[w] for w in words]).astype(np.float32)

    return model, words, word_vecs


w2v_baseline_configs = [
    {"name": "W1", "vector_size": 50,  "min_count": 2, "epochs": 20, "sg": 0},
    {"name": "W2", "vector_size": 100, "min_count": 3, "epochs": 30, "sg": 0},
    {"name": "W3", "vector_size": 200, "min_count": 5, "epochs": 40, "sg": 0},
]

K_FIXED = 8
w2v_results = []
w2v_models = {}

for cfg in w2v_baseline_configs:
    print(
        f"\nTraining Word2Vec {cfg['name']} | "
        f"vector_size={cfg['vector_size']}, "
        f"min_count={cfg['min_count']}, "
        f"epochs={cfg['epochs']}, "
        f"sg={cfg['sg']}"
    )

    model, words, word_vecs = train_word2vec_get_wordvecs(
        sentences=sentences,
        vector_size=cfg["vector_size"],
        min_count=cfg["min_count"],
        epochs=cfg["epochs"],
        sg=cfg["sg"],
        workers=1,
        seed=42
    )

    w2v_models[cfg["name"]] = {
        "config": cfg,
        "model": model,
        "words": words,
        "word_vecs": word_vecs
    }

    print("Vocab size:", len(words))
    print("Word vector shape:", word_vecs.shape)

    km = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    metrics = evaluate_clustering(X, labels)
    balance = compute_cluster_balance(labels)

    w2v_results.append({
        "config": cfg["name"],
        "vector_size": cfg["vector_size"],
        "min_count": cfg["min_count"],
        "epochs": cfg["epochs"],
        "silhouette_cosine": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],
        "largest_cluster_ratio": balance["largest_cluster_ratio"],
        "smallest_cluster_size": balance["smallest_cluster_size"],
        "num_tiny_clusters": balance["num_tiny_clusters"],
        "std_cluster_size": balance["std_cluster_size"],
    })
    

w2v_baseline_results = []

for cfg in w2v_baseline_configs:
    print(f"\nRunning Word2Vec+BoW {cfg['name']} | "
          f"vector_size={cfg['vector_size']}, "
          f"min_count={cfg['min_count']}, "
          f"epochs={cfg['epochs']}, sg={cfg['sg']}")

    info = w2v_models[cfg["name"]]
    words = info["words"]
    word_vecs = info["word_vecs"]
    num_bins = cfg["vector_size"]

    word_labels = cluster_words_into_bins(word_vecs, num_bins)
    word_to_bin = build_word_to_bin(words, word_labels)

    X_doc, zero_count = docs_to_bin_freq_vectors(tokenized_docs, word_to_bin, num_bins)
    print("Zero-vector docs:", zero_count)

    X_doc = normalize(X_doc, norm="l2")

    row_norms = np.linalg.norm(X_doc, axis=1)
    nonzero = row_norms > 0
    X_use = X_doc[nonzero]

    km = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    labels = km.fit_predict(X_use)

    metrics = evaluate_clustering(X_use, labels)
    balance = compute_cluster_balance(labels)

    w2v_baseline_results.append({
        "config": cfg["name"],
        "vector_size": cfg["vector_size"],
        "min_count": cfg["min_count"],
        "epochs": cfg["epochs"],
        "sg": cfg["sg"],
        "num_bins": num_bins,
        "silhouette_cosine": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],
        "largest_cluster_ratio": balance["largest_cluster_ratio"],
        "smallest_cluster_size": balance["smallest_cluster_size"],
        "num_tiny_clusters": balance["num_tiny_clusters"],
        "std_cluster_size": balance["std_cluster_size"],
    })

df_w2v_baseline = pd.DataFrame(w2v_baseline_results).sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n=== Word2Vec+BoW configuration comparison ===")
display(df_w2v_baseline)

best_w2v_cfg = {"name": "W1", "vector_size": 50, "min_count": 2, "epochs": 20, "sg": 0}
num_bins = best_w2v_cfg["vector_size"]

# 1) train Word2Vec
model, words, word_vecs = train_word2vec_get_wordvecs(
    sentences=sentences,
    vector_size=best_w2v_cfg["vector_size"],
    min_count=best_w2v_cfg["min_count"],
    epochs=best_w2v_cfg["epochs"],
    sg=best_w2v_cfg["sg"],
    workers=1,
    seed=42
)

# 2) cluster words into semantic bins
word_labels = cluster_words_into_bins(word_vecs, num_bins)
word_to_bin = build_word_to_bin(words, word_labels)

# 3) convert documents to normalized bin-frequency vectors
X_doc, zero_doc_count = docs_to_bin_freq_vectors(tokenized_docs, word_to_bin, num_bins)
print("Zero-vector docs:", zero_doc_count)

X_doc = normalize(X_doc, norm="l2")

# 4) drop zero vectors if any
row_norms = np.linalg.norm(X_doc, axis=1)
nonzero = row_norms > 0

X_use = X_doc[nonzero]
docs_use = [d for d, keep in zip(docs, nonzero) if keep]
tokenized_use = [t for t, keep in zip(tokenized_docs, nonzero) if keep]
records_use = [r for r, keep in zip(records, nonzero) if keep]

# 5) document clustering
kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_use)

# 6) interpretation
top_words = top_keywords_per_cluster(docs_use, labels, top_n=10)
reps = representative_docs(X_use, labels, docs_use, per_cluster=3, records=records_use)

save_pca_plot(X_use, labels, "w2v_best_baseline_pca.png", "Best Baseline Word2Vec Clusters")
show_cluster_samples(labels, docs_use, tokenized_use, n=3)

print("\n=== Top keywords per cluster (best baseline Word2Vec) ===")
for c, kws in top_words.items():
    print(f"Cluster {c}: {kws}")

vector_size_grid = [50, 100, 200]
min_count_grid = [2, 3, 5]
epochs_grid = [10, 20, 30, 40]
sg_grid = [0, 1]

task2_search_results = []

for vs, mc, ep, sg in product(vector_size_grid, min_count_grid, epochs_grid, sg_grid):
    num_bins = vs   # keep num_bins matched to embedding size

    print(
        f"Running Word2Vec | "
        f"vector_size={vs}, min_count={mc}, epochs={ep}, sg={sg}, num_bins={num_bins}"
    )

    # 1) train Word2Vec and get word vectors
    model, words, word_vecs = train_word2vec_get_wordvecs(
        sentences=sentences,
        vector_size=vs,
        min_count=mc,
        epochs=ep,
        sg=sg,
        workers=1,
        seed=42
    )

    # 2) word -> semantic bin
    word_labels = cluster_words_into_bins(word_vecs, num_bins)
    word_to_bin = build_word_to_bin(words, word_labels)

    # 3) docs -> normalized bin-frequency vectors
    X_doc, zero_doc_count = docs_to_bin_freq_vectors(tokenized_docs, word_to_bin, num_bins)
    print("Zero-vector docs:", zero_doc_count)

    X_doc = normalize(X_doc, norm="l2")

    # 4) drop zero vectors if any
    row_norms = np.linalg.norm(X_doc, axis=1)
    nonzero = row_norms > 0

    if not nonzero.all():
        print("[WARN] Dropping all-zero doc vectors:", int((~nonzero).sum()))

    X_use = X_doc[nonzero]
    docs_use = [d for d, keep in zip(docs, nonzero) if keep]
    tokenized_use = [t for t, keep in zip(tokenized_docs, nonzero) if keep]
    records_use = [r for r, keep in zip(records, nonzero) if keep]

    # 5) document clustering
    km = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    doc_labels = km.fit_predict(X_use)

    # 6) internal metrics
    metrics = evaluate_clustering(X_use, doc_labels)

    # 7) cluster balance
    balance = compute_cluster_balance(doc_labels)

    # 8) semantic inspection signals
    top_words = top_keywords_per_cluster(docs_use, doc_labels, top_n=10)
    reps = representative_docs(
        X_use,
        doc_labels,
        docs_use,
        per_cluster=3,
        records=records_use
    )

    task2_search_results.append({
        "vector_size": vs,
        "min_count": mc,
        "epochs": ep,
        "sg": sg,
        "num_bins": num_bins,

        "model": model,
        "words": words,
        "word_vecs": word_vecs,

        "X": X_use,
        "labels": doc_labels,
        "docs_use": docs_use,
        "tokenized_use": tokenized_use,
        "records_use": records_use,

        "silhouette_cosine": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],

        "cluster_sizes": balance["cluster_sizes"],
        "largest_cluster_ratio": balance["largest_cluster_ratio"],
        "smallest_cluster_size": balance["smallest_cluster_size"],
        "num_tiny_clusters": balance["num_tiny_clusters"],
        "std_cluster_size": balance["std_cluster_size"],

        "top_words": top_words,
        "representative_docs": reps,
    })

df_task2_search = pd.DataFrame([
    {
        "vector_size": r["vector_size"],
        "min_count": r["min_count"],
        "epochs": r["epochs"],
        "sg": r["sg"],
        "num_bins": r["num_bins"],
        "silhouette_cosine": r["silhouette_cosine"],
        "calinski_harabasz": r["calinski_harabasz"],
        "davies_bouldin": r["davies_bouldin"],
        "largest_cluster_ratio": r["largest_cluster_ratio"],
        "smallest_cluster_size": r["smallest_cluster_size"],
        "num_tiny_clusters": r["num_tiny_clusters"],
        "std_cluster_size": r["std_cluster_size"],
    }
    for r in task2_search_results
]).sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n=== All Word2Vec search results ===")
display(df_task2_search)


df_task2_metric_ranked = df_task2_search.sort_values(
    by=["silhouette_cosine", "calinski_harabasz", "davies_bouldin"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n=== Word2Vec metric-best ranking ===")
display(df_task2_metric_ranked.head(10))

best_metric_w2v_row = df_task2_metric_ranked.iloc[0]

best_metric_w2v_result = next(
    r for r in task2_search_results
    if r["vector_size"] == best_metric_w2v_row["vector_size"]
    and r["min_count"] == best_metric_w2v_row["min_count"]
    and r["epochs"] == best_metric_w2v_row["epochs"]
    and r["sg"] == best_metric_w2v_row["sg"]
)

print("\n=== Best Word2Vec model by internal metrics ===")
print(best_metric_w2v_row)

print("\nCluster sizes (metric-best Word2Vec):")
print(best_metric_w2v_result["cluster_sizes"])


save_pca_plot(
    best_metric_w2v_result["X"],
    best_metric_w2v_result["labels"],
    "w2v_metric_best.png",
    "Word2Vec Metric-Best Model"
)

print("\n=== Metric-best Word2Vec cluster samples ===")
show_cluster_samples(
    best_metric_w2v_result["labels"],
    best_metric_w2v_result["docs_use"],
    best_metric_w2v_result["tokenized_use"],
    n=3
)

print("\n=== Top keywords per cluster (metric-best Word2Vec) ===")
for c, kws in best_metric_w2v_result["top_words"].items():
    print(f"Cluster {c}: {kws}")


df_task2_reasonable = df_task2_search[
    (df_task2_search["largest_cluster_ratio"] < 0.50) &
    (df_task2_search["num_tiny_clusters"] == 0) &
    (df_task2_search["smallest_cluster_size"] >= 30) &
    (df_task2_search["silhouette_cosine"] >= 0.18)
].copy()

df_task2_semantic_ranked = df_task2_reasonable.sort_values(
    by=[
        "silhouette_cosine",
        "calinski_harabasz",
        "davies_bouldin",
        "largest_cluster_ratio",
        "std_cluster_size"
    ],
    ascending=[False, False, True, True, True]
).reset_index(drop=True)

print("\n=== Word2Vec semantic candidate ranking ===")
display(df_task2_semantic_ranked.head(10))


top_semantic_candidates_w2v = df_task2_semantic_ranked.head(3).to_dict("records")

semantic_candidates_results_w2v = []

for row in top_semantic_candidates_w2v:
    cfg = {
        "vector_size": int(row["vector_size"]),
        "min_count": int(row["min_count"]),
        "epochs": int(row["epochs"]),
        "sg": int(row["sg"]),
        "num_bins": int(row["num_bins"]),
    }

    result = next(
        r for r in task2_search_results
        if r["vector_size"] == cfg["vector_size"]
        and r["min_count"] == cfg["min_count"]
        and r["epochs"] == cfg["epochs"]
        and r["sg"] == cfg["sg"]
    )

    semantic_candidates_results_w2v.append((cfg, result))

    print("\n" + "=" * 80)
    print("Word2Vec semantic candidate:", cfg)
    print(
        f"silhouette={result['silhouette_cosine']:.4f}, "
        f"CH={result['calinski_harabasz']:.2f}, "
        f"DB={result['davies_bouldin']:.4f}"
    )
    print("Cluster sizes:", result["cluster_sizes"])

    print("\nTop keywords by cluster:")
    for c, kws in result["top_words"].items():
        print(f"Cluster {c}: {kws}")

    print("\nRepresentative docs by cluster:")
    for c, items in result["representative_docs"].items():
        print(f"\nCluster {c}")
        for item in items[:2]:
            title = item.get("title", "")
            preview = item.get("preview", "")
            print(f"- {title} | {preview}")

    save_pca_plot(
        result["X"],
        result["labels"],
        f"w2v_semantic_candidate_vs{cfg['vector_size']}_mc{cfg['min_count']}_ep{cfg['epochs']}_sg{cfg['sg']}.png",
        f"Word2Vec Candidate vs={cfg['vector_size']} mc={cfg['min_count']} ep={cfg['epochs']} sg={cfg['sg']}"
    )

    print("\nCluster samples:")
    show_cluster_samples(
        result["labels"],
        result["docs_use"],
        result["tokenized_use"],
        n=3
    )

semantic_choice_w2v = {
    "vector_size": 50,
    "min_count": 2,
    "epochs": 40,
    "sg": 1
}

best_semantic_w2v_result = next(
    r for r in task2_search_results
    if r["vector_size"] == semantic_choice_w2v["vector_size"]
    and r["min_count"] == semantic_choice_w2v["min_count"]
    and r["epochs"] == semantic_choice_w2v["epochs"]
    and r["sg"] == semantic_choice_w2v["sg"]
)

print("\n=== Final semantic-best Word2Vec model ===")
print(semantic_choice_w2v)
print("Cluster sizes:", best_semantic_w2v_result["cluster_sizes"])


save_pca_plot(
    best_semantic_w2v_result["X"],
    best_semantic_w2v_result["labels"],
    "w2v_semantic_best.png",
    "Word2Vec Semantic-Best Model"
)

show_cluster_samples(
    best_semantic_w2v_result["labels"],
    best_semantic_w2v_result["docs_use"],
    best_semantic_w2v_result["tokenized_use"],
    n=3
)


# Comparative Analysis

final_compare = pd.DataFrame([
    {
        "Model": "Doc2Vec semantic-best",
        "vector_size": 200,
        "other": "min_count=3, epochs=20, dm=1",
        "silhouette_cosine": best_semantic_d2v_result["silhouette_cosine"],
        "calinski_harabasz": best_semantic_d2v_result["calinski_harabasz"],
        "davies_bouldin": best_semantic_d2v_result["davies_bouldin"],
        "largest_cluster_ratio": best_semantic_d2v_result["largest_cluster_ratio"],
        "smallest_cluster_size": best_semantic_d2v_result["smallest_cluster_size"],
        "std_cluster_size": best_semantic_d2v_result["std_cluster_size"],
    },
    {
        "Model": "Word2Vec semantic-best",
        "vector_size": 50,
        "other": "min_count=2, epochs=40, sg=1",
        "silhouette_cosine": best_semantic_w2v_result["silhouette_cosine"],
        "calinski_harabasz": best_semantic_w2v_result["calinski_harabasz"],
        "davies_bouldin": best_semantic_w2v_result["davies_bouldin"],
        "largest_cluster_ratio": best_semantic_w2v_result["largest_cluster_ratio"],
        "smallest_cluster_size": best_semantic_w2v_result["smallest_cluster_size"],
        "std_cluster_size": best_semantic_w2v_result["std_cluster_size"],
    }
])

print("\n=== Final comparison: Doc2Vec vs Word2Vec semantic-best models ===")
display(final_compare)

get_ipython().system('jupyter nbconvert --to script "lab 8.ipynb"')

