import json
import re
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


DATA_PATH = os.path.join("..", "data", "posts_lab5_5000.jsonl")
OUT_DIR = os.path.join("..", "outputs", "w2v_bins")
os.makedirs(OUT_DIR, exist_ok=True)


def load_posts_jsonl(path):
    posts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = str(obj.get("title", "") or "")
            body = str(obj.get("selftext", "") or obj.get("text", "") or "")
            text = (title + " " + body).strip()
            post_id = obj.get("id", obj.get("post_id", i))
            posts.append({"post_id": post_id, "text": text})
    return posts

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) >= 2]
    return tokens


def build_doc_vectors(posts, bin_map, K):
    X = np.zeros((len(posts), K), dtype=np.float32)
    for i, p in enumerate(posts):
        toks = preprocess(p["text"])
        if not toks:
            continue
        cnt = 0
        for w in toks:
            b = bin_map.get(w)
            if b is None:
                continue
            X[i, b] += 1.0
            cnt += 1
        if cnt > 0:
            X[i, :] /= float(cnt)
    return X

def safe_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def main():
    posts = load_posts_jsonl(DATA_PATH)
    print("Loaded posts:", len(posts))

    tokenized = [preprocess(p["text"]) for p in posts]


    w2v_dim = 100
    w2v = Word2Vec(
        sentences=tokenized,
        vector_size=w2v_dim,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )

    vocab = list(w2v.wv.key_to_index.keys())
    W = np.array([w2v.wv[w] for w in vocab], dtype=np.float32)
    W = normalize(W)

    print("Word vocab size:", len(vocab))

    configs = [
        {"name": "cfg1_bins50", "K": 50},
        {"name": "cfg2_bins100", "K": 100},
        {"name": "cfg3_bins200", "K": 200},
    ]

    for cfg in configs:
        K = cfg["K"]
        print("\nRunning", cfg["name"])


        km_words = KMeans(n_clusters=K, random_state=0, n_init="auto")
        word_bins = km_words.fit_predict(W)

        bin_map = {vocab[i]: int(word_bins[i]) for i in range(len(vocab))}


        X = build_doc_vectors(posts, bin_map, K)
        X = safe_normalize(X)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        row_norm = np.linalg.norm(X, axis=1)
        valid_mask = row_norm > 0

        X_clean = X[valid_mask]

        km_docs = KMeans(n_clusters=10, random_state=0)
        labels_clean = km_docs.fit_predict(X_clean)

        labels = np.full(len(X), -1)
        labels[valid_mask] = labels_clean

        df = pd.DataFrame({
            "post_id": [p["post_id"] for p in posts],
            "cluster_id": labels,
            "text": [p["text"] for p in posts]
        })

        out_csv = os.path.join(OUT_DIR, f"{cfg['name']}_clusters.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print("Saved:", out_csv)

        for c in range(3):
            sample = df[df["cluster_id"] == c].head(5)
            print(f"\nCluster {c} sample:")
            for t in sample["text"].tolist():
                print("-", t[:120].replace("\n", " "))

if __name__ == "__main__":
    main()