import json
import re
import os
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

DATA_PATH = os.path.join("..", "data", "posts_lab5_5000.jsonl")
OUT_DIR = os.path.join("..", "outputs", "doc2vec")
os.makedirs(OUT_DIR, exist_ok=True)

def load_posts_jsonl(path):
    posts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # 尽量兼容你们lab5字段
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

def run_one_config(posts, vector_size, epochs, min_count=2, n_clusters=10, seed=0):
    tokenized = [preprocess(p["text"]) for p in posts]
    docs = [TaggedDocument(words=tokenized[i], tags=[i]) for i in range(len(tokenized))]

    model = Doc2Vec(
        documents=docs,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        seed=seed
    )

    vectors = np.array([model.dv[i] for i in range(len(posts))])
    vectors = normalize(vectors)

    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    row_norm = np.linalg.norm(vectors, axis=1)
    valid_mask = row_norm > 0

    vectors_clean = vectors[valid_mask]

    kmeans = KMeans(n_clusters=10, random_state=0)
    labels_clean = kmeans.fit_predict(vectors_clean)

    labels = np.full(len(vectors), -1)
    labels[valid_mask] = labels_clean

    return vectors, labels

def main():
    posts = load_posts_jsonl(DATA_PATH)
    print("Loaded posts:", len(posts))

    configs = [
        {"name": "cfg1_vs50_ep20", "vector_size": 50, "epochs": 20},
        {"name": "cfg2_vs100_ep30", "vector_size": 100, "epochs": 30},
        {"name": "cfg3_vs200_ep40", "vector_size": 200, "epochs": 40},
    ]

    for cfg in configs:
        print("\nRunning", cfg["name"])
        vecs, labels = run_one_config(
            posts,
            vector_size=cfg["vector_size"],
            epochs=cfg["epochs"],
            min_count=2,
            n_clusters=10
        )

        df = pd.DataFrame({
            "post_id": [p["post_id"] for p in posts],
            "cluster_id": labels,
            "text": [p["text"] for p in posts]
        })

        out_csv = os.path.join(OUT_DIR, f"{cfg['name']}_clusters.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print("Saved:", out_csv)

        # 打印每个cluster的5条样例（用于后面写分析）
        for c in range(3):
            sample = df[df["cluster_id"] == c].head(5)
            print(f"\nCluster {c} sample:")
            for t in sample["text"].tolist():
                print("-", t[:120].replace("\n", " "))

if __name__ == "__main__":
    main()