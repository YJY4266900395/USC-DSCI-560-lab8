"""
Part 2
Word2Vec word embeddings → word clustering into K bins →
Bag-of-Words document vectors (normalized word-bin frequency) →
KMeans clustering (cosine distance) + evaluation.
Uses the same 3 vector dimensions as Part 1 for comparison.
Usage:
    python word2vec_bow.py --input cleaned_5000.json --k 8 --out_dir outputs/w2v --plot
"""

import argparse
import json
import os
import numpy as np

from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from utils import prepare_data, run_config


# -- Word2Vec + BoW Configurations -----------------------------------
# doc_dim = number of word bins = final document vector dimension
# w2v_dim = internal Word2Vec word embedding dimension (for word clustering)

W2V_BOW_CONFIGS = [
    {"name": "w2v_bow_dim30",  "doc_dim": 30,  "w2v_dim": 100, "min_count": 3, "epochs": 30, "window": 5},
    {"name": "w2v_bow_dim50", "doc_dim": 50, "w2v_dim": 100, "min_count": 3, "epochs": 30, "window": 5},
    {"name": "w2v_bow_dim100", "doc_dim": 100, "w2v_dim": 100, "min_count": 3, "epochs": 30, "window": 5},
]


def compute_w2v_bow_embeddings(tokenized_docs, config):
    """
    1) Train Word2Vec on all words in the corpus
    2) Cluster word vectors into K bins using KMeans
    3) For each document, count how many words fall into each bin
    4) Normalize by total in-vocab word count → document vector
    """
    doc_dim = config["doc_dim"]

    # Train Word2Vec on words
    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=config["w2v_dim"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        window=config["window"],
        workers=4,
        seed=42,
    )
    vocab = list(w2v.wv.index_to_key)
    word_vectors = np.array([w2v.wv[w] for w in vocab], dtype=np.float32)
    print(f"        Word2Vec vocab size: {len(vocab)}")

    # Cluster words into bins
    km_words = KMeans(n_clusters=doc_dim, random_state=42, n_init="auto", max_iter=300)
    bin_labels = km_words.fit_predict(word_vectors)
    word_to_bin = {word: int(lbl) for word, lbl in zip(vocab, bin_labels)}

    # Build normalized bag-of-bins document vectors
    n_docs = len(tokenized_docs)
    doc_vectors = np.zeros((n_docs, doc_dim), dtype=np.float32)

    for i, tokens in enumerate(tokenized_docs):
        in_vocab = 0
        for tok in tokens:
            if tok in word_to_bin:
                doc_vectors[i, word_to_bin[tok]] += 1.0
                in_vocab += 1
        if in_vocab > 0:
            doc_vectors[i] /= in_vocab

    n_zero = np.all(doc_vectors == 0, axis=1).sum()
    if n_zero > 0:
        print(f"        [WARN] {n_zero} documents have all-zero vectors (no in-vocab tokens)")

    return doc_vectors


def main():
    ap = argparse.ArgumentParser(description="Lab 8 Part 2: Word2Vec + BoW Embeddings + Clustering")
    ap.add_argument("--input", required=True, help="Path to cleaned JSON/JSONL")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters")
    ap.add_argument("--out_dir", default="outputs/w2v")
    ap.add_argument("--plot", action="store_true", help="Generate PCA plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records, texts, tokenized_docs = prepare_data(args.input)

    all_results = []
    for config in W2V_BOW_CONFIGS:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"  doc_dim(bins)={config['doc_dim']}, w2v_dim={config['w2v_dim']}, "
              f"min_count={config['min_count']}, epochs={config['epochs']}, "
              f"window={config['window']}")
        print(f"{'='*60}")

        print(f"  Training Word2Vec + BoW (doc_dim={config['doc_dim']}) ...")
        embeddings = compute_w2v_bow_embeddings(tokenized_docs, config)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        result = run_config(
            method="w2v_bow", config=config, embeddings=embeddings,
            records=records, texts=texts, k=args.k, out_dir=args.out_dir,
            do_plot=args.plot,
        )
        all_results.append(result)

    # Save aggregated results
    results_path = os.path.join(args.out_dir, "w2v_bow_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Wrote {results_path}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'Config':<18} {'Dim':>4} {'Silhouette':>11} {'CH Index':>12} {'DB Index':>10}")
    print(f"  {'-'*18} {'-'*4} {'-'*11} {'-'*12} {'-'*10}")
    for r in all_results:
        print(f"  {r['config_name']:<18} {r['dim']:>4} "
              f"{r['silhouette']:>11.4f} {r['calinski_harabasz']:>12.2f} {r['davies_bouldin']:>10.4f}")

    best = max(all_results, key=lambda r: r["silhouette"])
    print(f"\n  Best by Silhouette: {best['config_name']} ({best['silhouette']:.4f})")


if __name__ == "__main__":
    main()
