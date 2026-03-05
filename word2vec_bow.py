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
from sklearn.preprocessing import normalize

from utils import prepare_data, run_config


# -- Word2Vec + BoW Configurations -----------------------------------
# doc_dim = number of word bins = final document vector dimension
# w2v_dim = internal Word2Vec word embedding dimension (for word clustering)

W2V_BOW_CONFIGS = [
    {"name": "w2v_bow_dim50",  "doc_dim": 50,  "w2v_dim": 100, "min_count": 3, "epochs": 20, "window": 5},
    {"name": "w2v_bow_dim100", "doc_dim": 100, "w2v_dim": 100, "min_count": 3, "epochs": 20, "window": 5},
    {"name": "w2v_bow_dim200", "doc_dim": 200, "w2v_dim": 100, "min_count": 3, "epochs": 20, "window": 5},
]


def train_word2vec(tokenized_docs, config):
    """Train a single Word2Vec model on the entire corpus."""
    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=config["w2v_dim"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        window=config["window"],
        workers=1,
        seed=42,
    )
    return w2v


def build_bow_vectors(tokenized_docs, w2v_model, num_bins):
    """
    1) Cluster word vectors into num_bins bins using KMeans
    2) For each document, count how many words fall into each bin
    3) Normalize by total in-vocab word count → document vector
    """
    vocab = list(w2v_model.wv.index_to_key)
    word_vectors = np.array([w2v_model.wv[w] for w in vocab], dtype=np.float32)

    # Cluster words into bins
    word_vecs_norm = normalize(word_vectors, norm="l2")
    km_words = KMeans(n_clusters=num_bins, random_state=42, n_init=10)
    bin_labels = km_words.fit_predict(word_vecs_norm)
    word_to_bin = {word: int(lbl) for word, lbl in zip(vocab, bin_labels)}

    # Build normalized bag-of-bins document vectors
    n_docs = len(tokenized_docs)
    doc_vectors = np.zeros((n_docs, num_bins), dtype=np.float32)

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

    # Train Word2Vec ONCE on all corpus words (params taken from first config)
    ref = W2V_BOW_CONFIGS[0]
    print(f"\n[INFO] Training Word2Vec (vector_size={ref['w2v_dim']}, "
          f"min_count={ref['min_count']}, epochs={ref['epochs']}) ...")
    w2v_model = train_word2vec(tokenized_docs, ref)
    print(f"[INFO] Word2Vec vocab size: {len(w2v_model.wv)}")

    all_results = []
    for config in W2V_BOW_CONFIGS:
        name = config["name"]
        num_bins = config["doc_dim"]

        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"  doc_dim(bins)={num_bins}, w2v_dim={config['w2v_dim']}")
        print(f"{'='*60}")

        print(f"  Building BoW vectors with {num_bins} bins ...")
        embeddings = build_bow_vectors(tokenized_docs, w2v_model, num_bins)
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