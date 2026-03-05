"""
Part 1
Doc2Vec embeddings with 3 different configurations,
KMeans clustering (cosine distance), and evaluation.
Usage:
    python doc2vec.py --input cleaned_5000.json --k 8 --out_dir outputs/d2v --plot
"""

import argparse
import json
import os
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils import prepare_data, run_config


# -- Doc2Vec Configurations ------------------------------------------
# Three configs with DIFFERENT vector sizes (required by assignment)

DOC2VEC_CONFIGS = [
    {"name": "d2v_dim30",  "vector_size": 30,  "min_count": 3, "epochs": 30, "window": 5,  "dm": 1},
    {"name": "d2v_dim50", "vector_size": 50, "min_count": 3, "epochs": 30, "window": 5,  "dm": 1},
    {"name": "d2v_dim100", "vector_size": 100, "min_count": 3, "epochs": 30, "window": 5, "dm": 1},
]


def compute_doc2vec_embeddings(tokenized_docs, config):
    """Train Doc2Vec and return document vectors."""
    tagged = [TaggedDocument(words=tokens, tags=[i]) for i, tokens in enumerate(tokenized_docs)]

    model = Doc2Vec(
        vector_size=config["vector_size"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        window=config["window"],
        dm=config["dm"],
        workers=4,
        seed=42,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

    vectors = np.array([model.dv[i] for i in range(len(tagged))], dtype=np.float32)
    return vectors


def main():
    ap = argparse.ArgumentParser(description="Lab 8 Part 1: Doc2Vec Embeddings + Clustering")
    ap.add_argument("--input", required=True, help="Path to cleaned JSON/JSONL")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters")
    ap.add_argument("--out_dir", default="outputs/d2v")
    ap.add_argument("--plot", action="store_true", help="Generate PCA plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records, texts, tokenized_docs = prepare_data(args.input)

    all_results = []
    for config in DOC2VEC_CONFIGS:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"  vector_size={config['vector_size']}, min_count={config['min_count']}, "
              f"epochs={config['epochs']}, window={config['window']}, dm={config['dm']}")
        print(f"{'='*60}")

        print(f"  Training Doc2Vec (vector_size={config['vector_size']}) ...")
        embeddings = compute_doc2vec_embeddings(tokenized_docs, config)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        result = run_config(
            method="doc2vec", config=config, embeddings=embeddings,
            records=records, texts=texts, k=args.k, out_dir=args.out_dir,
            do_plot=args.plot,
        )
        all_results.append(result)

    # Save aggregated results
    results_path = os.path.join(args.out_dir, "doc2vec_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Wrote {results_path}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'Config':<16} {'Dim':>4} {'Silhouette':>11} {'CH Index':>12} {'DB Index':>10}")
    print(f"  {'-'*16} {'-'*4} {'-'*11} {'-'*12} {'-'*10}")
    for r in all_results:
        print(f"  {r['config_name']:<16} {r['dim']:>4} "
              f"{r['silhouette']:>11.4f} {r['calinski_harabasz']:>12.2f} {r['davies_bouldin']:>10.4f}")

    best = max(all_results, key=lambda r: r["silhouette"])
    print(f"\n  Best by Silhouette: {best['config_name']} ({best['silhouette']:.4f})")


if __name__ == "__main__":
    main()
