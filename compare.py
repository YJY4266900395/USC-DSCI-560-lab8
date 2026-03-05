"""
Part 3
Comparative Analysis: Doc2Vec vs Word2Vec+BoW
Loads results JSON from Part 1 and Part 2, generates:
  - Side-by-side metric bar charts (per dimension)
  - Per-method metric bar charts
  - Combined summary table
  - Combined results JSON
Usage:
    python compare.py --d2v_dir outputs/d2v --w2v_dir outputs/w2v --out_dir outputs
"""

import argparse
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison(d2v_results, w2v_results, out_dir):
    """Side-by-side bar charts: Doc2Vec vs Word2Vec+BoW at each dimension."""
    d2v_by_dim = {r["dim"]: r for r in d2v_results}
    w2v_by_dim = {r["dim"]: r for r in w2v_results}
    dims = sorted(set(d2v_by_dim.keys()) & set(w2v_by_dim.keys()))

    if not dims:
        print("[WARN] No matching dimensions to compare.")
        return

    metrics_info = [
        ("silhouette",        "Silhouette Score (higher = better)"),
        ("calinski_harabasz", "Calinski-Harabasz Index (higher = better)"),
        ("davies_bouldin",    "Davies-Bouldin Index (lower = better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(dims))
    width = 0.35

    for ax, (key, title) in zip(axes, metrics_info):
        d2v_vals = [d2v_by_dim[d][key] for d in dims]
        w2v_vals = [w2v_by_dim[d][key] for d in dims]

        bars1 = ax.bar(x - width/2, d2v_vals, width, label="Doc2Vec", color="#4C72B0")
        bars2 = ax.bar(x + width/2, w2v_vals, width, label="Word2Vec+BoW", color="#C44E52")

        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.legend(loc="upper left", fontsize=8)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("Doc2Vec vs Word2Vec+BoW - Clustering Quality Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison_d2v_vs_w2v_bow.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[DONE] Wrote {out_path}")


def plot_method_metrics(results, method_name, out_dir):
    """Bar chart comparing 3 configs within one method."""
    names = [r["config_name"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, title) in zip(axes, [
        ("silhouette",        "Silhouette Score"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("davies_bouldin",    "Davies-Bouldin"),
    ]):
        vals = [r[key] for r in results]
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle(f"{method_name} - Metrics Across Configurations", fontsize=13)
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("+", "_")
    out_path = os.path.join(out_dir, f"{safe_name}_metrics.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[DONE] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Lab 8 Part 3: Comparative Analysis")
    ap.add_argument("--d2v_dir", default="outputs/d2v", help="Directory with doc2vec_results.json")
    ap.add_argument("--w2v_dir", default="outputs/w2v", help="Directory with w2v_bow_results.json")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    d2v_path = os.path.join(args.d2v_dir, "doc2vec_results.json")
    w2v_path = os.path.join(args.w2v_dir, "w2v_bow_results.json")

    if not os.path.exists(d2v_path):
        print(f"[ERROR] Not found: {d2v_path}, run doc2vec.py first.")
        return
    if not os.path.exists(w2v_path):
        print(f"[ERROR] Not found: {w2v_path}, run word2vec_bow.py first.")
        return

    d2v_results = load_results(d2v_path)
    w2v_results = load_results(w2v_path)
    print(f"[INFO] Loaded {len(d2v_results)} Doc2Vec configs, {len(w2v_results)} Word2Vec+BoW configs")

    # -- Plots ------------------------------------------------------
    plot_method_metrics(d2v_results, "Doc2Vec", args.out_dir)
    plot_method_metrics(w2v_results, "Word2Vec+BoW", args.out_dir)
    plot_comparison(d2v_results, w2v_results, args.out_dir)

    # -- Summary table ----------------------------------------------
    print(f"\n{'='*78}")
    print(f"  {'Method':<16} {'Config':<18} {'Dim':>4} {'Silhouette':>11} {'CH Index':>12} {'DB Index':>10}")
    print(f"  {'-'*16} {'-'*18} {'-'*4} {'-'*11} {'-'*12} {'-'*10}")
    for r in d2v_results + w2v_results:
        print(f"  {r['method']:<16} {r['config_name']:<18} {r['dim']:>4} "
              f"{r['silhouette']:>11.4f} {r['calinski_harabasz']:>12.2f} {r['davies_bouldin']:>10.4f}")
    print(f"{'='*78}")

    # -- Per-dimension winner ---------------------------------------
    d2v_by_dim = {r["dim"]: r for r in d2v_results}
    w2v_by_dim = {r["dim"]: r for r in w2v_results}

    print(f"\n  Per-dimension comparison (by Silhouette Score):")
    for dim in sorted(set(d2v_by_dim.keys()) & set(w2v_by_dim.keys())):
        d = d2v_by_dim[dim]
        w = w2v_by_dim[dim]
        winner = "Doc2Vec" if d["silhouette"] > w["silhouette"] else "Word2Vec+BoW"
        print(f"    dim={dim:>3}: Doc2Vec={d['silhouette']:.4f}  "
              f"Word2Vec+BoW={w['silhouette']:.4f}  -> {winner}")

    # -- Save combined report ---------------------------------------
    report = {"doc2vec": d2v_results, "w2v_bow": w2v_results}
    report_path = os.path.join(args.out_dir, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Wrote {report_path}")


if __name__ == "__main__":
    main()
