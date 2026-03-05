"""
Reads raw scraped output from fetch_reddit.py,
do same processing with load_to_mysql.py in Lab5.
Usage:
    python clean_data.py posts_lab5_5000.jsonl --out_prefix cleaned
    python clean_data.py posts_lab5_5000.json  --out_prefix cleaned
"""

import argparse
import hashlib
import html
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# Cleaning utilities  (from load_jsonl_to_mysql.py)
# ════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","you","he","she",
    "they","we","me","him","her","them","us","my","your","our","their","from","not","no","yes",
    "about","into","over","under","after","before","between","during","than","too","very",
    "can","could","should","would","will","just","also","more","most","some","any","all",
}

TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z_]+")
WS_RE = re.compile(r"\s+")


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def mask_author(author: Optional[str]) -> str:
    if not author:
        return "user_unknown"
    if author.startswith("user_") or author.startswith("anon_") or author.startswith("masked_"):
        return author
    h = hashlib.sha256(author.encode("utf-8")).hexdigest()[:10]
    return f"user_{h}"


def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    text = text.lower()
    tokens = [t for t in NON_WORD_RE.split(text) if t]
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    if not tokens:
        return []
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for (w, _) in items[:top_k]]


# ════════════════════════════════════════════════════════════════════
# Row normalization  (from load_jsonl_to_mysql.py norm_row)
# ════════════════════════════════════════════════════════════════════

def norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    subreddit = row.get("subreddit") or row.get("topic") or ""
    title = clean_text(row.get("title"))
    body = clean_text(row.get("body"))
    final_text = clean_text(row.get("final_text") or (title + "\n" + body))

    ocr_text = clean_text(
        row.get("ocr_text")
        or row.get("image_text")
        or row.get("ocr")
        or ""
    )

    topic = clean_text(row.get("topic") or subreddit)

    kws = row.get("keywords")
    if isinstance(kws, list) and len(kws) > 0:
        keywords = [clean_text(str(x)).lower() for x in kws if str(x).strip()]
    else:
        keywords = extract_keywords((final_text + " " + ocr_text).strip(), top_k=12)

    author = mask_author(clean_text(row.get("author")))
    created = row.get("created")
    if isinstance(created, (int, float)):
        created = datetime.utcfromtimestamp(created).isoformat()
    created = clean_text(str(created)) if created is not None else ""

    return {
        "fullname": clean_text(row.get("fullname")),
        "subreddit": clean_text(subreddit),
        "title": title,
        "body": body,
        "final_text": final_text,
        "author": author,
        "created": created,
        "permalink": clean_text(row.get("permalink")),
        "out_url": clean_text(row.get("out_url")),
        "is_image": 1 if row.get("is_image") else 0,
        "topic": topic,
        "keywords": keywords,
        "ocr_text": ocr_text,
        # preserve extra fields from fetch_reddit.py
        "score": row.get("score", 0),
        "num_comments": row.get("num_comments", 0),
        "image_url": clean_text(row.get("image_url", "")),
    }


# ════════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════════

def load_raw(path: str) -> List[dict]:
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


def main():
    ap = argparse.ArgumentParser(description="Clean raw Reddit data from fetch_reddit.py")
    ap.add_argument("input", help="Path to raw JSON or JSONL from fetch_reddit.py")
    ap.add_argument("--out_prefix", default="cleaned", help="Output filename prefix (default: cleaned)")
    ap.add_argument("--min_text_len", type=int, default=20,
                    help="Drop records where final_text + ocr_text < this length (default: 20)")
    args = ap.parse_args()

    raw = load_raw(args.input)
    print(f"[INFO] Loaded {len(raw)} raw records from {args.input}")

    cleaned = []
    dropped = 0
    for row in raw:
        normed = norm_row(row)

        # Skip records without primary key
        if not normed["fullname"]:
            dropped += 1
            continue

        # Skip records with insufficient text
        combined = (normed["final_text"] + " " + normed["ocr_text"]).strip()
        if len(combined) < args.min_text_len:
            dropped += 1
            continue

        cleaned.append(normed)

    print(f"[INFO] Kept {len(cleaned)} records, dropped {dropped}")

    # Write outputs
    out_json = f"{args.out_prefix}_{len(cleaned)}.json"
    out_jsonl = f"{args.out_prefix}_{len(cleaned)}.jsonl"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {out_json}")
    print(f"[DONE] Wrote {out_jsonl}")

    print(f"\n[SAMPLE] first 3 cleaned records:")
    for p in cleaned[:3]:
        print(f"  - {p['subreddit']} | {p['title'][:60]} | keywords={p['keywords'][:5]}")


if __name__ == "__main__":
    main()