"""
Build the frozen review corpus used in this assignment (run once; output is committed in data/).

Source
------
Amazon Reviews 2023, McAuley Lab (UC San Diego).
Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024).
"Bridging Language and Items for Retrieval and Recommendation". arXiv:2403.03952
Files: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023  (raw/review_categories/*.jsonl)

Sampling procedure (seeded, reproducible)
-----------------------------------------
1. For every selected product category we read N_CHUNKS byte ranges of CHUNK_BYTES at seeded random
   offsets of the category's JSONL file (HTTP Range requests), so the sample is spread over the whole file
   instead of only its first lines. Partial first/last lines of each chunk are discarded.
2. Reviews are kept if they have between MIN_WORDS and MAX_WORDS whitespace tokens and look English.
3. At most MAX_PER_USER reviews per user are kept (the files are grouped by user, so one prolific
   reviewer could otherwise dominate a chunk).
4. DOCS_PER_CLASS reviews are drawn per category.
5. Exact / near-duplicate reviews that occur naturally in the downloaded pool are retained, and a small,
   clearly marked set of near-duplicate variants is added (is_injected = True) so that duplicate detection
   and its effect on classification / clustering / recommendation can be measured against ground truth.
6. Documents are shuffled and given neutral IDs (R0001 ...), so the ID never reveals the category.

Usage:  python scripts/build_dataset.py            (downloads ~16 MB, writes data/)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

SEED = 537  # course code AIMLCZG537
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/"
TREE_URL = "https://huggingface.co/api/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories"

# file name -> human readable label used as the class / department
CATEGORIES = {
    "Video_Games": "Video Games",
    "Software": "Software",
    "Musical_Instruments": "Musical Instruments",
    "Appliances": "Appliances",
    "Pet_Supplies": "Pet Supplies",
    "All_Beauty": "Beauty",
}
N_CHUNKS = 40
CHUNK_BYTES = 64 * 1024
MIN_WORDS, MAX_WORDS = 30, 350
MAX_PER_USER = 2
DOCS_PER_CLASS = 200
N_INJECTED = 24  # documented near-duplicate variants (2% of the corpus)
NATURAL_GROUPS_PER_CLASS = 6  # naturally occurring duplicate groups retained per category

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.environ.get("AMZ_CACHE_DIR", os.path.join(ROOT, ".cache_raw"))

EN_HINT = {"the", "and", "is", "it", "to", "a", "of", "for", "this", "i", "my", "was", "in", "that", "with", "but",
           "not", "on", "they", "have", "you", "are", "so", "very", "be", "just"}


def http_get(url: str, headers: dict | None = None, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # network hiccup -> back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url[-60:]} ({exc})")
            time.sleep(2 * (attempt + 1))
    return b""


def file_sizes() -> dict:
    tree = json.loads(http_get(TREE_URL))
    return {f["path"].split("/")[-1].replace(".jsonl", ""): int(f["size"]) for f in tree}


def read_chunk(category: str, offset: int) -> list:
    """Return complete JSON records inside bytes [offset, offset+CHUNK_BYTES) (cached on disk)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{category}_{offset}.bin")
    if os.path.exists(cache):
        raw = open(cache, "rb").read()
    else:
        raw = http_get(BASE_URL + category + ".jsonl",
                       {"Range": f"bytes={offset}-{offset + CHUNK_BYTES - 1}"})
        with open(cache, "wb") as fh:
            fh.write(raw)
        time.sleep(0.15)
    lines = raw.decode("utf-8", errors="ignore").split("\n")[1:-1]  # drop partial first/last line
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def looks_english(text: str) -> bool:
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks:
        return False
    return sum(t in EN_HINT for t in toks) / len(toks) >= 0.12


def norm_for_hash(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def shingles(text: str, k: int = 3) -> set:
    toks = norm_for_hash(text).split()
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- injected near-duplicate variants
def make_variant(text: str, rng: random.Random, kind: str) -> str:
    """Create a realistic near-duplicate of a review (re-post / copy-paste with small edits)."""
    if kind == "casing_markup":
        # re-posted with shouting caps in the first sentence and HTML line breaks
        parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        head = parts[0].upper()
        tail = parts[1] if len(parts) > 1 else ""
        return (head + "<br /><br />" + tail).strip()
    if kind == "punctuation":
        out = re.sub(r"\.(\s|$)", "!!\\1", text, count=3)
        return out.replace(",", " ,").replace("  ", " ") + " :)"
    if kind == "edit":
        words = text.split()
        if len(words) > 12:
            # drop a couple of words and append a short follow-up, as in an edited re-post
            for _ in range(2):
                del words[rng.randrange(1, len(words) - 1)]
        return " ".join(words) + " UPDATE: still works fine after a few weeks."
    return text


def main() -> None:
    rng = random.Random(SEED)
    sizes = file_sizes()
    pool = defaultdict(list)
    for cat in CATEGORIES:
        size = sizes[cat]
        offsets = sorted(rng.randrange(0, size - CHUNK_BYTES) for _ in range(N_CHUNKS))
        n_raw = 0
        for off in offsets:
            recs = read_chunk(cat, off)
            n_raw += len(recs)
            for r in recs:
                text = (r.get("text") or "").strip()
                n_words = len(text.split())
                if MIN_WORDS <= n_words <= MAX_WORDS and looks_english(text):
                    r["_category_file"] = cat
                    pool[cat].append(r)
        print(f"{cat:22s} raw={n_raw:5d} kept={len(pool[cat]):5d}")

    # ---- natural duplicate groups inside the whole pool (exact = same normalised text, near = Jaccard >= .8)
    all_pool = [r for cat in CATEGORIES for r in pool[cat]]
    by_hash = defaultdict(list)
    for r in all_pool:
        by_hash[hashlib.md5(norm_for_hash(r["text"]).encode()).hexdigest()].append(r)
    natural_groups = [g for g in by_hash.values() if len(g) > 1]
    # near duplicates among same-user reviews (copy-paste across products is the common real case)
    by_user = defaultdict(list)
    for r in all_pool:
        by_user[r["user_id"]].append(r)
    seen_ids = {id(r) for g in natural_groups for r in g}
    for user, revs in by_user.items():
        if len(revs) < 2:
            continue
        for i in range(len(revs)):
            for j in range(i + 1, len(revs)):
                a, b = revs[i], revs[j]
                if id(a) in seen_ids or id(b) in seen_ids:
                    continue
                if jaccard(shingles(a["text"]), shingles(b["text"])) >= 0.8:
                    natural_groups.append([a, b])
                    seen_ids.update({id(a), id(b)})
    n_pool = len(all_pool)
    n_in_groups = sum(len(g) for g in natural_groups)
    print(f"natural duplicate groups found in pool: {len(natural_groups)} "
          f"({n_in_groups} of {n_pool} pooled reviews = {100 * n_in_groups / n_pool:.1f}%)")

    # ---- keep a controlled number of natural groups (size 2-3) per category
    chosen_groups = []
    for cat in CATEGORIES:
        cands = [g for g in natural_groups if 2 <= len(g) <= 3 and all(r["_category_file"] == cat for r in g)]
        cands.sort(key=lambda g: min(int(r["timestamp"]) for r in g))  # deterministic order before shuffling
        rng.shuffle(cands)
        chosen_groups.extend(cands[:NATURAL_GROUPS_PER_CLASS])
    must_keep = {id(r) for g in chosen_groups for r in g}
    excluded = {id(r) for g in natural_groups for r in g} - must_keep  # other duplicates are not sampled

    # ---- per-class sampling with a per-user cap
    selected = []
    for cat in CATEGORIES:
        cands = pool[cat][:]
        rng.shuffle(cands)
        keep = [r for r in cands if id(r) in must_keep]
        per_user = Counter(r["user_id"] for r in keep)
        for r in cands:
            if len(keep) >= DOCS_PER_CLASS:
                break
            if id(r) in must_keep or id(r) in excluded or per_user[r["user_id"]] >= MAX_PER_USER:
                continue
            keep.append(r)
            per_user[r["user_id"]] += 1
        selected.extend(keep)

    docs = []
    for r in selected:
        ts = int(r["timestamp"]) / 1000.0
        docs.append({
            "text": r["text"],
            "title": (r.get("title") or "").strip(),
            "category": CATEGORIES[r["_category_file"]],
            "rating": float(r.get("rating") or 0),
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "helpful_votes": int(r.get("helpful_vote") or 0),
            "verified_purchase": bool(r.get("verified_purchase")),
            "has_images": bool(r.get("images")),
            "asin": r.get("asin", ""),
            "parent_asin": r.get("parent_asin", ""),
            "user_id": r.get("user_id", ""),
            "source": f"Amazon Reviews 2023 / {r['_category_file']}.jsonl",
            "dup_group": "",
            "dup_type": "",
            "is_injected": False,
            "_twin": None,
        })

    # ---- injected near-duplicate variants (documented, 2% of the corpus)
    natural_ids = {id(r) for r in selected if id(r) in must_keep}
    originals = [d for d, r in zip(docs, selected) if id(r) not in natural_ids]
    rng.shuffle(originals)
    kinds = ["casing_markup", "punctuation", "edit"]
    for i, orig in enumerate(originals[:N_INJECTED]):
        variant = dict(orig)
        variant["text"] = make_variant(orig["text"], rng, kinds[i % len(kinds)])
        variant["is_injected"] = True
        variant["source"] = orig["source"] + " (injected near-duplicate variant)"
        variant["_twin"] = orig
        docs.append(variant)

    # ---- shuffle and assign neutral IDs
    rng.shuffle(docs)
    for i, d in enumerate(docs, start=1):
        d["doc_id"] = f"R{i:04d}"

    # ---- duplicate ground truth over the FINAL corpus (union-find over exact / near / injected links)
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pos = {id(d): i for i, d in enumerate(docs)}
    hashes = defaultdict(list)
    for i, d in enumerate(docs):
        hashes[hashlib.md5(norm_for_hash(d["text"]).encode()).hexdigest()].append(i)
    for idx in hashes.values():
        for j in idx[1:]:
            union(idx[0], j)
    sh = [shingles(d["text"]) for d in docs]
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if find(i) != find(j) and jaccard(sh[i], sh[j]) >= 0.8:
                union(i, j)
    for i, d in enumerate(docs):
        if d["_twin"] is not None:
            union(i, pos[id(d["_twin"])])
    groups = defaultdict(list)
    for i in range(len(docs)):
        groups[find(i)].append(i)
    gno = 0
    for root in sorted(groups):
        members = groups[root]
        if len(members) < 2:
            continue
        gno += 1
        exact = len({hashlib.md5(norm_for_hash(docs[m]["text"]).encode()).hexdigest() for m in members}) == 1
        for m in members:
            docs[m]["dup_group"] = f"DUP{gno:03d}"
            docs[m]["dup_type"] = "exact" if exact else "near"

    os.makedirs(DATA_DIR, exist_ok=True)
    meta_fields = ["title", "category", "rating", "date", "helpful_votes", "verified_purchase", "has_images",
                   "asin", "parent_asin", "user_id", "source", "dup_group", "dup_type", "is_injected"]
    csv_path = os.path.join(DATA_DIR, "amazon_reviews_corpus.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "text"] + meta_fields)
        w.writeheader()
        for d in docs:
            w.writerow({k: d[k] for k in ["doc_id", "text"] + meta_fields})
    json_path = os.path.join(DATA_DIR, "amazon_reviews_corpus.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([{"doc_id": d["doc_id"], "text": d["text"], "metadata": {k: d[k] for k in meta_fields}}
                   for d in docs], fh, ensure_ascii=False, indent=1)

    cat_counts = Counter(d["category"] for d in docs)
    n_bytes = sum(len(d["text"].encode("utf-8")) for d in docs)
    print(f"\nwrote {len(docs)} documents ({n_bytes / 1e6:.2f} MB raw text) -> {csv_path}")
    for c, n in sorted(cat_counts.items()):
        print(f"  {c:20s} {n}")
    in_groups = [d for d in docs if d["dup_group"]]
    print(f"  duplicate groups: {len({d['dup_group'] for d in in_groups})} "
          f"({len(in_groups)} docs = {100 * len(in_groups) / len(docs):.1f}% of corpus); "
          f"by type: {dict(Counter(d['dup_type'] for d in in_groups))}; "
          f"injected variants: {sum(d['is_injected'] for d in docs)}")


if __name__ == "__main__":
    sys.exit(main())
