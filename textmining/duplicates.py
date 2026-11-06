"""
Duplicate and near-duplicate detection.

* exact duplicates : identical text after normalisation (lower-case, markup removed, punctuation removed),
                     found with a hash table of the normalised text.
* near-duplicates  : pairs whose TF-IDF cosine similarity is >= threshold (default 0.80); the word
                     3-shingle Jaccard similarity of each pair is reported as a second opinion.
Pairs are merged into duplicate groups with union-find. When a ground-truth column is available
(dup_group in the bundled dataset) the detector is scored with pair-level precision / recall / F1.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z0-9]+")


def normalized_text(text: str) -> str:
    return " ".join(_WORD.findall(_TAG.sub(" ", text.lower())))


def text_hash(text: str) -> str:
    return hashlib.md5(normalized_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 3) -> Set[str]:
    toks = normalized_text(text).split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> List[List[int]]:
        g: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return sorted((m for m in g.values() if len(m) > 1), key=lambda m: m[0])


@dataclass
class DuplicateReport:
    n_docs: int
    threshold: float
    exact_groups: List[List[int]]
    near_pairs: List[Tuple[int, int, float, float]]        # (i, j, cosine, jaccard) with i < j
    groups: List[List[int]]                                # merged exact + near groups
    group_of: Dict[int, int] = field(default_factory=dict)  # doc -> group number (1-based)

    @property
    def n_in_groups(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def n_removable(self) -> int:
        """Documents that would be dropped when keeping one representative per group."""
        return sum(len(g) - 1 for g in self.groups)

    def keep_mask(self, policy: str) -> List[bool]:
        """policy: keep_all | remove_exact | remove_near (keeps the first document of each group)."""
        keep = [True] * self.n_docs
        if policy == "keep_all":
            return keep
        groups = self.exact_groups if policy == "remove_exact" else self.groups
        for g in groups:
            for i in g[1:]:
                keep[i] = False
        return keep

    def group_labels(self) -> List[Optional[str]]:
        return [f"G{self.group_of[i]:03d}" if i in self.group_of else None for i in range(self.n_docs)]


def detect_duplicates(texts: Sequence[str], sim: np.ndarray, threshold: float = 0.80) -> DuplicateReport:
    n = len(texts)
    # exact duplicates through a hash table
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        buckets[text_hash(t)].append(i)
    exact_groups = sorted((b for b in buckets.values() if len(b) > 1), key=lambda b: b[0])
    # near duplicates from the cosine similarity matrix (upper triangle only)
    iu, ju = np.where(np.triu(sim, k=1) >= threshold)
    near_pairs = []
    sh_cache: Dict[int, Set[str]] = {}
    for i, j in zip(iu.tolist(), ju.tolist()):
        if i not in sh_cache:
            sh_cache[i] = shingles(texts[i])
        if j not in sh_cache:
            sh_cache[j] = shingles(texts[j])
        near_pairs.append((i, j, float(sim[i, j]), jaccard(sh_cache[i], sh_cache[j])))
    uf = UnionFind(n)
    for g in exact_groups:
        for i in g[1:]:
            uf.union(g[0], i)
    for i, j, _, _ in near_pairs:
        uf.union(i, j)
    groups = uf.groups()
    group_of = {i: gi for gi, g in enumerate(groups, start=1) for i in g}
    return DuplicateReport(n, threshold, exact_groups, near_pairs, groups, group_of)


def _pairs_from_groups(groups: Sequence[Sequence[int]]) -> Set[Tuple[int, int]]:
    out = set()
    for g in groups:
        g = sorted(g)
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                out.add((g[a], g[b]))
    return out


def evaluate_detection(pred_groups: Sequence[Sequence[int]], true_group_ids: Sequence[Optional[str]]) -> Dict[str, float]:
    """Pair-level precision / recall / F1 of detected duplicate pairs against a ground-truth column."""
    truth: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(true_group_ids):
        if g:
            truth[str(g)].append(i)
    true_pairs = _pairs_from_groups(list(truth.values()))
    pred_pairs = _pairs_from_groups(pred_groups)
    tp = len(true_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(true_pairs) if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true pairs": len(true_pairs), "detected pairs": len(pred_pairs), "correct pairs": tp,
            "precision": precision, "recall": recall, "f1": f1}


def threshold_sweep(texts: Sequence[str], sim: np.ndarray, true_group_ids: Sequence[Optional[str]],
                    thresholds: Sequence[float]) -> List[Dict[str, float]]:
    rows = []
    for th in thresholds:
        rep = detect_duplicates(texts, sim, th)
        ev = evaluate_detection(rep.groups, true_group_ids)
        rows.append({"threshold": th, **ev, "groups": len(rep.groups)})
    return rows
