"""
Content-based Top-K document recommendation.

A selected document (or free text typed by the analyst) is turned into a TF-IDF query vector with the
same preprocessing and the corpus idf. Candidate documents are scored with cosine similarity, accumulated
term-at-a-time through the inverted index (only documents sharing at least one term are touched).
Optional re-ranking rules:
    * exclude the query document itself
    * collapse near-duplicates: skip a candidate whose cosine with the query or with an already selected
      recommendation is >= dup_threshold (so the list is not filled with copies of the same review)
    * restrict to the same category / a minimum score
Evaluation without user feedback: Precision@K = share of the K recommendations that belong to the same
product category as the query, averaged over every document used as a query (leave-one-out).
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indexing import SparseVector, TfidfModel, cosine_similarity


class ContentRecommender:
    def __init__(self, model: TfidfModel, labels: Optional[Sequence[str]] = None):
        self.model = model
        self.labels = list(labels) if labels is not None else None

    def explain(self, query_vec: SparseVector, doc_idx: int, n: int = 6) -> List[Tuple[str, float]]:
        """Shared terms that contribute most to the cosine score: w(t,q) * w(t,d)."""
        dvec = self.model.doc_vectors[doc_idx]
        contrib = [(t, wq * dvec[t]) for t, wq in query_vec.items() if t in dvec]
        return sorted(contrib, key=lambda kv: (-kv[1], kv[0]))[:n]

    def recommend(self, query_vec: SparseVector, k: int = 5, exclude: Optional[int] = None,
                  collapse_duplicates: bool = True, dup_threshold: float = 0.80,
                  same_label_as: Optional[str] = None, min_score: float = 0.0) -> List[Dict[str, object]]:
        scores = {d: round(v, 12) for d, v in self.model.score(query_vec).items()}
        if exclude is not None:
            scores.pop(exclude, None)
        # rank candidates (heap keeps it O(n log k) even for large corpora)
        candidates = heapq.nlargest(max(k * 10, 50), scores.items(), key=lambda kv: (kv[1], -kv[0]))
        results: List[Dict[str, object]] = []
        skipped = 0
        for doc_idx, score in candidates:
            if score < min_score:
                break
            if same_label_as is not None and self.labels is not None and self.labels[doc_idx] != same_label_as:
                continue
            if collapse_duplicates:
                if score >= dup_threshold:
                    skipped += 1
                    continue
                dvec = self.model.doc_vectors[doc_idx]
                if any(cosine_similarity(dvec, self.model.doc_vectors[r["doc_idx"]]) >= dup_threshold for r in results):
                    skipped += 1
                    continue
            results.append({"rank": len(results) + 1, "doc_idx": doc_idx, "score": float(score),
                            "shared_terms": self.explain(query_vec, doc_idx)})
            if len(results) == k:
                break
        for r in results:
            r["skipped_duplicates"] = skipped
        return results

    def recommend_for_doc(self, doc_idx: int, k: int = 5, **kw) -> List[Dict[str, object]]:
        return self.recommend(self.model.doc_vectors[doc_idx], k=k, exclude=doc_idx, **kw)


def evaluate_precision_at_k(sim: np.ndarray, labels: Sequence[str], k: int = 5,
                            collapse_duplicates: bool = False, dup_threshold: float = 0.80) -> Dict[str, float]:
    """
    Leave-one-out category agreement of the Top-K lists for every document, computed from the full cosine
    matrix (identical scores to the inverted-index scorer, but vectorised for speed).
    If collapse_duplicates is set, candidates with cosine >= dup_threshold to the query are skipped.
    Also reports the share of recommended items that are near-copies of the query (duplicate rate).
    """
    labels = np.asarray(labels)
    n = sim.shape[0]
    hits, dup_items, mrr = 0, 0, 0.0
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    for q in range(n):
        row = S[q]
        order = np.argsort(-row, kind="stable")
        if collapse_duplicates:
            # same rule as ContentRecommender.recommend: skip copies of the query and of earlier picks
            picked: List[int] = []
            for c in order:
                if row[c] >= dup_threshold or any(sim[c, r] >= dup_threshold for r in picked):
                    continue
                picked.append(int(c))
                if len(picked) == k:
                    break
            top = np.array(picked, dtype=int)
        else:
            top = order[:k]
        same = labels[top] == labels[q]
        hits += int(same.sum())
        dup_items += int((row[top] >= dup_threshold).sum())
        first = np.flatnonzero(same)
        mrr += 1.0 / (first[0] + 1) if first.size else 0.0
    return {"precision_at_k": hits / (n * k), "mrr": mrr / n, "duplicate_rate": dup_items / (n * k), "k": k}


def precision_at_k_by_label(sim: np.ndarray, labels: Sequence[str], k: int = 5, collapse_duplicates: bool = True,
                            dup_threshold: float = 0.80) -> List[Dict[str, object]]:
    """P@K averaged separately for the queries of every category (same rules as evaluate_precision_at_k)."""
    lab = np.asarray(labels)
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    per: Dict[str, List[float]] = {}
    for q in range(len(lab)):
        row = S[q]
        picked: List[int] = []
        for j in np.argsort(-row, kind="stable"):
            if collapse_duplicates and (row[j] >= dup_threshold or any(sim[j, p] >= dup_threshold for p in picked)):
                continue
            picked.append(int(j))
            if len(picked) == k:
                break
        per.setdefault(str(lab[q]), []).append(float(np.mean(lab[picked] == lab[q])) if picked else 0.0)
    return [{"category": c, "queries": len(v), f"P@{k}": float(np.mean(v))} for c, v in sorted(per.items())]
