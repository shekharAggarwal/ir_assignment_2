"""Corpus analytics helpers for domain analysts (distinctive vocabulary of document subsets)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def distinctive_terms(docs_a: Sequence[Sequence[str]], docs_b: Sequence[Sequence[str]], n: int = 10,
                      alpha: float = 0.5, min_count: int = 3) -> List[Tuple[str, float, int, int]]:
    """
    Terms over-represented in subset A compared with subset B, ranked by the smoothed log-odds ratio
        log[(c_A(t) + a) / (N_A + a|V|)] - log[(c_B(t) + a) / (N_B + a|V|)]
    Returns (term, log-odds, count in A, count in B); only terms seen >= min_count times in A.
    """
    ca = Counter(t for d in docs_a for t in d)
    cb = Counter(t for d in docs_b for t in d)
    na, nb = sum(ca.values()), sum(cb.values())
    v = len(set(ca) | set(cb)) or 1
    out = []
    for t, c in ca.items():
        if c < min_count:
            continue
        lo = math.log((c + alpha) / (na + alpha * v)) - math.log((cb.get(t, 0) + alpha) / (nb + alpha * v))
        out.append((t, lo, c, cb.get(t, 0)))
    return sorted(out, key=lambda r: (-r[1], r[0]))[:n]


def department_insights(terms: Sequence[Sequence[str]], labels: Sequence[str],
                        ratings: Optional[Sequence[float]] = None, n: int = 8) -> List[Dict[str, object]]:
    """Per category: distinctive vocabulary (vs other categories) and what its 1-2 star reviews talk about
    (vs its own 4-5 star reviews)."""
    rows = []
    for cat in sorted(set(labels)):
        a_docs = [t for t, l in zip(terms, labels) if l == cat]
        b_docs = [t for t, l in zip(terms, labels) if l != cat]
        row: Dict[str, object] = {"category": cat, "documents": len(a_docs),
                                  "distinctive terms": ", ".join(t for t, *_ in distinctive_terms(a_docs, b_docs, n=n))}
        if ratings is not None:
            neg = [t for t, l, r in zip(terms, labels, ratings) if l == cat and r <= 2]
            pos = [t for t, l, r in zip(terms, labels, ratings) if l == cat and r >= 4]
            row["negative reviews"] = len(neg)
            row["share negative"] = len(neg) / max(1, len(a_docs))
            row["what negative reviews talk about"] = ", ".join(
                t for t, *_ in distinctive_terms(neg, pos, n=n, min_count=2))
        rows.append(row)
    return rows


def cluster_rating_summary(cluster_labels: Sequence[int], ratings: Sequence[float], majority: Dict,
                           top_terms: Sequence[Sequence[Tuple[str, float]]]) -> List[Dict[str, object]]:
    """Mean rating and share of 1-2 star reviews per K-Means cluster (themes ranked by dissatisfaction)."""
    rows = []
    for k in sorted(set(cluster_labels)):
        rs = [r for c, r in zip(cluster_labels, ratings) if c == k]
        rows.append({"cluster": f"C{k}", "documents": len(rs), "mean rating": sum(rs) / len(rs) if rs else 0.0,
                     "share 1-2★": sum(1 for r in rs if r <= 2) / len(rs) if rs else 0.0,
                     "majority category": majority[k][0], "top terms": ", ".join(t for t, _ in top_terms[k][:8])})
    return sorted(rows, key=lambda r: -r["share 1-2★"])


def collaborative_feasibility(user_ids: Sequence[str], item_ids: Sequence[str]) -> Dict[str, float]:
    """How much user-item interaction data exists (evidence for choosing content-based recommendation)."""
    pairs = {(u, i) for u, i in zip(user_ids, item_ids) if u and i}
    users = Counter(u for u, _ in pairs)
    items = Counter(i for _, i in pairs)
    n_u, n_i = len(users), len(items)
    # users who share at least one item with another user = the only ones collaborative filtering could link
    item_users: Dict[str, set] = {}
    for u, i in pairs:
        item_users.setdefault(i, set()).add(u)
    linked = set()
    for us in item_users.values():
        if len(us) > 1:
            linked |= us
    return {"reviews": len(user_ids), "distinct users": n_u, "users with >= 2 reviews": sum(1 for c in users.values() if c >= 2),
            "distinct items (parent_asin)": n_i, "items with >= 2 reviewers": sum(1 for c in items.values() if c >= 2),
            "user-item matrix density": len(pairs) / (n_u * n_i) if n_u and n_i else 0.0,
            "users sharing an item with another user": len(linked)}
