"""
External clustering evaluation implemented manually (IIR ch. 16.3).

Purity      = (1 / N) * sum_k max_j |w_k ∩ c_j|
              (each cluster is credited with its most frequent true class)
Rand Index  = (TP + TN) / (TP + FP + FN + TN), computed over all N(N-1)/2 document pairs:
              TP: same class & same cluster         FP: different class, same cluster
              FN: same class, different cluster     TN: different class & different cluster
Two implementations of the Rand Index are provided: a literal pair-by-pair count (definition) and an
equivalent O(N) version using combinations of the contingency-table counts; the tests check they agree.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple


def comb2(n: int) -> int:
    """Number of unordered pairs: n choose 2."""
    return n * (n - 1) // 2


def contingency_table(clusters: Sequence[Hashable], labels: Sequence[Hashable]) -> Dict[Hashable, Counter]:
    """cluster id -> Counter(true label -> number of documents)."""
    table: Dict[Hashable, Counter] = defaultdict(Counter)
    for k, c in zip(clusters, labels):
        table[k][c] += 1
    return dict(table)


def purity(clusters: Sequence[Hashable], labels: Sequence[Hashable]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    table = contingency_table(clusters, labels)
    return sum(max(counts.values()) for counts in table.values()) / n


def majority_labels(clusters: Sequence[Hashable], labels: Sequence[Hashable]) -> Dict[Hashable, Tuple[Hashable, int, int]]:
    """cluster -> (majority true label, documents of that label, cluster size)."""
    out = {}
    for k, counts in contingency_table(clusters, labels).items():
        lab, cnt = max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
        out[k] = (lab, cnt, sum(counts.values()))
    return out


def pair_counts_bruteforce(clusters: Sequence[Hashable], labels: Sequence[Hashable]) -> Dict[str, int]:
    """Literal definition: visit every pair of documents once (O(N^2))."""
    tp = fp = fn = tn = 0
    n = len(labels)
    for i in range(n):
        ki, ci = clusters[i], labels[i]
        for j in range(i + 1, n):
            same_cluster = ki == clusters[j]
            same_class = ci == labels[j]
            if same_cluster and same_class:
                tp += 1
            elif same_cluster:
                fp += 1
            elif same_class:
                fn += 1
            else:
                tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def pair_counts(clusters: Sequence[Hashable], labels: Sequence[Hashable]) -> Dict[str, int]:
    """Same counts from the contingency table (O(N)):
    TP+FP = sum_k C(n_k, 2);  TP = sum_kj C(n_kj, 2);  TP+FN = sum_j C(n_j, 2);  TN = C(N, 2) - TP - FP - FN."""
    n = len(labels)
    table = contingency_table(clusters, labels)
    tp = sum(comb2(v) for counts in table.values() for v in counts.values())
    same_cluster = sum(comb2(sum(counts.values())) for counts in table.values())
    same_class = sum(comb2(v) for v in Counter(labels).values())
    fp = same_cluster - tp
    fn = same_class - tp
    tn = comb2(n) - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def rand_index(clusters: Sequence[Hashable], labels: Sequence[Hashable], method: str = "contingency") -> float:
    counts = pair_counts_bruteforce(clusters, labels) if method == "pairs" else pair_counts(clusters, labels)
    total = sum(counts.values())
    return (counts["TP"] + counts["TN"]) / total if total else 0.0


def chance_baseline(clusters: Sequence[Hashable], labels: Sequence[Hashable], rounds: int = 20,
                    seed: int = 537) -> Dict[str, float]:
    """
    Purity and Rand Index of RANDOM clusterings that have exactly the same cluster sizes
    (the cluster ids are shuffled across documents). Tells how much of a score is due to chance:
    with 6 balanced classes, a random 6-way split already reaches a Rand Index of about 0.72,
    because most document pairs are 'different class & different cluster' (TN).
    """
    from .classification import seeded_shuffle

    rng = random.Random(seed)
    pur, ri = [], []
    for _ in range(rounds):
        shuffled = seeded_shuffle(list(clusters), rng)
        pur.append(purity(shuffled, labels))
        ri.append(rand_index(shuffled, labels))
    return {"purity": sum(pur) / rounds, "rand_index": sum(ri) / rounds, "rounds": rounds}
