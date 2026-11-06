"""
Multinomial Naive Bayes text classifier, stratified train/test split and evaluation metrics - all manual.

Training (IIR ch. 13, "TrainMultinomialNB"):
    prior        P(c)   = N_c / N
    likelihood   P(t|c) = (T_ct + alpha) / (sum_t' T_ct' + alpha * |V|)      (Laplace / add-alpha smoothing)
    where T_ct is the number of occurrences of term t in training documents of class c and V is the
    vocabulary of the TRAINING split only.
Prediction ("ApplyMultinomialNB"), in log space to avoid underflow:
    c_map = argmax_c  log P(c) + sum_{t in d} tf(t, d) * log P(t|c)          (terms unseen in training ignored)
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


# ------------------------------------------------------------------------------------------------ split
def seeded_shuffle(items: List, rng: random.Random) -> List:
    """Fisher-Yates shuffle driven only by rng.random() (identical across Python versions)."""
    items = list(items)
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]
    return items


def stratified_split(labels: Sequence[str], test_size: float = 0.25, seed: int = 537,
                     groups: Optional[Sequence[Optional[str]]] = None) -> Tuple[List[int], List[int]]:
    """
    Per-class random split. If `groups` is given (e.g. duplicate-group ids), all members of a group are
    kept on the same side of the split, so a document and its near-duplicate can never be split between
    train and test (prevents leakage).
    Returns (train_indices, test_indices), both sorted.
    """
    rng = random.Random(seed)
    units_by_class: Dict[str, List[List[int]]] = defaultdict(list)
    seen_group: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        g = groups[i] if groups is not None else None
        if g:
            if g in seen_group:
                seen_group[g].append(i)
                continue
            unit = [i]
            seen_group[g] = unit
            units_by_class[lab].append(unit)
        else:
            units_by_class[lab].append([i])
    train, test = [], []
    for lab in sorted(units_by_class):
        units = seeded_shuffle(units_by_class[lab], rng)
        n_docs = sum(len(u) for u in units)
        target = round(test_size * n_docs)
        taken = 0
        for unit in units:
            if taken < target:
                test.extend(unit)
                taken += len(unit)
            else:
                train.extend(unit)
    return sorted(train), sorted(test)


# ------------------------------------------------------------------------------------------------ model
class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: List[str] = []
        self.log_prior_: Dict[str, float] = {}
        self.log_likelihood_: Dict[str, Dict[str, float]] = {}
        self.vocabulary_: set = set()
        self.class_term_counts_: Dict[str, Counter] = {}
        self.class_doc_counts_: Dict[str, int] = {}

    def fit(self, docs_terms: Sequence[Sequence[str]], labels: Sequence[str]) -> "MultinomialNaiveBayes":
        n_docs = len(docs_terms)
        self.classes_ = sorted(set(labels))
        self.class_doc_counts_ = Counter(labels)
        self.class_term_counts_ = {c: Counter() for c in self.classes_}
        for terms, lab in zip(docs_terms, labels):
            self.class_term_counts_[lab].update(terms)          # T_ct
        self.vocabulary_ = set()
        for counts in self.class_term_counts_.values():
            self.vocabulary_.update(counts)
        v = len(self.vocabulary_)
        for c in self.classes_:
            self.log_prior_[c] = math.log(self.class_doc_counts_[c] / n_docs)
            counts = self.class_term_counts_[c]
            denom = sum(counts.values()) + self.alpha * v
            self.log_likelihood_[c] = {t: math.log((counts.get(t, 0) + self.alpha) / denom) for t in self.vocabulary_}
        return self

    def joint_log_scores(self, terms: Sequence[str]) -> Dict[str, float]:
        """log P(c) + sum tf * log P(t|c) for every class (unseen terms are skipped)."""
        tf = Counter(t for t in terms if t in self.vocabulary_)
        return {c: self.log_prior_[c] + sum(n * self.log_likelihood_[c][t] for t, n in tf.items())
                for c in self.classes_}

    def predict_proba_one(self, terms: Sequence[str]) -> Dict[str, float]:
        scores = self.joint_log_scores(terms)
        m = max(scores.values())                                 # log-sum-exp for a stable softmax
        exp = {c: math.exp(s - m) for c, s in scores.items()}
        z = sum(exp.values())
        return {c: e / z for c, e in exp.items()}

    def predict(self, docs_terms: Sequence[Sequence[str]]) -> List[str]:
        out = []
        for terms in docs_terms:
            scores = self.joint_log_scores(terms)
            out.append(max(self.classes_, key=lambda c: (scores[c], c)))
        return out

    def top_terms(self, cls: str, n: int = 15, mode: str = "log-odds") -> List[Tuple[str, float]]:
        """
        'likelihood': terms with the largest P(t|c).
        'log-odds'  : log P(t|c) - mean over other classes of log P(t|c') -> terms that distinguish c.
        Only terms seen at least 3 times in class c are considered for log-odds.
        """
        ll = self.log_likelihood_[cls]
        if mode == "likelihood":
            items = [(t, math.exp(s)) for t, s in ll.items()]
        else:
            others = [c for c in self.classes_ if c != cls]
            counts = self.class_term_counts_[cls]
            items = [(t, ll[t] - sum(self.log_likelihood_[o][t] for o in others) / max(1, len(others)))
                     for t in ll if counts.get(t, 0) >= 3]
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))[:n]


# ------------------------------------------------------------------------------------------------ metrics
def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[List[int]]:
    """rows = actual class, columns = predicted class."""
    pos = {c: i for i, c in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[pos[t]][pos[p]] += 1
    return m


def classification_report(y_true: Sequence[str], y_pred: Sequence[str],
                          labels: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Accuracy, per-class precision / recall / F1 / support, macro and weighted averages."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    per_class = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": c, "precision": precision, "recall": recall, "f1": f1,
                          "support": tp + fn, "tp": tp, "fp": fp, "fn": fn})
    k = len(labels)
    total_support = sum(r["support"] for r in per_class) or 1
    macro = {m: sum(r[m] for r in per_class) / k for m in ("precision", "recall", "f1")}
    weighted = {m: sum(r[m] * r["support"] for r in per_class) / total_support for m in ("precision", "recall", "f1")}
    return {"accuracy": correct / n if n else 0.0, "n_test": n, "n_correct": correct, "per_class": per_class,
            "macro": macro, "weighted": weighted, "labels": labels}
