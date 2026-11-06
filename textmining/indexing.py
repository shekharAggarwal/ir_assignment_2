"""
Inverted index, TF-IDF weighting with cosine (L2) length normalisation, and cosine similarity.
Everything here is implemented from first principles (no scikit-learn / gensim vectorisers).

Weighting (SMART notation of Manning, Raghavan & Schütze, "Introduction to IR", ch. 6):
    tf weight      raw:  tf          log ("l"):  1 + log10(tf)        binary: 1
    idf weight     idf ("t"): log10(N / df)       smooth: log10((1 + N) / (1 + df)) + 1      none: 1
    w(t, d)      = tf_weight(tf(t, d)) * idf(t)
    normalised   = w(t, d) / sqrt( sum_t' w(t', d)^2 )        ("c" = cosine normalisation)
Because every document vector has unit length, cosine(d1, d2) is simply the dot product of the two vectors.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SparseVector = Dict[str, float]


def tf_weight(tf: float, scheme: str = "log") -> float:
    if tf <= 0:
        return 0.0
    if scheme == "raw":
        return float(tf)
    if scheme == "log":
        return 1.0 + math.log10(tf)
    if scheme == "binary":
        return 1.0
    raise ValueError(f"unknown tf scheme {scheme!r}")


def idf_weight(df: int, n_docs: int, scheme: str = "idf") -> float:
    if df <= 0:
        return 0.0
    if scheme == "idf":
        return math.log10(n_docs / df)
    if scheme == "smooth":
        return math.log10((1 + n_docs) / (1 + df)) + 1.0
    if scheme == "none":
        return 1.0
    raise ValueError(f"unknown idf scheme {scheme!r}")


def vector_norm(vec: SparseVector) -> float:
    # math.fsum is exactly rounded, so the result is identical on every Python version / platform
    return math.sqrt(math.fsum(w * w for w in vec.values()))


def l2_normalize(vec: SparseVector) -> SparseVector:
    """Cosine length normalisation: divide every weight by the Euclidean length of the vector."""
    norm = vector_norm(vec)
    return {t: w / norm for t, w in vec.items()} if norm > 0 else {}


def cosine_similarity(u: SparseVector, v: SparseVector) -> float:
    """cos(u, v) = (u . v) / (|u| |v|) for sparse dict vectors (works for normalised or raw vectors)."""
    if not u or not v:
        return 0.0
    if len(u) > len(v):
        u, v = v, u
    dot = math.fsum(w * v.get(t, 0.0) for t, w in u.items())
    denom = vector_norm(u) * vector_norm(v)
    return dot / denom if denom > 0 else 0.0


class InvertedIndex:
    """
    term -> postings list [(doc_index, term_frequency), ...] kept in increasing doc order.
    Vocabulary can be pruned by document frequency (min_df, max_df) and capped (max_features).
    """

    def __init__(self, docs_terms: Sequence[Sequence[str]], min_df: int = 1, max_df: float = 1.0,
                 max_features: Optional[int] = None):
        self.n_docs = len(docs_terms)
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for doc_idx, terms in enumerate(docs_terms):
            for term, tf in Counter(terms).items():
                postings[term].append((doc_idx, tf))
        self.full_vocabulary_size = len(postings)

        max_df_abs = max_df if isinstance(max_df, int) and max_df > 1 else math.floor(float(max_df) * self.n_docs)
        kept = {t: p for t, p in postings.items() if min_df <= len(p) <= max(max_df_abs, min_df)}
        if max_features and len(kept) > max_features:
            # keep the terms with the highest document frequency (ties: collection frequency, then alphabet)
            ranked = sorted(kept, key=lambda t: (-len(kept[t]), -sum(tf for _, tf in kept[t]), t))
            kept = {t: kept[t] for t in ranked[:max_features]}
        self.postings: Dict[str, List[Tuple[int, int]]] = dict(sorted(kept.items()))
        self.vocabulary: List[str] = list(self.postings)
        self.term_index: Dict[str, int] = {t: i for i, t in enumerate(self.vocabulary)}
        self.df: Dict[str, int] = {t: len(p) for t, p in self.postings.items()}
        self.cf: Dict[str, int] = {t: sum(tf for _, tf in p) for t, p in self.postings.items()}
        self.removed_terms = self.full_vocabulary_size - len(self.vocabulary)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def lookup(self, term: str) -> List[Tuple[int, int]]:
        return self.postings.get(term, [])

    def n_postings(self) -> int:
        return sum(self.df.values())


class TfidfModel:
    """TF-IDF document vectors computed by walking the postings lists of an InvertedIndex."""

    def __init__(self, index: InvertedIndex, tf_scheme: str = "log", idf_scheme: str = "idf",
                 normalize: bool = True):
        self.index = index
        self.tf_scheme, self.idf_scheme, self.normalize = tf_scheme, idf_scheme, normalize
        n = index.n_docs
        self.idf: Dict[str, float] = {t: idf_weight(df, n, idf_scheme) for t, df in index.df.items()}

        # pass 1 - raw tf-idf weights read straight from the postings lists
        vectors: List[SparseVector] = [dict() for _ in range(n)]
        for term, plist in index.postings.items():
            idf = self.idf[term]
            for doc_idx, tf in plist:
                w = tf_weight(tf, tf_scheme) * idf
                if w != 0.0:
                    vectors[doc_idx][term] = w
        # pass 2 - document lengths and cosine normalisation
        self.doc_norms: List[float] = [vector_norm(v) for v in vectors]
        if normalize:
            vectors = [{t: w / nrm for t, w in v.items()} if nrm > 0 else {} for v, nrm in zip(vectors, self.doc_norms)]
        self.doc_vectors: List[SparseVector] = vectors

        # weighted postings (term -> [(doc, normalised weight)]) used for fast cosine scoring of a query
        self.weighted_postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for doc_idx, vec in enumerate(self.doc_vectors):
            for term, w in vec.items():
                self.weighted_postings[term].append((doc_idx, w))

    # ------------------------------------------------------------------ queries / new documents
    def vectorize(self, terms: Iterable[str]) -> SparseVector:
        """Weight a new document or query with the corpus idf (unknown terms are ignored)."""
        counts = Counter(t for t in terms if t in self.idf)
        vec = {t: tf_weight(tf, self.tf_scheme) * self.idf[t] for t, tf in counts.items()}
        vec = {t: w for t, w in vec.items() if w != 0.0}
        return l2_normalize(vec) if self.normalize else vec

    def score(self, query_vec: SparseVector) -> Dict[int, float]:
        """
        Term-at-a-time cosine scoring through the inverted index (IIR fig. 6.14 'CosineScore'):
        for each query term walk its postings list and accumulate w(t,q) * w(t,d) per document.
        """
        scores: Dict[int, float] = defaultdict(float)
        for term, wq in query_vec.items():
            for doc_idx, wd in self.weighted_postings.get(term, ()):
                scores[doc_idx] += wq * wd
        return scores

    def top_terms(self, doc_idx: int, n: int = 10) -> List[Tuple[str, float]]:
        vec = self.doc_vectors[doc_idx]
        return sorted(vec.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def to_dense(self, dtype=np.float64) -> np.ndarray:
        """N x V matrix with the same (normalised) weights, used by K-Means and the 2-D projection."""
        idx = self.index.term_index
        X = np.zeros((self.index.n_docs, len(idx)), dtype=dtype)
        for d, vec in enumerate(self.doc_vectors):
            for term, w in vec.items():
                X[d, idx[term]] = w
        return X


def similarity_matrix(X: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity. Rows of X are unit length, so cos = X . X^T (zero rows stay 0).
    Rounded to 12 decimals so that rankings do not depend on last-bit differences between BLAS libraries."""
    norms = np.sqrt((X * X).sum(axis=1))
    safe = np.where(norms > 0, norms, 1.0)
    Xn = X / safe[:, None]
    return np.round(Xn @ Xn.T, 12)
