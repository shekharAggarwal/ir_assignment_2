"""
K-Means clustering implemented manually (IIR ch. 16, fig. 16.5) on the L2-normalised TF-IDF vectors.

    1. initialisation  : choose K seed documents ("random" or "k-means++": each new seed is drawn with
                         probability proportional to its squared distance from the nearest existing seed)
    2. assignment step : every document goes to the centroid with the smallest squared Euclidean distance
                         |x - mu|^2 = |x|^2 + |mu|^2 - 2 x.mu
    3. update step     : every centroid becomes the mean vector of the documents assigned to it
    4. stopping        : stop when no document changes cluster, or the largest centroid movement < tol,
                         or max_iter is reached. Objective = residual sum of squares
                         RSS = sum_k sum_{x in w_k} |x - mu_k|^2   (never increases between iterations)
Several restarts (n_init) are run with different seeds; the run with the lowest RSS is kept.

metric="cosine" gives spherical K-Means: centroids are re-normalised to unit length after every update.
Because documents and centroids then all have length 1, |x - mu|^2 = 2 - 2 cos(x, mu), so the same
assignment step picks the centroid with the highest cosine similarity.

numpy is used only for vector arithmetic; no library clustering routine is called.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class KMeansRun:
    seed: int
    init_indices: List[int]
    labels: np.ndarray
    centroids: np.ndarray
    rss: float
    n_iter: int
    stop_reason: str
    history: List[Dict[str, float]] = field(default_factory=list)


def _squared_distances(X: np.ndarray, C: np.ndarray, x_sq: np.ndarray) -> np.ndarray:
    """N x K matrix of |x - c|^2, computed as |x|^2 + |c|^2 - 2 x.c (clipped at 0 for rounding)."""
    c_sq = (C * C).sum(axis=1)
    d = x_sq[:, None] + c_sq[None, :] - 2.0 * (X @ C.T)
    return np.maximum(d, 0.0)


TIE_TOL = 1e-9


def _nearest(dist: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid per row. Distances within TIE_TOL of the minimum count as ties and the
    lowest cluster index wins - e.g. a document sharing no term with two unit-length seeds is at distance
    exactly 2 from both; without this rule last-bit rounding noise (different BLAS / Python versions)
    would decide, and K-Means would not be reproducible across machines."""
    mins = dist.min(axis=1, keepdims=True)
    return (dist <= mins + TIE_TOL).argmax(axis=1)


def _pick_weighted(weights: np.ndarray, rng: random.Random) -> int:
    """Draw an index with probability proportional to weights (manual roulette wheel)."""
    total = float(weights.sum())
    if total <= 0:
        return int(rng.random() * len(weights))
    r = rng.random() * total
    cum = np.cumsum(weights)
    return int(min(np.searchsorted(cum, r, side="right"), len(weights) - 1))


def initial_seeds(X: np.ndarray, k: int, method: str, rng: random.Random) -> List[int]:
    n = X.shape[0]
    if method == "random":
        chosen: List[int] = []
        while len(chosen) < k:
            i = int(rng.random() * n)
            if i not in chosen:
                chosen.append(i)
        return chosen
    # k-means++
    x_sq = (X * X).sum(axis=1)
    chosen = [int(rng.random() * n)]
    closest = _squared_distances(X, X[chosen], x_sq).min(axis=1)
    while len(chosen) < k:
        weights = np.round(closest, 12)            # platform-independent sampling weights
        weights[chosen] = 0.0
        nxt = _pick_weighted(weights, rng)
        chosen.append(nxt)
        closest = np.minimum(closest, _squared_distances(X, X[[nxt]], x_sq)[:, 0])
    return chosen


def _unit_rows(C: np.ndarray) -> np.ndarray:
    norms = np.sqrt((C * C).sum(axis=1))
    return C / np.where(norms > 0, norms, 1.0)[:, None]


def kmeans_single(X: np.ndarray, k: int, init: str = "k-means++", max_iter: int = 100, tol: float = 1e-4,
                  seed: int = 537, metric: str = "euclidean") -> KMeansRun:
    rng = random.Random(seed)
    n = X.shape[0]
    x_sq = (X * X).sum(axis=1)
    seeds = initial_seeds(X, k, init, rng)
    centroids = X[seeds].copy()
    if metric == "cosine":
        centroids = _unit_rows(centroids)
    labels = np.full(n, -1, dtype=int)
    history: List[Dict[str, float]] = []
    stop_reason = f"reached max_iter = {max_iter}"
    it = 0
    for it in range(1, max_iter + 1):
        # ---- assignment step
        dist = _squared_distances(X, centroids, x_sq)
        new_labels = _nearest(dist)
        changed = int((new_labels != labels).sum())
        labels = new_labels
        # ---- update step (empty cluster -> re-seed with the document farthest from its centroid)
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            members = labels == j
            if members.any():
                new_centroids[j] = X[members].mean(axis=0)
            else:
                far = int(dist[np.arange(n), labels].argmax())
                new_centroids[j] = X[far]
                labels[far] = j
        if metric == "cosine":
            new_centroids = _unit_rows(new_centroids)
        shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
        centroids = new_centroids
        rss = float(_squared_distances(X, centroids, x_sq)[np.arange(n), labels].sum())
        sizes = np.bincount(labels, minlength=k)
        history.append({"iteration": it, "documents reassigned": changed, "RSS": rss,
                        "max centroid shift": shift, "cluster sizes": ", ".join(str(int(s)) for s in sizes)})
        if it > 1 and changed == 0:
            stop_reason = f"converged: no document changed cluster in iteration {it}"
            break
        if shift < tol:
            stop_reason = f"converged: largest centroid shift {shift:.2e} < tol {tol:g} in iteration {it}"
            break
    rss = float(_squared_distances(X, centroids, x_sq)[np.arange(n), labels].sum())
    return KMeansRun(seed=seed, init_indices=seeds, labels=labels.copy(), centroids=centroids, rss=rss,
                     n_iter=it, stop_reason=stop_reason, history=history)


@dataclass
class KMeansResult:
    best: KMeansRun
    runs: List[KMeansRun]

    @property
    def labels(self) -> np.ndarray:
        return self.best.labels

    @property
    def centroids(self) -> np.ndarray:
        return self.best.centroids


def kmeans(X: np.ndarray, k: int, init: str = "k-means++", n_init: int = 10, max_iter: int = 100,
           tol: float = 1e-4, seed: int = 537, metric: str = "euclidean") -> KMeansResult:
    """Run K-Means n_init times (seeds seed, seed+1, ...) and keep the run with the lowest RSS."""
    if k < 1 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of documents")
    runs = [kmeans_single(X, k, init, max_iter, tol, seed + r, metric) for r in range(max(1, n_init))]
    best = min(runs, key=lambda r: (round(r.rss, 9), r.seed))   # rounded: ties do not depend on float noise
    return KMeansResult(best=best, runs=runs)


def cluster_top_terms(centroids: np.ndarray, vocabulary: Sequence[str], n: int = 10) -> List[List[Tuple[str, float]]]:
    """Highest-weighted terms of each centroid - a readable description of every cluster."""
    out = []
    for c in centroids:
        idx = np.argsort(-c)[:n]
        out.append([(vocabulary[i], float(c[i])) for i in idx if c[i] > 0])
    return out


def pca_2d(X: np.ndarray, extra: Optional[np.ndarray] = None, basis: str = "documents"
           ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float]]:
    """
    Linear 2-D projection used ONLY to draw the cluster scatter plot (the clustering itself uses all terms).
    basis="documents": principal components of the document vectors, computed through the N x N Gram matrix
                       of the centred data (cheaper than an SVD of the N x V matrix when V >> N).
    basis="centroids": principal components of the K cluster centroids; documents are projected onto that
                       plane, which separates the clusters more clearly (sparse text spreads its variance over
                       thousands of directions, so ordinary PCA keeps < 2% of it in two dimensions).
    Returns (document coordinates, coordinates of `extra` rows such as centroids, variance share per axis).
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    if basis == "centroids" and extra is not None and len(extra) >= 3:
        Cc = extra - extra.mean(axis=0)
        _, _, vt = np.linalg.svd(Cc, full_matrices=False)
        axes = vt[:2].T                                       # V x 2 orthonormal directions
    else:
        gram = Xc @ Xc.T
        eigval, eigvec = np.linalg.eigh(gram)                 # ascending order
        order = np.argsort(eigval)[::-1][:2]
        lam = np.maximum(eigval[order], 1e-12)
        axes = (Xc.T @ eigvec[:, order]) / np.sqrt(lam)       # principal directions (V x 2)
    # deterministic sign so the picture does not flip between runs
    for j in range(axes.shape[1]):
        if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
            axes[:, j] *= -1
    coords = Xc @ axes
    total_var = float((Xc * Xc).sum()) or 1.0
    ratios = tuple(float((coords[:, j] ** 2).sum() / total_var) for j in range(2))
    extra_coords = (extra - mean) @ axes if extra is not None else None
    return coords, extra_coords, (ratios[0], ratios[1])
