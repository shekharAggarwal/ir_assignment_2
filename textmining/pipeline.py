"""
End-to-end pipeline shared by the Streamlit app and the command-line experiment runner, so that the
numbers shown in the UI and in the report come from exactly the same code.

    corpus --preprocess--> terms --inverted index + TF-IDF--> vectors --cosine--> duplicates
           --dedup policy--> analysis corpus --> Naive Bayes | K-Means (+Purity, Rand Index) | Top-K recommender
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .classification import MultinomialNaiveBayes, classification_report, confusion_matrix, stratified_split
from .clustering import KMeansResult, cluster_top_terms, kmeans, pca_2d
from .duplicates import DuplicateReport, detect_duplicates
from .evaluation import majority_labels, pair_counts, purity
from .indexing import InvertedIndex, TfidfModel, similarity_matrix
from .ingestion import Corpus
from .preprocessing import PreprocessConfig, Preprocessor
from .recommendation import ContentRecommender, evaluate_precision_at_k


@dataclass(frozen=True)
class FeatureConfig:
    min_df: int = 2                 # drop terms that occur in fewer than 2 documents
    max_df: float = 0.95            # drop terms that occur in more than 95% of documents
    max_features: int = 5000        # keep the 5,000 highest-df terms (bounds the dense K-Means matrix)
    tf_scheme: str = "log"          # 1 + log10(tf)
    idf_scheme: str = "idf"         # log10(N / df)


@dataclass(frozen=True)
class DedupConfig:
    policy: str = "remove_near"     # keep_all | remove_exact | remove_near
    threshold: float = 0.80         # cosine similarity that defines a near-duplicate (see threshold sweep)


@dataclass(frozen=True)
class ModelConfig:
    test_size: float = 0.25
    seed: int = 537
    alpha: float = 1.0              # Laplace smoothing
    group_split: bool = True        # keep duplicate groups on one side of the split
    k: int = 0                      # 0 -> number of categories
    init: str = "k-means++"
    metric: str = "euclidean"       # euclidean (standard K-Means) | cosine (spherical K-Means)
    n_init: int = 10
    max_iter: int = 100
    tol: float = 1e-4
    top_k: int = 5
    collapse_duplicates: bool = True


@dataclass
class FeatureSet:
    terms: List[List[str]]
    index: InvertedIndex
    model: TfidfModel
    X: np.ndarray
    sim: np.ndarray


@dataclass
class PipelineResult:
    full_corpus: Corpus
    corpus: Corpus                   # analysis corpus after the duplicate policy
    kept_indices: List[int]          # positions of `corpus` documents inside `full_corpus`
    features: FeatureSet             # features of the analysis corpus
    full_features: FeatureSet        # features of the full corpus (used for duplicate detection)
    duplicates: DuplicateReport      # detected on the full corpus
    prep: PreprocessConfig
    feat: FeatureConfig
    dedup: DedupConfig
    backend: str
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def group_ids(self) -> List[Optional[str]]:
        """Detected duplicate group of every analysis document (None if it has no duplicate)."""
        full = self.duplicates.group_labels()
        return [full[i] for i in self.kept_indices]


def build_features(terms: List[List[str]], feat: FeatureConfig) -> FeatureSet:
    index = InvertedIndex(terms, min_df=feat.min_df, max_df=feat.max_df, max_features=feat.max_features)
    model = TfidfModel(index, tf_scheme=feat.tf_scheme, idf_scheme=feat.idf_scheme, normalize=True)
    X = model.to_dense()
    return FeatureSet(terms=terms, index=index, model=model, X=X, sim=similarity_matrix(X))


def run_pipeline(full: Corpus, prep: PreprocessConfig, feat: FeatureConfig, dedup: DedupConfig,
                 keep_override: Optional[List[int]] = None) -> PipelineResult:
    timings: Dict[str, float] = {}
    t = time.perf_counter()
    pre = Preprocessor(prep)
    terms_full = pre.process_corpus(full.texts)
    timings["preprocessing"] = time.perf_counter() - t

    t = time.perf_counter()
    full_fs = build_features(terms_full, feat)
    timings["inverted index + TF-IDF"] = time.perf_counter() - t

    t = time.perf_counter()
    dup = detect_duplicates(full.texts, full_fs.sim, dedup.threshold)
    timings["duplicate detection"] = time.perf_counter() - t

    if keep_override is not None:
        kept = list(keep_override)
    else:
        mask = dup.keep_mask(dedup.policy)
        kept = [i for i, keep in enumerate(mask) if keep]
    if len(kept) == len(full):
        corpus, fs = full, full_fs
    else:
        t = time.perf_counter()
        corpus = full.subset(kept)
        fs = build_features([terms_full[i] for i in kept], feat)
        timings["re-index after de-duplication"] = time.perf_counter() - t
    return PipelineResult(full, corpus, kept, fs, full_fs, dup, prep, feat, dedup, pre.backend, timings)


# ------------------------------------------------------------------------------------------------ analytics
@dataclass
class ClassificationResult:
    train_idx: List[int]
    test_idx: List[int]
    y_true: List[str]
    y_pred: List[str]
    confidence: List[float]
    report: Dict[str, object]
    matrix: List[List[int]]
    labels: List[str]
    model: MultinomialNaiveBayes
    seconds: float


def run_classification(pr: PipelineResult, mc: ModelConfig) -> ClassificationResult:
    t = time.perf_counter()
    labels = pr.corpus.labels
    if not labels:
        raise ValueError("The corpus has no category label field - classification needs labels.")
    groups = pr.group_ids if mc.group_split else None
    train_idx, test_idx = stratified_split(labels, mc.test_size, mc.seed, groups)
    terms = pr.features.terms
    nb = MultinomialNaiveBayes(alpha=mc.alpha).fit([terms[i] for i in train_idx], [labels[i] for i in train_idx])
    y_true = [labels[i] for i in test_idx]
    y_pred, conf = [], []
    for i in test_idx:
        proba = nb.predict_proba_one(terms[i])
        best = max(nb.classes_, key=lambda c: (proba[c], c))
        y_pred.append(best)
        conf.append(proba[best])
    class_labels = sorted(set(labels))
    report = classification_report(y_true, y_pred, class_labels)
    matrix = confusion_matrix(y_true, y_pred, class_labels)
    return ClassificationResult(train_idx, test_idx, y_true, y_pred, conf, report, matrix, class_labels, nb,
                                time.perf_counter() - t)


@dataclass
class ClusteringResult:
    result: KMeansResult
    k: int
    labels_pred: List[int]
    purity: float
    pair_counts: Dict[str, int]
    rand_index: float
    coords: np.ndarray
    centroid_coords: np.ndarray
    explained: tuple
    top_terms: list
    majority: dict
    seconds: float


def run_clustering(pr: PipelineResult, mc: ModelConfig, with_projection: bool = True,
                   projection: str = "documents") -> ClusteringResult:
    t = time.perf_counter()
    labels = pr.corpus.labels or ["?"] * len(pr.corpus)
    k = max(1, min(mc.k or len(set(labels)), len(pr.corpus)))
    res = kmeans(pr.features.X, k, init=mc.init, n_init=mc.n_init, max_iter=mc.max_iter, tol=mc.tol, seed=mc.seed,
                 metric=mc.metric)
    pred = [int(c) for c in res.labels]
    pc = pair_counts(pred, labels)
    ri = (pc["TP"] + pc["TN"]) / max(1, sum(pc.values()))
    if with_projection:
        coords, cent, ratios = pca_2d(pr.features.X, res.centroids, basis=projection)
    else:
        coords, cent, ratios = np.zeros((len(pred), 2)), np.zeros((k, 2)), (0.0, 0.0)
    return ClusteringResult(res, k, pred, purity(pred, labels), pc, ri, coords, cent, ratios,
                            cluster_top_terms(res.centroids, pr.features.index.vocabulary, 12),
                            majority_labels(pred, labels), time.perf_counter() - t)


def recommender_for(pr: PipelineResult) -> ContentRecommender:
    return ContentRecommender(pr.features.model, pr.corpus.labels)


def run_recommendation_eval(pr: PipelineResult, mc: ModelConfig) -> Dict[str, float]:
    labels = pr.corpus.labels or ["?"] * len(pr.corpus)
    return evaluate_precision_at_k(pr.features.sim, labels, k=mc.top_k, collapse_duplicates=mc.collapse_duplicates,
                                   dup_threshold=pr.dedup.threshold)
