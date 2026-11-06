"""
Experiments that produce the evidence for the compulsory inference questions (report section F).
Used by the Streamlit 'Performance analytics' page and by scripts/run_experiments.py.

* preprocessing ablation  (Q1) - vocabulary, Naive Bayes, K-Means and recommendation quality per pipeline
* duplicate study         (Q2) - leakage through duplicates: random split vs duplicate-aware split vs de-duplication
* K-Means variants / k    (Q5) - initialisation, distance, number of clusters, stability over seeds
* detection threshold     (Q2) - precision / recall of the near-duplicate detector against ground truth
NB results are averaged over several random train/test splits and K-Means over several seeds, because a
single run of either can be lucky or unlucky.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .duplicates import evaluate_detection, threshold_sweep
from .ingestion import Corpus
from .pipeline import (DedupConfig, FeatureConfig, ModelConfig, PipelineResult, run_classification, run_clustering,
                       run_pipeline, run_recommendation_eval)
from .preprocessing import PreprocessConfig

DEFAULT_PREP = PreprocessConfig()
DEFAULT_FEAT = FeatureConfig()
DEFAULT_DEDUP = DedupConfig()
DEFAULT_MODEL = ModelConfig()

ABLATIONS: List[Tuple[str, PreprocessConfig]] = [
    ("A0 minimal: regex tokens only (markup, case, stop words kept)",
     PreprocessConfig(strip_markup=False, remove_urls=False, expand_contractions=False, lowercase=False,
                      remove_numbers=False, min_token_len=1, remove_stopwords=False, custom_stopwords=(),
                      normalizer="none", ngram_max=1)),
    ("A1 + markup removal & lower-casing",
     PreprocessConfig(remove_stopwords=False, custom_stopwords=(), normalizer="none", ngram_max=1)),
    ("A2 + stop-word removal", PreprocessConfig(normalizer="none", ngram_max=1)),
    ("A3 + Porter stemming (default)", PreprocessConfig(normalizer="stem", ngram_max=1)),
    ("A4 + lemmatisation instead of stemming", PreprocessConfig(normalizer="lemma", ngram_max=1)),
    ("A5 + stemming + bigrams", PreprocessConfig(normalizer="stem", ngram_max=2)),
    ("A6 + stemming + 1-3 grams", PreprocessConfig(normalizer="stem", ngram_max=3)),
]


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = list(values)
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


def evaluate_pipeline(pr: PipelineResult, mc: ModelConfig, n_split_seeds: int = 10, n_cluster_seeds: int = 10) -> Dict[str, float]:
    """NB over several splits, K-Means over several seeds, recommender once (deterministic)."""
    t = time.perf_counter()
    nb = [run_classification(pr, replace(mc, seed=mc.seed + s)).report for s in range(n_split_seeds)]
    km = [run_clustering(pr, replace(mc, seed=mc.seed + 100 * s), with_projection=False) for s in range(n_cluster_seeds)]
    rec = run_recommendation_eval(pr, mc)
    acc, acc_sd = _mean_std(r["accuracy"] for r in nb)
    f1, f1_sd = _mean_std(r["macro"]["f1"] for r in nb)
    pur, pur_sd = _mean_std(c.purity for c in km)
    ri, ri_sd = _mean_std(c.rand_index for c in km)
    n_terms = [len(t) for t in pr.features.terms]
    return {"documents": len(pr.corpus), "vocabulary (all terms)": pr.features.index.full_vocabulary_size,
            "vocabulary (after df pruning)": len(pr.features.index), "mean terms / doc": statistics.mean(n_terms),
            "NB accuracy": acc, "NB accuracy sd": acc_sd, "NB macro-F1": f1, "NB macro-F1 sd": f1_sd,
            "K-Means purity": pur, "K-Means purity sd": pur_sd, "K-Means Rand Index": ri, "K-Means RI sd": ri_sd,
            f"Rec P@{mc.top_k}": rec["precision_at_k"], "Rec MRR": rec["mrr"], "seconds": time.perf_counter() - t}


def ablation_study(corpus: Corpus, feat: FeatureConfig = DEFAULT_FEAT, dedup: DedupConfig = DEFAULT_DEDUP,
                   mc: ModelConfig = DEFAULT_MODEL, configs: Sequence[Tuple[str, PreprocessConfig]] = ABLATIONS,
                   progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, object]]:
    """Every preprocessing variant is evaluated on the SAME de-duplicated document set (detected once with the
    default pipeline) so that only the preprocessing differs between rows."""
    base = run_pipeline(corpus, DEFAULT_PREP, feat, dedup)
    rows = []
    for i, (name, prep) in enumerate(configs):
        if progress:
            progress(i / len(configs), name)
        pr = run_pipeline(corpus, prep, feat, dedup, keep_override=base.kept_indices)
        rows.append({"configuration": name, "pipeline": prep.describe(), **evaluate_pipeline(pr, mc)})
    if progress:
        progress(1.0, "done")
    return rows


def duplicate_study(corpus: Corpus, prep: PreprocessConfig = DEFAULT_PREP, feat: FeatureConfig = DEFAULT_FEAT,
                    dedup: DedupConfig = DEFAULT_DEDUP, mc: ModelConfig = DEFAULT_MODEL, n_seeds: int = 10,
                    progress: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, object]]:
    """
    S1 keep duplicates + plain random split    (a review and its copy can land in train AND test -> leakage)
    S2 keep duplicates + duplicate-aware split (copies always stay together)
    S3 remove near-duplicates before analysis  (default handling)
    For S1/S2 the NB accuracy is also reported separately for test reviews whose duplicate is in training.
    """
    settings = [("S1 keep duplicates, random split", "keep_all", False),
                ("S2 keep duplicates, duplicate-aware split", "keep_all", True),
                ("S3 remove near-duplicates (default)", "remove_near", True)]
    rows = []
    for si, (name, policy, group_split) in enumerate(settings):
        if progress:
            progress(si / len(settings), name)
        pr = run_pipeline(corpus, prep, feat, replace(dedup, policy=policy))
        groups = pr.group_ids
        accs, leak_hits, leak_n, clean_hits, clean_n = [], 0, 0, 0, 0
        for s in range(n_seeds):
            cr = run_classification(pr, replace(mc, seed=mc.seed + s, group_split=group_split))
            accs.append(cr.report["accuracy"])
            train_groups = {groups[i] for i in cr.train_idx if groups[i]}
            for pos, i in enumerate(cr.test_idx):
                ok = cr.y_true[pos] == cr.y_pred[pos]
                if groups[i] and groups[i] in train_groups:
                    leak_n += 1
                    leak_hits += ok
                else:
                    clean_n += 1
                    clean_hits += ok
        km = [run_clustering(pr, replace(mc, seed=mc.seed + 100 * s), with_projection=False) for s in range(10)]
        rec_raw = run_recommendation_eval(pr, replace(mc, collapse_duplicates=False))
        rec_col = run_recommendation_eval(pr, replace(mc, collapse_duplicates=True))
        acc, acc_sd = _mean_std(accs)
        group_sizes: Dict[str, int] = {}
        for g in groups:
            if g:
                group_sizes[g] = group_sizes.get(g, 0) + 1
        rows.append({
            "setting": name, "documents": len(pr.corpus),
            "documents in duplicate groups": sum(n for n in group_sizes.values() if n > 1),
            "NB accuracy (mean of splits)": acc, "NB accuracy sd": acc_sd,
            "test docs with a copy in training (total over splits)": leak_n,
            "NB accuracy on those docs": (leak_hits / leak_n) if leak_n else None,
            "NB accuracy on other test docs": (clean_hits / clean_n) if clean_n else None,
            "K-Means purity": statistics.mean(c.purity for c in km),
            "K-Means Rand Index": statistics.mean(c.rand_index for c in km),
            f"Rec P@{mc.top_k} (no collapsing)": rec_raw["precision_at_k"],
            f"near-copies in Top-{mc.top_k} (no collapsing)": rec_raw["duplicate_rate"],
            f"Rec P@{mc.top_k} (collapse duplicates)": rec_col["precision_at_k"],
        })
    if progress:
        progress(1.0, "done")
    return rows


def kmeans_variants(pr: PipelineResult, mc: ModelConfig = DEFAULT_MODEL, n_seeds: int = 10) -> List[Dict[str, object]]:
    rows = []
    for init in ("random", "k-means++"):
        for metric in ("euclidean", "cosine"):
            runs = [run_clustering(pr, replace(mc, init=init, metric=metric, n_init=1, seed=mc.seed + s),
                                   with_projection=False) for s in range(n_seeds)]
            restarts = run_clustering(pr, replace(mc, init=init, metric=metric), with_projection=False)
            pur, pur_sd = _mean_std(r.purity for r in runs)
            ri, ri_sd = _mean_std(r.rand_index for r in runs)
            rows.append({"initialisation": init, "distance": metric,
                         "purity (single runs, mean)": pur, "purity sd": pur_sd,
                         "Rand Index (single runs, mean)": ri, "RI sd": ri_sd,
                         "mean iterations": statistics.mean(r.result.best.n_iter for r in runs),
                         f"purity (best of {mc.n_init} restarts)": restarts.purity,
                         f"Rand Index (best of {mc.n_init} restarts)": restarts.rand_index})
    return rows


def k_sensitivity(pr: PipelineResult, mc: ModelConfig = DEFAULT_MODEL, ks: Sequence[int] = tuple(range(2, 13)),
                  n_seeds: int = 3) -> List[Dict[str, object]]:
    rows = []
    for k in ks:
        runs = [run_clustering(pr, replace(mc, k=k, seed=mc.seed + 100 * s), with_projection=False) for s in range(n_seeds)]
        rows.append({"k": k, "RSS": statistics.mean(r.result.best.rss for r in runs),
                     "purity": statistics.mean(r.purity for r in runs),
                     "Rand Index": statistics.mean(r.rand_index for r in runs)})
    return rows


def detection_sweep(pr: PipelineResult, truth_field: str = "dup_group",
                    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)) -> List[Dict[str, object]]:
    meta = pr.full_corpus.metadata
    if truth_field not in meta.columns:
        return []
    truth = [str(v) if str(v).strip() else None for v in meta[truth_field].tolist()]
    return threshold_sweep(pr.full_corpus.texts, pr.full_features.sim, truth, thresholds)


def detection_quality(pr: PipelineResult, truth_field: str = "dup_group") -> Optional[Dict[str, float]]:
    meta = pr.full_corpus.metadata
    if truth_field not in meta.columns:
        return None
    truth = [str(v) if str(v).strip() else None for v in meta[truth_field].tolist()]
    return evaluate_detection(pr.duplicates.groups, truth)
