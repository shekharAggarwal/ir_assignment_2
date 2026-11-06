"""
Reproduce every number used in the report (writes CSV / JSON files to results/).

    python scripts/run_experiments.py           # ~1-2 minutes on a laptop

The Streamlit app uses the same textmining package and the same default configuration, so the
'default run' numbers here are identical to what the app shows on first launch.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import replace
from importlib import metadata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from textmining.analytics import cluster_rating_summary, collaborative_feasibility, department_insights  # noqa: E402
from textmining.evaluation import chance_baseline  # noqa: E402
from textmining.experiments import (DEFAULT_DEDUP, DEFAULT_FEAT, DEFAULT_MODEL, DEFAULT_PREP, ablation_study,  # noqa: E402
                                    detection_quality, detection_sweep, duplicate_study, evaluate_pipeline,
                                    k_sensitivity, kmeans_variants)
from textmining.recommendation import precision_at_k_by_label  # noqa: E402
from textmining.ingestion import corpus_statistics, load_bundled, noise_profile  # noqa: E402
from textmining.pipeline import (recommender_for, run_classification, run_clustering, run_pipeline,  # noqa: E402
                                 run_recommendation_eval)
from textmining.preprocessing import Preprocessor, backend_names, raw_tokens, vocabulary_funnel  # noqa: E402

FREE_TEXT_QUERY = "The leash broke after a week and my dog got loose, the stitching is poor quality."

OUT = os.path.join(ROOT, "results")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"  wrote results/{name} ({len(df)} rows)")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def claim_evidence(pr, cr) -> dict:
    """Measurements behind statements made in the report text."""
    bigram_pre = Preprocessor(replace(DEFAULT_PREP, ngram_max=2))
    train = [bigram_pre(pr.corpus.texts[i]) for i in cr.train_idx]
    bigrams = Counter(t for doc in train for t in doc if " " in t)
    unigrams = Counter(t for doc in train for t in doc if " " not in t)
    labels = np.array(pr.corpus.labels)
    iu = np.triu_indices(len(labels), 1)
    same = labels[iu[0]] == labels[iu[1]]
    sims = pr.features.sim[iu]
    return {"train_bigram_types": len(bigrams), "train_bigrams_seen_once": sum(v == 1 for v in bigrams.values()) / len(bigrams),
            "train_unigram_types": len(unigrams), "train_unigrams_seen_once": sum(v == 1 for v in unigrams.values()) / len(unigrams),
            "mean_cosine_same_category": float(sims[same].mean()), "mean_cosine_different_category": float(sims[~same].mean())}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    corpus, log = load_bundled("json")
    print("ingestion:", log)
    stats = corpus_statistics(corpus)
    save(noise_profile(corpus), "noise_profile.csv")

    # ------------------------------------------------------------ default pipeline (what the app shows)
    pr = run_pipeline(corpus, DEFAULT_PREP, DEFAULT_FEAT, DEFAULT_DEDUP)
    mc = DEFAULT_MODEL
    det = detection_quality(pr)
    cr = run_classification(pr, mc)
    cl = run_clustering(pr, mc)
    rec_eval = run_recommendation_eval(pr, mc)

    save(pd.DataFrame(vocabulary_funnel(pr.corpus.texts, DEFAULT_PREP)), "vocabulary_funnel.csv")
    before = Counter(t for txt in pr.corpus.texts for t in raw_tokens(txt)).most_common(25)
    after = Counter(t for terms in pr.features.terms for t in terms).most_common(25)
    save(pd.DataFrame({"rank": range(1, 26), "raw token": [t for t, _ in before], "raw freq": [c for _, c in before],
                       "processed term": [t for t, _ in after], "processed freq": [c for _, c in after]}),
         "top_terms_before_after.csv")

    meta = pr.corpus.metadata
    titles = pr.corpus.titles
    # duplicates found on the full corpus
    dup_rows = []
    for gi, g in enumerate(pr.duplicates.groups, start=1):
        for i in g:
            dup_rows.append({"group": f"G{gi:03d}", "doc_id": corpus.doc_ids[i],
                             "category": corpus.labels[i], "kept": i in set(pr.kept_indices),
                             "text": corpus.texts[i][:140]})
    save(pd.DataFrame(dup_rows), "duplicate_groups.csv")

    # classification outputs
    save(pd.DataFrame({"doc_id": [pr.corpus.doc_ids[i] for i in cr.test_idx],
                       "title": [titles[i] for i in cr.test_idx], "actual": cr.y_true, "predicted": cr.y_pred,
                       "confidence": [round(c, 4) for c in cr.confidence],
                       "correct": [a == p for a, p in zip(cr.y_true, cr.y_pred)]}), "nb_predictions.csv")
    save(pd.DataFrame(cr.report["per_class"]), "nb_per_class_metrics.csv")
    cm = pd.DataFrame(cr.matrix, index=cr.labels, columns=cr.labels)
    cm.index.name = "actual \\ predicted"
    cm.reset_index().to_csv(os.path.join(OUT, "nb_confusion_matrix.csv"), index=False)
    save(pd.DataFrame([{"class": c, "rank": r + 1, "term": t, "log-odds": round(s, 3)}
                       for c in cr.labels for r, (t, s) in enumerate(cr.model.top_terms(c, 10))]), "nb_top_terms.csv")

    # clustering outputs
    save(pd.DataFrame({"doc_id": pr.corpus.doc_ids, "title": titles, "true label": pr.corpus.labels,
                       "cluster": cl.labels_pred, "x": cl.coords[:, 0], "y": cl.coords[:, 1]}), "cluster_assignments.csv")
    save(pd.DataFrame(cl.result.best.history), "kmeans_iterations.csv")
    save(pd.DataFrame([{"restart seed": r.seed, "RSS": r.rss, "iterations": r.n_iter, "stop": r.stop_reason,
                        "seed documents": ", ".join(pr.corpus.doc_ids[i] for i in r.init_indices)}
                       for r in cl.result.runs]), "kmeans_restarts.csv")
    save(pd.DataFrame([{"cluster": k, "size": size, "majority label": lab, "majority share": cnt / size,
                        "top terms": ", ".join(t for t, _ in cl.top_terms[k][:10])}
                       for k, (lab, cnt, size) in sorted(cl.majority.items())]), "cluster_summary.csv")
    cont = pd.crosstab(pd.Series(cl.labels_pred, name="cluster"), pd.Series(pr.corpus.labels, name="true label"))
    cont.reset_index().to_csv(os.path.join(OUT, "cluster_contingency.csv"), index=False)

    # recommendation examples (fixed, documented query documents)
    rec = recommender_for(pr)
    examples = []
    query_ids = []
    for cat in sorted(set(pr.corpus.labels)):
        cand = [i for i, lab in enumerate(pr.corpus.labels) if lab == cat and 40 <= len(pr.corpus.texts[i].split()) <= 90]
        query_ids.append(cand[0])
    for q in query_ids:
        for r in rec.recommend_for_doc(q, k=mc.top_k, collapse_duplicates=True, dup_threshold=DEFAULT_DEDUP.threshold):
            d = r["doc_idx"]
            examples.append({"query doc_id": pr.corpus.doc_ids[q], "query category": pr.corpus.labels[q],
                             "query title": titles[q], "rank": r["rank"], "doc_id": pr.corpus.doc_ids[d],
                             "score": round(r["score"], 4), "category": pr.corpus.labels[d], "title": titles[d],
                             "rating": meta.iloc[d]["rating"], "date": meta.iloc[d]["date"],
                             "shared terms": ", ".join(t for t, _ in r["shared_terms"][:5])})
    save(pd.DataFrame(examples), "recommendation_examples.csv")
    qvec = pr.features.model.vectorize(Preprocessor(DEFAULT_PREP)(FREE_TEXT_QUERY))
    free = [{"query": FREE_TEXT_QUERY, "rank": r["rank"], "doc_id": pr.corpus.doc_ids[r["doc_idx"]],
             "score": round(r["score"], 4), "category": pr.corpus.labels[r["doc_idx"]], "title": titles[r["doc_idx"]],
             "rating": meta.iloc[r["doc_idx"]]["rating"], "date": meta.iloc[r["doc_idx"]]["date"],
             "shared terms": ", ".join(t for t, _ in r["shared_terms"][:5])}
            for r in rec.recommend(qvec, k=mc.top_k, dup_threshold=DEFAULT_DEDUP.threshold)]
    save(pd.DataFrame(free), "recommendation_free_text.csv")
    save(pd.DataFrame(precision_at_k_by_label(pr.features.sim, pr.corpus.labels, mc.top_k, True,
                                              DEFAULT_DEDUP.threshold)), "recommendation_p_at_k_by_category.csv")

    # analyst insights
    ratings = pd.to_numeric(meta["rating"], errors="coerce").fillna(3).tolist()
    save(pd.DataFrame(department_insights(pr.features.terms, pr.corpus.labels, ratings)), "department_insights.csv")
    save(pd.DataFrame(cluster_rating_summary(cl.labels_pred, ratings, cl.majority, cl.top_terms)),
         "cluster_rating_summary.csv")

    summary = {
        "corpus": {**stats, **log}, "backends": backend_names(),
        "default_config": {"preprocessing": DEFAULT_PREP.describe(), "features": DEFAULT_FEAT.__dict__,
                           "dedup": DEFAULT_DEDUP.__dict__, "model": DEFAULT_MODEL.__dict__},
        "analysis_documents": len(pr.corpus),
        "vocabulary": {"all terms": pr.features.index.full_vocabulary_size, "after pruning": len(pr.features.index),
                       "postings": pr.features.index.n_postings(), "matrix shape": list(pr.features.X.shape),
                       "matrix density": float((pr.features.X > 0).mean())},
        "duplicates": {"groups": len(pr.duplicates.groups), "exact groups": len(pr.duplicates.exact_groups),
                       "near pairs": len(pr.duplicates.near_pairs), "docs removed": len(corpus) - len(pr.corpus),
                       "detection vs ground truth": det},
        "naive_bayes": {"accuracy": cr.report["accuracy"], "macro": cr.report["macro"],
                        "weighted": cr.report["weighted"], "n_train": len(cr.train_idx), "n_test": len(cr.test_idx)},
        "kmeans": {"k": cl.k, "purity": cl.purity, "rand_index": cl.rand_index, "pair_counts": cl.pair_counts,
                   "rss": cl.result.best.rss, "iterations": cl.result.best.n_iter,
                   "stop_reason": cl.result.best.stop_reason, "pca_explained": list(cl.explained),
                   "cluster_sizes": Counter(cl.labels_pred)},
        "recommendation": rec_eval, "timings": pr.timings,
        "chance_baseline": chance_baseline(cl.labels_pred, pr.corpus.labels),
        "majority_class_rate_test": max(Counter(cr.y_true).values()) / len(cr.y_true),
        "variability_default_config": evaluate_pipeline(pr, mc),
        "collaborative_feasibility": collaborative_feasibility(corpus.metadata["user_id"].astype(str).tolist(),
                                                               corpus.metadata["parent_asin"].astype(str).tolist()),
        "dataset_duplicates": {
            "ground-truth groups": int(corpus.metadata["dup_group"].replace("", pd.NA).dropna().nunique()),
            "documents in ground-truth groups": int((corpus.metadata["dup_group"].astype(str).str.strip() != "").sum()),
            "injected variants": int(corpus.metadata["is_injected"].astype(str).isin(["True", "true", "1"]).sum()),
        },
        "free_text_query": FREE_TEXT_QUERY,
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                        "macos": platform.mac_ver()[0],
                        **{p: package_version(p) for p in ("streamlit", "pandas", "numpy", "plotly", "nltk")}},
        "claim_evidence": claim_evidence(pr, cr),
        "recommendation_p_at_1": run_recommendation_eval(pr, replace(mc, top_k=1))["precision_at_k"],
    }

    # ------------------------------------------------------------ experiments
    print("ablation study ...")
    save(pd.DataFrame(ablation_study(corpus)), "ablation_preprocessing.csv")
    print("duplicate study ...")
    save(pd.DataFrame(duplicate_study(corpus)), "duplicate_study.csv")
    print("K-Means variants / k sensitivity ...")
    save(pd.DataFrame(kmeans_variants(pr)), "kmeans_variants.csv")
    save(pd.DataFrame(k_sensitivity(pr)), "k_sensitivity.csv")
    save(pd.DataFrame(detection_sweep(pr)), "duplicate_threshold_sweep.csv")

    summary["runtime_seconds"] = time.perf_counter() - t0
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summary[k] for k in ("naive_bayes", "kmeans", "recommendation", "duplicates")}, indent=1,
                     default=str)[:2500])
    print(f"done in {summary['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
