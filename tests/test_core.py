"""
Correctness tests for the manual implementations, using the worked examples of
Manning, Raghavan & Schütze, "Introduction to Information Retrieval" (Cambridge University Press, 2008).

Run:   python tests/test_core.py        (or: python -m pytest tests)
"""
from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textmining.classification import MultinomialNaiveBayes, classification_report, confusion_matrix, stratified_split
from textmining.clustering import kmeans
from textmining.duplicates import detect_duplicates, evaluate_detection
from textmining.evaluation import pair_counts, pair_counts_bruteforce, purity, rand_index
from textmining.indexing import InvertedIndex, TfidfModel, cosine_similarity, idf_weight, l2_normalize, similarity_matrix, tf_weight
from textmining.preprocessing import PreprocessConfig, Preprocessor
from textmining.recommendation import ContentRecommender


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------------ IIR Example 13.1 (Naive Bayes)
def test_naive_bayes_textbook_example():
    train = [("Chinese Beijing Chinese", "china"), ("Chinese Chinese Shanghai", "china"),
             ("Chinese Macao", "china"), ("Tokyo Japan Chinese", "not-china")]
    docs = [t.split() for t, _ in train]
    labels = [c for _, c in train]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(docs, labels)
    assert approx(math.exp(nb.log_prior_["china"]), 3 / 4)
    assert approx(math.exp(nb.log_likelihood_["china"]["Chinese"]), 3 / 7)      # (5+1)/(8+6)
    assert approx(math.exp(nb.log_likelihood_["china"]["Tokyo"]), 1 / 14)       # (0+1)/(8+6)
    assert approx(math.exp(nb.log_likelihood_["not-china"]["Chinese"]), 2 / 9)  # (1+1)/(3+6)
    test_doc = "Chinese Chinese Chinese Tokyo Japan".split()
    scores = nb.joint_log_scores(test_doc)
    assert approx(math.exp(scores["china"]), 0.0003, 5e-5)       # 3/4 * (3/7)^3 * 1/14 * 1/14
    assert approx(math.exp(scores["not-china"]), 0.0001, 5e-5)   # 1/4 * (2/9)^3 * 2/9 * 2/9
    assert nb.predict([test_doc]) == ["china"]


# ------------------------------------------------------------------ IIR Figure 16.4 (Purity, Rand Index)
def _iir_fig_16_4():
    clusters = [1] * 6 + [2] * 6 + [3] * 5
    labels = ["x"] * 5 + ["o"] + ["x"] + ["o"] * 4 + ["d"] + ["x"] * 2 + ["d"] * 3
    return clusters, labels


def test_purity_textbook_example():
    clusters, labels = _iir_fig_16_4()
    assert approx(purity(clusters, labels), 12 / 17)            # (5 + 4 + 3) / 17 = 0.71


def test_rand_index_textbook_example():
    clusters, labels = _iir_fig_16_4()
    assert pair_counts(clusters, labels) == {"TP": 20, "FP": 20, "FN": 24, "TN": 72}
    assert pair_counts_bruteforce(clusters, labels) == {"TP": 20, "FP": 20, "FN": 24, "TN": 72}
    assert approx(rand_index(clusters, labels), 92 / 136)       # 0.68


def test_rand_index_two_methods_agree():
    rng = random.Random(1)
    clusters = [rng.randrange(5) for _ in range(300)]
    labels = [rng.choice("abcd") for _ in range(300)]
    assert pair_counts(clusters, labels) == pair_counts_bruteforce(clusters, labels)
    assert approx(rand_index(clusters, labels, "pairs"), rand_index(clusters, labels, "contingency"), 1e-12)


# ------------------------------------------------------------------ IIR section 6.4, lnc.ltc worked example (tf-idf, cosine normalisation)
def test_tfidf_textbook_example():
    n_docs = 1_000_000
    df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
    q_tf = {"best": 1, "car": 1, "insurance": 1}
    d_tf = {"auto": 1, "car": 1, "insurance": 2}
    q = l2_normalize({t: tf_weight(tf, "log") * idf_weight(df[t], n_docs, "idf") for t, tf in q_tf.items()})  # ltc
    d = l2_normalize({t: tf_weight(tf, "log") for t, tf in d_tf.items()})                                      # lnc
    assert approx(q["best"], 0.34, 0.01) and approx(q["car"], 0.52, 0.01) and approx(q["insurance"], 0.78, 0.01)
    assert approx(d["auto"], 0.52, 0.01) and approx(d["insurance"], 0.68, 0.01)
    assert approx(sum(q[t] * d.get(t, 0) for t in q), 0.80, 0.01)
    assert approx(cosine_similarity(q, d), 0.80, 0.01)


def test_inverted_index_and_tfidf_by_hand():
    docs = [["apple", "apple", "banana"], ["banana", "cherry"], ["apple", "cherry", "cherry", "date"]]
    index = InvertedIndex(docs)
    assert index.postings["apple"] == [(0, 2), (2, 1)]
    assert index.df == {"apple": 2, "banana": 2, "cherry": 2, "date": 1}
    model = TfidfModel(index, tf_scheme="log", idf_scheme="idf")
    idf2, idf1 = math.log10(3 / 2), math.log10(3)
    raw = {"apple": (1 + math.log10(2)) * idf2, "banana": idf2}
    norm = math.sqrt(sum(v * v for v in raw.values()))
    assert approx(model.doc_vectors[0]["apple"], raw["apple"] / norm, 1e-9)
    assert approx(model.idf["date"], idf1, 1e-12)
    for vec in model.doc_vectors:                                  # unit length after normalisation
        assert approx(math.sqrt(sum(w * w for w in vec.values())), 1.0, 1e-9)


def test_index_scoring_equals_matrix_cosine():
    pre = Preprocessor(PreprocessConfig(ngram_max=1))
    texts = ["The battery died after two days", "Battery life is great, charging is fast",
             "My dog loves this chew toy", "The dog toy fell apart", "Great sound from this guitar amp"]
    model = TfidfModel(InvertedIndex(pre.process_corpus(texts)))
    S = similarity_matrix(model.to_dense())
    scores = model.score(model.doc_vectors[0])
    for j in range(len(texts)):
        assert approx(scores.get(j, 0.0), S[0, j], 1e-9)
    rec = ContentRecommender(model).recommend_for_doc(2, k=2, collapse_duplicates=False)
    assert rec[0]["doc_idx"] == 3                                # the other dog-toy review


# ------------------------------------------------------------------ K-Means, split, preprocessing, duplicates
def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.05, (20, 3)) + [1, 0, 0], rng.normal(0, 0.05, (20, 3)) + [0, 1, 0]])
    res = kmeans(X, 2, init="k-means++", n_init=3, seed=7)
    labels = res.labels
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]
    assert "converged" in res.best.stop_reason
    rss = [h["RSS"] for h in res.best.history]
    assert all(rss[i + 1] <= rss[i] + 1e-9 for i in range(len(rss) - 1))   # RSS never increases


def test_stratified_split_keeps_groups_together():
    labels = ["a"] * 40 + ["b"] * 40
    groups = [None] * 80
    groups[0] = groups[1] = "g1"
    groups[50] = groups[51] = "g2"
    train, test = stratified_split(labels, 0.25, seed=3, groups=groups)
    assert len(test) == 20 and set(train).isdisjoint(test)
    for a, b in ((0, 1), (50, 51)):
        assert (a in test) == (b in test)


def test_metrics_by_hand():
    y_true = ["a", "a", "b", "b", "b"]
    y_pred = ["a", "b", "b", "b", "a"]
    rep = classification_report(y_true, y_pred, ["a", "b"])
    assert approx(rep["accuracy"], 3 / 5)
    a = rep["per_class"][0]
    assert approx(a["precision"], 1 / 2) and approx(a["recall"], 1 / 2)
    assert confusion_matrix(y_true, y_pred, ["a", "b"]) == [[1, 1], [1, 2]]


def test_preprocessing_pipeline():
    pre = Preprocessor(PreprocessConfig(normalizer="none", ngram_max=2, custom_stopwords=()))
    terms = pre("It DIDN'T work!!<br /><br />Not worth &amp; the money")
    assert "br" not in terms and "amp" not in terms           # markup and entities removed
    assert "not" in terms and "not work" in terms              # negation kept and joined in a bigram
    assert all(t == t.lower() for t in terms)


def test_duplicate_detection():
    texts = ["Great guitar strings, bright tone and they stay in tune for weeks.",
             "GREAT guitar strings,<br />bright tone and they stay in tune for weeks!!",
             "The litter box is easy to clean and my cats like it.",
             "Terrible software, crashes on start up every single time."]
    pre = Preprocessor(PreprocessConfig())
    model = TfidfModel(InvertedIndex([pre(t) for t in texts]))
    rep = detect_duplicates(texts, similarity_matrix(model.to_dense()), 0.9)
    assert rep.groups == [[0, 1]] and rep.exact_groups == [[0, 1]]
    ev = evaluate_detection(rep.groups, ["g", "g", None, None])
    assert ev["precision"] == 1.0 and ev["recall"] == 1.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
