# ReviewLens: Text Mining & Document Intelligence System

**BITS WILP · Information Retrieval (AIMLCZG537 / DSECLZG537) · S1-26 · Assignment 2**
Group: `G47` (see `G47_Contribution.xlsx`)

ReviewLens is a Streamlit application for a marketplace's customer-experience team. It analyses free-text
**product reviews from six departments**. It can:

* **route** each review to its department with a manually implemented Multinomial **Naive Bayes** classifier,
* **discover themes** without labels with a manually implemented **K-Means**, evaluated with manual **Purity** and **Rand Index**,
* **recommend related reviews** with a content-based **Top-K** recommender (TF-IDF cosine similarity through an inverted index),
* show corpus analytics, duplicate analysis and performance experiments.

No library implementation of TF-IDF, Naive Bayes, K-Means, Purity or Rand Index is used. There is no
scikit-learn import anywhere in the project.

---

## 1. Installation

Python 3.9 or newer. The commands are the same on Linux and macOS.

```bash
cd G47_Assignment2                  # the unzipped folder
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: lemmatisation uses the WordNet data of NLTK. The app downloads it automatically when internet is
available. On an offline machine, run this once while online, or skip it; the app then falls back to a built-in
rule-based lemmatiser and shows which backend is active.

```bash
python -m nltk.downloader wordnet omw-1.4
```

## 2. Run the application

```bash
streamlit run app.py
```

Open the URL printed in the terminal (normally http://localhost:8501). On a remote or virtual machine whose
browser runs elsewhere, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The bundled dataset is loaded automatically, so every page works immediately.

## 3. Reproduce the report numbers and run the tests

```bash
python tests/test_core.py            # 12 correctness tests (textbook examples of Manning et al., IIR)
python scripts/run_experiments.py    # writes every table / number used in the report to results/ (~2-3 min)
```

The app and the script share the same code and the same seeded defaults, so the default-configuration numbers
in the app are identical to `results/summary.json`. All 23 result files and every value in `summary.json` were
verified to agree to 9 decimal places on Python 3.9 (numpy 1.24, Streamlit 1.30), Python 3.12 (numpy 2.5, Streamlit
1.66) and Python 3.14 (numpy 2.5, Streamlit 1.67). The one exception is the optional lemmatisation row of the ablation, which depends on the installed
NLTK/WordNet version.

Optional: `python scripts/build_dataset.py` re-downloads and re-samples the corpus. This needs internet and is
**not** needed: the frozen dataset is in `data/`.

## 4. Application workflow (sidebar pages)

| Page | What it shows |
|---|---|
| **Dashboard** | business context, KPIs, headline results, category / year / rating / length charts, workflow |
| **Data Ingestion** | bundled data or upload (CSV, JSON, JSONL, ZIP folder of .txt), column mapping, text and metadata kept in separate tables, corpus statistics, noise profile |
| **Preprocessing & Features** | configurable pipeline (markup, contractions, lower-case, stop words, keep negations, Porter stemming / WordNet lemmatisation, n-grams, df pruning, tf/idf variants, duplicate policy). Tabs: step-by-step pipeline preview, vocabulary and term frequencies before vs after, inverted-index explorer and TF-IDF vector inspector, cosine similarity, duplicate detection |
| **Text Mining Hub** | Naive Bayes (actual vs predicted, confusion matrix, metrics, discriminative terms, try-it box); K-Means (seed documents, iterations, RSS, stopping reason, 2-D scatter, cluster sizes, membership, top terms); Purity / Rand Index with chance baselines and a comparison with Naive Bayes; corpus insights |
| **Recommendations** | choose a review or type text, Top-K results with cosine scores, metadata and shared terms, P@K evaluation, method and limitations |
| **Performance Analytics** | scorecard, run times, preprocessing ablation (Q1), duplicate-leakage study (Q2), K-Means variants and k-sensitivity (Q5) |

## 5. Where each algorithm is implemented

| Requirement | File / function |
|---|---|
| Tokenisation, stop words, n-grams | `textmining/preprocessing.py` (`tokenize`, `stopword_set`, `make_ngrams`, `Preprocessor`) |
| Inverted index | `textmining/indexing.py` · `InvertedIndex` |
| TF-IDF with cosine length normalisation | `textmining/indexing.py` · `tf_weight`, `idf_weight`, `TfidfModel` |
| Cosine similarity | `textmining/indexing.py` · `cosine_similarity`, `TfidfModel.score` (term-at-a-time), `similarity_matrix` |
| Multinomial Naive Bayes, split, metrics | `textmining/classification.py` · `MultinomialNaiveBayes`, `stratified_split`, `classification_report`, `confusion_matrix` |
| K-Means (k-means++ / random init, Euclidean / cosine) | `textmining/clustering.py` · `kmeans_single`, `kmeans` |
| Purity, Rand Index | `textmining/evaluation.py` · `purity`, `pair_counts_bruteforce`, `pair_counts`, `rand_index` |
| Duplicate detection | `textmining/duplicates.py` · `detect_duplicates` |
| Top-K recommender | `textmining/recommendation.py` · `ContentRecommender` |
| Experiments for the report | `textmining/experiments.py`, `scripts/run_experiments.py` |

numpy is used only for vector arithmetic (dot products, means, eigen-decomposition for the 2-D plot) and NLTK
only for the Porter stemmer and the WordNet lemmatiser.

## 6. Dataset

`data/amazon_reviews_corpus.json` holds text and metadata separately. `data/amazon_reviews_corpus.csv` is the
same data as a flat table.

* **Source:** Amazon Reviews 2023, McAuley Lab, UC San Diego. Hou et al. (2024), *Bridging Language and Items
  for Retrieval and Recommendation*, arXiv:2403.03952 (https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
* **Documents:** 1,224 reviews from 6 departments: Appliances, Beauty, Musical Instruments, Pet Supplies, Software, Video Games.
* **Fields:** `doc_id` (R0001 ...), `text`, plus metadata `title`, `category` (label), `rating`, `date`,
  `helpful_votes`, `verified_purchase`, `has_images`, `asin`, `parent_asin`, `user_id`, `source`, and the
  duplicate ground truth `dup_group`, `dup_type`, `is_injected`.
* **Noise** (`results/noise_profile.csv`): HTML markup in 18.3% of reviews, ALL-CAPS words 21.7%, repeated punctuation 16.5%, emoji and curly quotes 16.3%, HTML entities 3.5%, digits and model numbers 41.1%.
* **Duplicates:** 36 groups of naturally re-posted reviews from the source data plus 24 documented near-duplicate
  variants (`is_injected = True`), 9.9% of the corpus in total, used to measure duplicate detection and its effects.
* Sampling is seeded and documented in `scripts/build_dataset.py`.

## 7. Folder structure

```
app.py                     Streamlit entry point (sidebar navigation)
views/                     one module per page + shared UI helpers
textmining/                manual IR / text-mining library (no scikit-learn)
data/                      dataset (JSON with separate metadata, and CSV)
scripts/                   build_dataset.py, run_experiments.py
tests/test_core.py         correctness tests against textbook examples
results/                   CSV / JSON outputs used in the report
report/                    the report (PDF)
evidence/                  execution log, screenshots and the screen recording of the local run
.streamlit/config.toml     theme
```
