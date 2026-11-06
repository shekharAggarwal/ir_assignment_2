"""Dashboard and corpus overview."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from textmining.ingestion import corpus_statistics
from textmining.evaluation import chance_baseline
from views import common as c

RATING_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, 1 -> 5 stars


def render() -> None:
    c.page_header("Dashboard & corpus overview",
                  "Voice-of-Customer analytics for an online marketplace: route, group and relate customer "
                  "reviews across six product departments.")
    corpus = c.corpus()
    pr = c.pipeline()
    stats = corpus_statistics(corpus)

    with c.card():
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Business context.** The marketplace's customer-experience team receives free-text "
                        "reviews for every department. ReviewLens helps analysts **(1)** route each review to the "
                        "responsible department automatically (Naive Bayes), **(2)** discover recurring themes "
                        "without predefined labels (K-Means), and **(3)** pull up related reviews when "
                        "investigating an issue (content-based Top-K recommendation).")
        with right:
            st.markdown("**Dataset.** Amazon Reviews 2023 (McAuley Lab, UC San Diego) - real reviews with "
                        "title, star rating, date, verified-purchase flag and helpful votes; noisy text "
                        "(HTML line breaks, entities, ALL-CAPS, emoji) and real re-posted duplicates.")

    m = st.columns(6)
    m[0].metric("Documents ingested", f"{len(corpus):,}")
    m[1].metric("After de-duplication", f"{len(pr.corpus):,}", delta=f"-{len(corpus) - len(pr.corpus)} copies",
                delta_color="off")
    m[2].metric("Categories", stats["categories"])
    m[3].metric("Index terms", f"{len(pr.features.index):,}",
                help="Vocabulary after preprocessing and document-frequency pruning")
    m[4].metric("Mean tokens / doc", stats["mean tokens / doc"])
    m[5].metric("Duplicate groups", len(pr.duplicates.groups))

    if c.has_labels():
        cr, cl, rec = c.classification(), c.clustering(), c.rec_eval()
        base = chance_baseline(cl.labels_pred, pr.corpus.labels)
        st.markdown("##### Headline results (current configuration)")
        h = st.columns(5)
        h[0].metric("Naive Bayes accuracy", c.pct(cr.report["accuracy"]),
                    help=f"{cr.report['n_correct']} of {cr.report['n_test']} held-out reviews")
        h[1].metric("Naive Bayes macro-F1", c.f3(cr.report["macro"]["f1"]))
        h[2].metric("K-Means purity", c.f3(cl.purity), help=f"Random clustering of the same sizes: {base['purity']:.3f}")
        h[3].metric("K-Means Rand Index", c.f3(cl.rand_index),
                    help=f"Random clustering of the same sizes: {base['rand_index']:.3f}")
        h[4].metric(f"Recommendation P@{rec['k']}", c.f3(rec["precision_at_k"]),
                    help="Share of Top-K recommendations from the same department as the query review")
        st.caption(f"Chance levels for comparison - accuracy {1 / max(1, stats['categories']):.3f} ({stats['categories']} classes), "
                   f"purity {base['purity']:.3f}, Rand Index {base['rand_index']:.3f} (random clusterings with the "
                   f"same cluster sizes).")

    meta = corpus.metadata
    col1, col2 = st.columns(2)
    if corpus.labels:
        counts = pd.Series(corpus.labels).value_counts().rename_axis("category").reset_index(name="documents")
        counts = counts.sort_values("documents", ascending=False)
        with col1:
            fig = c.bar(counts, "category", "documents", horizontal=True, height=300,
                        title="Documents per category", text_fmt="%{x}")
            c.chart_with_table(fig, counts, "dash_cat")
    if "date" in meta.columns:
        years = pd.to_datetime(meta["date"], errors="coerce").dt.year.dropna().astype(int)
        per_year = years.value_counts().sort_index().rename_axis("year").reset_index(name="documents")
        with col2:
            fig = c.bar(per_year, "year", "documents", height=300, title="Reviews per year")
            fig.update_xaxes(dtick=2)
            c.chart_with_table(fig, per_year, "dash_year")

    col3, col4 = st.columns(2)
    if "rating" in meta.columns and corpus.labels:
        df = pd.DataFrame({"category": corpus.labels, "rating": pd.to_numeric(meta["rating"], errors="coerce")})
        df = df.dropna()
        share = (df.groupby(["category", "rating"]).size() / df.groupby("category").size()).rename("share").reset_index()
        fig = go.Figure()
        for i, r in enumerate(sorted(share["rating"].unique())):
            sub = share[share["rating"] == r]
            fig.add_trace(go.Bar(y=sub["category"], x=sub["share"], orientation="h", name=f"{int(r)}★",
                                 marker=dict(color=RATING_RAMP[min(i, 4)], line=dict(color=c.SURFACE, width=2)),
                                 hovertemplate="%{y} · " + f"{int(r)}★" + ": %{x:.0%}<extra></extra>"))
        c.style(fig, 300, "Star-rating mix per category")
        fig.update_layout(barmode="stack", bargap=0.3)
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=c.GRID)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        with col3:
            c.chart_with_table(fig, share.assign(share=share["share"].round(3)), "dash_rating")
    lengths = pd.Series([len(t.split()) for t in corpus.texts], name="tokens")
    hist, edges = np.histogram(lengths, bins=30)
    hdf = pd.DataFrame({"tokens (bin start)": edges[:-1].round(0).astype(int), "documents": hist})
    with col4:
        fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist, width=(edges[1] - edges[0]) * 0.85,
                               marker=dict(color=c.SERIES[0]), hovertemplate="~%{x:.0f} tokens: %{y} docs<extra></extra>"))
        c.style(fig, 300, "Document length (whitespace tokens)", legend=False, x_title="tokens per review",
                y_title="documents")
        c.chart_with_table(fig, hdf, "dash_len")

    st.markdown("##### Workflow")
    steps = [
        ("Data ingestion", "CSV / JSON / ZIP of .txt; text kept separate from metadata; corpus statistics"),
        ("Preprocessing", "markup & entity removal, contractions, lower-casing, stop words, stemming or "
                          "lemmatisation, n-grams - all configurable"),
        ("Features", "inverted index -> TF-IDF (log tf × idf) with cosine length normalisation, built by hand"),
        ("Duplicates", "hash of normalised text + cosine >= threshold; one copy kept per group"),
        ("Text mining", "manual Multinomial Naive Bayes, manual K-Means, manual Purity & Rand Index"),
        ("Recommendation", "Top-K related reviews by cosine similarity through the inverted index"),
        ("Performance analytics", "metrics, chance baselines, preprocessing ablation, duplicate-leakage study"),
    ]
    html = "".join(f"<div class='rl-step'><div class='rl-num'>{i}</div><div><b>{t}</b><br><small>{d}</small></div></div>"
                   for i, (t, d) in enumerate(steps, start=1))
    with c.card():
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("##### Sample of the corpus")
    show = pd.DataFrame({"doc_id": corpus.doc_ids[:200],
                         "category": (corpus.labels or [""] * len(corpus))[:200],
                         "title": corpus.titles[:200],
                         "text": [c.snippet(t, 140) for t in corpus.texts[:200]]})
    for col in ("rating", "date"):
        if col in meta.columns:
            show[col] = meta[col].tolist()[:200]
    c.table(show, height=320)
