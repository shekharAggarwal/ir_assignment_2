"""Document recommendation panel (content-based Top-K)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from textmining.pipeline import recommender_for
from textmining.preprocessing import Preprocessor
from textmining.recommendation import evaluate_precision_at_k, precision_at_k_by_label
from views import common as c


def render() -> None:
    c.page_header("Document recommendation panel",
                  "Pick a review (or type one) and retrieve the Top-K most related reviews with their similarity "
                  "scores and metadata.")
    pr = c.pipeline()
    corpus, model = pr.corpus, pr.features.model
    labels = corpus.labels or ["-"] * len(corpus)
    meta = corpus.metadata
    rec = recommender_for(pr)
    dedup = st.session_state.dedup_cfg

    with c.card():
        mode = st.radio("Query", ["Select an existing review", "Type a new review / free-text query"], horizontal=True)
        query_idx, query_vec, query_label = None, None, None
        if mode.startswith("Select"):
            a, b = st.columns([1, 3])
            cats = ["All categories"] + sorted(set(labels))
            cat = a.selectbox("Filter by category", cats, key="rec_cat")
            pool = [i for i in range(len(corpus)) if cat == "All categories" or labels[i] == cat]
            query_idx = b.selectbox("Review", pool, index=0, key="rec_doc",
                                    format_func=lambda i: f"{corpus.doc_ids[i]} · {labels[i]} · {c.snippet(corpus.titles[i], 70)}")
            query_vec = model.doc_vectors[query_idx]
            query_label = labels[query_idx]
        else:
            text = st.text_area("Review text or analyst query",
                                "The leash broke after a week and my dog got loose, the stitching is poor quality.",
                                height=90, key="rec_text")
            query_vec = model.vectorize(Preprocessor(st.session_state.prep_cfg)(text))
            if not query_vec:
                st.warning("None of the query terms are in the index - try different words.")
        o1, o2, o3, o4 = st.columns(4)
        k = o1.slider("Top-K", 1, 20, int(st.session_state.model_cfg.top_k), key="rec_k")
        collapse = o2.checkbox("Collapse near-duplicates", True, key="rec_collapse",
                               help=f"Skip candidates with cosine ≥ {dedup.threshold:.2f} to the query or to an "
                                    f"earlier recommendation")
        same_cat = o3.checkbox("Same category only", False, key="rec_same", disabled=query_label is None)
        min_score = o4.slider("Minimum score", 0.0, 0.5, 0.0, 0.01, key="rec_min")

    if query_idx is not None:
        q1, q2 = st.columns([3, 2])
        with q1:
            st.markdown(f"**Query review {corpus.doc_ids[query_idx]}** - {corpus.titles[query_idx]}")
            c.text_block(c.snippet(corpus.texts[query_idx], 700), mono=False)
        with q2:
            info = {"category": labels[query_idx]}
            for col in ("rating", "date", "verified_purchase", "helpful_votes", "parent_asin"):
                if col in meta.columns:
                    info[col] = meta.iloc[query_idx][col]
            c.table(pd.DataFrame({"field": list(info), "value": [str(v) for v in info.values()]}))
            st.markdown("Top query terms: " + " ".join(f"<span class='rl-pill'>{t} {w:.2f}</span>"
                                                        for t, w in model.top_terms(query_idx, 8)),
                        unsafe_allow_html=True)

    if not query_vec:
        return
    results = rec.recommend(query_vec, k=k, exclude=query_idx, collapse_duplicates=collapse,
                            dup_threshold=dedup.threshold, same_label_as=query_label if same_cat else None,
                            min_score=min_score)
    if not results:
        st.info("No document shares a term with the query above the minimum score.")
        return
    rows = []
    for r in results:
        d = r["doc_idx"]
        row = {"rank": r["rank"], "doc_id": corpus.doc_ids[d], "score": r["score"], "category": labels[d],
               "title": corpus.titles[d],
               "shared terms (contribution)": ", ".join(f"{t} {v:.3f}" for t, v in r["shared_terms"][:4])}
        for col in ("rating", "date", "verified_purchase", "helpful_votes"):
            if col in meta.columns:
                row[col] = meta.iloc[d][col]
        row["text"] = c.snippet(corpus.texts[d], 160)
        rows.append(row)
    res = pd.DataFrame(rows)
    st.markdown(f"##### Top-{len(res)} recommendations")
    if results[0].get("skipped_duplicates"):
        st.caption(f"{results[0]['skipped_duplicates']} near-duplicate candidate(s) were collapsed.")
    if query_label is not None:
        agree = float(np.mean(res["category"] == query_label))
        st.markdown(f"<span class='rl-note'>Same category as the query: <b>{agree:.0%}</b> of the list.</span>",
                    unsafe_allow_html=True)
    c.table(res, column_config={"score": c.progress_column("cosine score", 1.0, "%.3f"),
                                "text": st.column_config.TextColumn(width="large")})
    a, b = st.columns([2, 3])
    with a:
        chart = res.assign(label=[f"#{r} {d}" for r, d in zip(res["rank"], res["doc_id"])])
        fig = c.bar(chart, "label", "score", horizontal=True, height=max(220, 40 * len(res) + 60),
                    title="Similarity scores", text_fmt="%{x:.3f}")
        fig.update_xaxes(range=[0, min(1.0, float(res["score"].max()) * 1.25 + 0.05)])
        c.plot(fig, key="rec_scores")
    with b:
        for r in results[:5]:
            d = r["doc_idx"]
            with st.expander(f"#{r['rank']} {corpus.doc_ids[d]} · {labels[d]} · score {r['score']:.3f}"):
                st.markdown(f"**{corpus.titles[d]}**")
                st.write(c.snippet(corpus.texts[d], 900))
    st.download_button("Download these recommendations (CSV)", res.to_csv(index=False).encode("utf-8"),
                       "recommendations.csv", "text/csv")

    st.markdown("##### How good are the recommendations? (no user feedback available)")
    if corpus.labels:
        ev_on = evaluate_precision_at_k(pr.features.sim, labels, k=k, collapse_duplicates=True, dup_threshold=dedup.threshold)
        ev_off = evaluate_precision_at_k(pr.features.sim, labels, k=k, collapse_duplicates=False, dup_threshold=dedup.threshold)
        m = st.columns(4)
        m[0].metric(f"P@{k} (collapse on)", c.f3(ev_on["precision_at_k"]),
                    help="Leave-one-out: every review is used once as the query")
        m[1].metric("MRR (first same-category hit)", c.f3(ev_on["mrr"]))
        m[2].metric(f"P@{k} (collapse off)", c.f3(ev_off["precision_at_k"]))
        m[3].metric(f"Near-copies in Top-{k} (collapse off)", c.pct(ev_off["duplicate_rate"], 2))
        per = pd.DataFrame(precision_at_k_by_label(pr.features.sim, labels, k, True, dedup.threshold))
        fig = c.bar(per, "category", f"P@{k}", height=300, title=f"P@{k} by category of the query", text_fmt="%{y:.2f}")
        fig.update_yaxes(range=[0, 1])
        c.chart_with_table(fig, per.round(3), "rec_percat")

    c.explain("How the recommender works and its limitations", f"""
**Method - content-based filtering.** The query review (or typed text) goes through the *same* preprocessing as the
corpus and is weighted with the corpus idf into a unit-length TF-IDF vector. Scores are cosine similarities accumulated
term-at-a-time through the inverted index - only reviews that share at least one term with the query are touched -
and the Top-K are returned. Near-duplicates of the query or of an earlier recommendation (cosine ≥ {dedup.threshold:.2f})
are skipped so the list is not filled with copies. The "shared terms" column shows w(t,q)·w(t,d), i.e. why a review was
recommended.

**Why content-based here.** The corpus has rich text but almost no interaction data: each anonymous reviewer
appears once or twice, so a user-document matrix would be too sparse for collaborative filtering.

**Limitations on this dataset**
* *Vocabulary mismatch* - "leash" vs "lead", "broke" vs "snapped": synonyms do not match (no semantic embeddings).
* *Short, generic reviews* ("works great, love it") share only sentiment words and get vague neighbours.
* *Cross-department themes* - "game / app / fun" links Software and Video Games; "battery / charge" spans departments.
  That is useful for issue investigation but lowers category agreement.
* *No personalisation or quality signal* - helpful votes, recency and ratings are shown but not used for ranking.
* *Evaluation proxy* - P@K uses department agreement as relevance; true relevance would need analyst judgements.
""", expanded=False)
