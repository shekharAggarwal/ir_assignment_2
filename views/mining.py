"""Text Mining & Corpus Analytics hub: Naive Bayes, K-Means, Purity / Rand Index, comparison, insights."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from textmining.analytics import cluster_rating_summary, department_insights
from textmining.evaluation import chance_baseline, pair_counts, pair_counts_bruteforce
from views import common as c


# ------------------------------------------------------------------------------------------------ Naive Bayes
def _nb_tab() -> None:
    mc = st.session_state.model_cfg
    with st.expander("Classifier settings", expanded=False):
        with st.form("nb_form"):
            a, b, cc, d = st.columns(4)
            test_size = a.slider("Test share", 0.10, 0.50, float(mc.test_size), 0.05)
            alpha = b.number_input("Laplace smoothing α", 0.01, 5.0, float(mc.alpha), 0.1)
            split = cc.radio("Split", ["duplicate-aware stratified", "plain stratified random"],
                             index=0 if mc.group_split else 1)
            seed = d.number_input("Random seed", 0, 99999, int(mc.seed))
            if st.form_submit_button("Train & evaluate", type="primary"):
                c.set_model_cfg(test_size=float(test_size), alpha=float(alpha), seed=int(seed),
                                group_split=split.startswith("duplicate"))
                c.flash("Naive Bayes retrained with the new settings.")
    cr = c.classification()
    rep, labels = cr.report, cr.labels
    m = st.columns(6)
    m[0].metric("Accuracy", c.pct(rep["accuracy"]), help=f"{rep['n_correct']} / {rep['n_test']} test reviews")
    m[1].metric("Macro precision", c.f3(rep["macro"]["precision"]))
    m[2].metric("Macro recall", c.f3(rep["macro"]["recall"]))
    m[3].metric("Macro F1", c.f3(rep["macro"]["f1"]))
    m[4].metric("Weighted F1", c.f3(rep["weighted"]["f1"]))
    m[5].metric("Train / test", f"{len(cr.train_idx)} / {len(cr.test_idx)}")

    c.explain("How the manual Multinomial Naive Bayes works",
              "Trained only on the training split: class priors and term likelihoods with add-α (Laplace) smoothing "
              "over the training vocabulary. Prediction sums log-probabilities (no underflow); terms never seen in "
              "training are ignored. The split is stratified by class and, by default, duplicate-aware.",
              [r"\hat P(c) = \frac{N_c}{N} \qquad \hat P(t \mid c) = \frac{T_{ct} + \alpha}{\sum_{t'} T_{ct'} + \alpha |V|}",
               r"c_{map} = \arg\max_c \Bigl[\log \hat P(c) + \sum_{t \in d} \mathrm{tf}_{t,d} \log \hat P(t \mid c)\Bigr]"])

    a, b = st.columns(2)
    actual = pd.Series(cr.y_true).value_counts().reindex(labels, fill_value=0)
    pred = pd.Series(cr.y_pred).value_counts().reindex(labels, fill_value=0)
    dist = pd.DataFrame({"category": labels, "actual": actual.values, "predicted": pred.values})
    with a:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dist["category"], y=dist["actual"], name="actual", marker=dict(color=c.SERIES[0]),
                             hovertemplate="%{x}<br>actual: %{y}<extra></extra>"))
        fig.add_trace(go.Bar(x=dist["category"], y=dist["predicted"], name="predicted", marker=dict(color=c.SERIES[1]),
                             hovertemplate="%{x}<br>predicted: %{y}<extra></extra>"))
        c.style(fig, 360, "Actual vs predicted class distribution (test set)", y_title="test reviews")
        fig.update_layout(barmode="group")
        c.chart_with_table(fig, dist, "nb_dist")
    with b:
        cm = np.array(cr.matrix)
        row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig = go.Figure(go.Heatmap(z=row_share, x=labels, y=labels, colorscale=c.SEQ_BLUE, zmin=0, zmax=1,
                                   customdata=cm, text=cm, texttemplate="%{text}",
                                   textfont=dict(size=13),
                                   colorbar=dict(thickness=12, tickformat=".0%"),
                                   hovertemplate="actual %{y}<br>predicted %{x}<br>%{customdata} reviews "
                                                 "(%{z:.0%} of the actual class)<extra></extra>"))
        c.style(fig, 360, "Confusion matrix (counts; colour = share of actual class)", legend=False,
                x_title="predicted", y_title="actual")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels).rename_axis("actual \\ predicted").reset_index()
        c.chart_with_table(fig, cm_df, "nb_cm")

    per = pd.DataFrame(rep["per_class"])
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        for i, metric in enumerate(["precision", "recall", "f1"]):
            fig.add_trace(go.Bar(x=per["class"], y=per[metric], name=metric, marker=dict(color=c.SERIES[i]),
                                 hovertemplate="%{x}<br>" + metric + ": %{y:.3f}<extra></extra>"))
        c.style(fig, 330, "Per-class precision, recall and F1")
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1])
        c.plot(fig, key="nb_per")
    with b:
        show = per[["class", "precision", "recall", "f1", "support"]].copy()
        tot = pd.DataFrame([{"class": "macro avg", **rep["macro"], "support": rep["n_test"]},
                            {"class": "weighted avg", **rep["weighted"], "support": rep["n_test"]}])
        show = pd.concat([show, tot], ignore_index=True)
        c.table(show.round(3), height=330)

    st.markdown("**Actual and predicted labels for every test document**")
    corpus = c.pipeline().corpus
    meta = corpus.metadata
    pred_df = pd.DataFrame({"doc_id": [corpus.doc_ids[i] for i in cr.test_idx],
                            "title": [corpus.titles[i] for i in cr.test_idx],
                            "actual": cr.y_true, "predicted": cr.y_pred,
                            "confidence": cr.confidence,
                            "correct": ["✓" if a == p else "✗" for a, p in zip(cr.y_true, cr.y_pred)],
                            "text": [c.snippet(corpus.texts[i], 120) for i in cr.test_idx]})
    if "rating" in meta.columns:
        pred_df.insert(2, "rating", [meta.iloc[i]["rating"] for i in cr.test_idx])
    only_err = st.checkbox("Show misclassified documents only", False)
    view = pred_df[pred_df["correct"] == "✗"] if only_err else pred_df
    c.table(view, height=360, column_config={"confidence": c.progress_column(max_value=1.0)})
    st.download_button("Download predictions (CSV)", pred_df.to_csv(index=False).encode("utf-8"), "nb_predictions.csv",
                       "text/csv")

    st.markdown("**Most discriminative terms per class** (log P(t|c) minus the mean over the other classes)")
    cols = st.columns(3)
    for i, cls in enumerate(labels):
        terms = cr.model.top_terms(cls, 10)
        with cols[i % 3]:
            st.markdown(f"**{cls}**")
            st.markdown(" ".join(f"<span class='rl-pill'>{t}</span>" for t, _ in terms), unsafe_allow_html=True)

    st.markdown("**Try it: route a new review**")
    txt = st.text_area("Review text", "The fan is loud and the water filter had to be replaced after two months.",
                       height=80, key="nb_try")
    if txt.strip():
        from textmining.preprocessing import Preprocessor
        terms = Preprocessor(st.session_state.prep_cfg)(txt)
        proba = cr.model.predict_proba_one(terms)
        pdf = pd.DataFrame({"category": list(proba), "probability": list(proba.values())}).sort_values("probability",
                                                                                                        ascending=False)
        a, b = st.columns([1, 2])
        a.metric("Predicted department", pdf.iloc[0]["category"], delta=f"{pdf.iloc[0]['probability']:.1%} posterior",
                 delta_color="off")
        with b:
            fig = c.bar(pdf, "category", "probability", horizontal=True, height=240, text_fmt="%{x:.1%}")
            fig.update_xaxes(tickformat=".0%", range=[0, 1.1])
            c.plot(fig, key="nb_try_chart")


# ------------------------------------------------------------------------------------------------ K-Means
def _kmeans_settings() -> None:
    mc = st.session_state.model_cfg
    corpus = c.corpus()
    n_cls = len(set(corpus.labels or [])) or 3
    with st.expander("Clustering settings", expanded=False):
        with st.form("km_form"):
            a, b, cc, d = st.columns(4)
            k = a.number_input("Number of clusters k", 2, 20, min(20, max(2, int(mc.k or n_cls))))
            init = b.radio("Centroid initialisation", ["k-means++", "random"], index=0 if mc.init == "k-means++" else 1)
            metric = cc.radio("Distance", ["euclidean", "cosine"], index=0 if mc.metric == "euclidean" else 1,
                              format_func={"euclidean": "Euclidean (standard K-Means)",
                                           "cosine": "cosine (spherical K-Means)"}.get)
            proj = d.radio("2-D view (visualisation only)", ["documents", "centroids"],
                           index=0 if st.session_state.projection == "documents" else 1,
                           format_func={"documents": "PCA of documents", "centroids": "PCA of centroids"}.get)
            e, f, g, h = st.columns(4)
            n_init = e.number_input("Restarts (best RSS kept)", 1, 30, int(mc.n_init))
            max_iter = f.number_input("Max iterations", 5, 500, int(mc.max_iter))
            tol = g.number_input("Tolerance (centroid shift)", 0.0, 0.1, float(mc.tol), 0.0001, format="%.4f")
            seed = h.number_input("Seed", 0, 99999, int(mc.seed), key="km_seed")
            if st.form_submit_button("Run K-Means", type="primary"):
                c.set_model_cfg(k=0 if int(k) == n_cls else int(k), init=init, metric=metric, n_init=int(n_init),
                                max_iter=int(max_iter), tol=float(tol), seed=int(seed))
                st.session_state.projection = proj
                c.flash("K-Means re-run with the new settings.")


def _scatter(cl, corpus, color_by: str, highlight) -> go.Figure:
    labels = corpus.labels or ["?"] * len(corpus)
    xy = cl.coords
    fig = go.Figure()
    hover = "%{customdata[0]}<br>%{customdata[1]}<br>category: %{customdata[2]}<br>cluster %{customdata[3]}<extra></extra>"
    custom = np.array([[corpus.doc_ids[i], c.snippet(corpus.titles[i], 50), labels[i], cl.labels_pred[i]]
                       for i in range(len(corpus))], dtype=object)
    if color_by == "cluster":
        groups = [(f"Cluster {k}", np.array(cl.labels_pred) == k, c.cluster_color(k), c.SYMBOLS[k % len(c.SYMBOLS)])
                  for k in range(cl.k)]
    else:
        cmap = c.color_map(labels)
        lab = np.array(labels)
        groups = [(name, lab == name, col, c.SYMBOLS[i % len(c.SYMBOLS)]) for i, (name, col) in enumerate(cmap.items())]
    for gi, (name, mask, color, symbol) in enumerate(groups):
        faded = highlight is not None and gi != highlight
        fig.add_trace(go.Scatter(x=xy[mask, 0], y=xy[mask, 1], mode="markers", name=name, customdata=custom[mask],
                                 hovertemplate=hover,
                                 marker=dict(size=8, symbol=symbol, color="#d9d8d2" if faded else color,
                                             opacity=0.35 if faded else 0.85,
                                             line=dict(color=c.SURFACE, width=1))))
    if color_by == "cluster":
        cxy = cl.centroid_coords
        fig.add_trace(go.Scatter(x=cxy[:, 0], y=cxy[:, 1], mode="markers", name="centroids",
                                 customdata=[f"C{k}" for k in range(cl.k)],
                                 marker=dict(size=16, symbol="x-thin", color=c.INK, line=dict(width=3, color=c.INK)),
                                 hovertemplate="centroid %{customdata}<extra></extra>"))
        # direct labels fanned out around the centroids so that close centroids stay readable
        for k in range(cl.k):
            angle = 2 * np.pi * k / max(1, cl.k)
            fig.add_annotation(x=cxy[k, 0], y=cxy[k, 1], text=f"<b>C{k}</b>", showarrow=True, arrowhead=0,
                               arrowwidth=1, arrowcolor=c.INK_2, ax=48 * np.cos(angle), ay=-48 * np.sin(angle),
                               bgcolor="rgba(255,255,255,0.9)", bordercolor=c.cluster_color(k), borderwidth=2,
                               borderpad=3, font=dict(color=c.INK, size=12))
    c.style(fig, 520, f"Documents in 2-D ({'PCA of documents' if st.session_state.projection == 'documents' else 'PCA of centroids'}; "
                      f"variance shown {100 * (cl.explained[0] + cl.explained[1]):.1f}%)",
            x_title="component 1", y_title="component 2")
    fig.update_xaxes(showgrid=True, gridcolor=c.GRID, zeroline=False)
    return fig


def _kmeans_tab() -> None:
    _kmeans_settings()
    cl = c.clustering()
    pr = c.pipeline()
    corpus = pr.corpus
    best = cl.result.best
    labels = corpus.labels or ["?"] * len(corpus)
    m = st.columns(5)
    m[0].metric("Clusters k", cl.k)
    m[1].metric("Iterations (best run)", best.n_iter)
    m[2].metric("RSS (best run)", f"{best.rss:.2f}")
    m[3].metric("Purity", c.f3(cl.purity))
    m[4].metric("Rand Index", c.f3(cl.rand_index))
    c.explain("How the manual K-Means works",
              "1. **Initialisation** - k seed documents (k-means++: each next seed drawn with probability "
              "proportional to its squared distance from the nearest seed).\n"
              "2. **Assignment** - each document joins the nearest centroid.\n"
              "3. **Update** - each centroid moves to the mean of its documents (re-normalised for cosine).\n"
              "4. **Stop** - when no document changes cluster, the centroid shift is below the tolerance, or "
              "max iterations is reached. The objective RSS never increases. Several restarts are run and the "
              "lowest-RSS run is kept.",
              [r"\mathrm{RSS} = \sum_{k=1}^{K} \sum_{\vec x \in \omega_k} \lVert \vec x - \vec\mu_k \rVert^2 \qquad "
               r"\vec\mu_k = \frac{1}{|\omega_k|} \sum_{\vec x \in \omega_k} \vec x"])

    st.markdown("##### Step 1 · Centroid initialisation")
    a, b = st.columns([3, 2])
    with a:
        seeds = pd.DataFrame([{"cluster": k, "seed doc_id": corpus.doc_ids[i], "category": labels[i],
                               "title": corpus.titles[i]} for k, i in enumerate(best.init_indices)])
        st.caption(f"Seed documents of the best run (seed {best.seed}, initialisation: {st.session_state.model_cfg.init})")
        c.table(seeds)
    with b:
        runs = pd.DataFrame([{"restart seed": r.seed, "RSS": round(r.rss, 3), "iterations": r.n_iter,
                              "best": "✓" if r is best else ""} for r in cl.result.runs])
        st.caption("All restarts - the lowest RSS is kept")
        c.table(runs, height=250)

    st.markdown("##### Steps 2-3 · Assignment and centroid update, iteration by iteration")
    hist = pd.DataFrame(best.history)
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=hist["iteration"], y=hist["RSS"], mode="lines+markers",
                                   line=dict(color=c.SERIES[0], width=2),
                                   marker=dict(size=8, line=dict(color=c.SURFACE, width=2)),
                                   hovertemplate="iteration %{x}: RSS %{y:.3f}<extra></extra>"))
        c.style(fig, 280, "RSS after each iteration (never increases)", legend=False, x_title="iteration", y_title="RSS")
        c.plot(fig, key="km_rss")
    with b:
        fig = c.bar(hist, "iteration", "documents reassigned", height=280, title="Documents that changed cluster",
                    x_title="iteration", y_title="documents")
        c.plot(fig, key="km_moves")
    c.table(hist.round(5), height=240)
    st.markdown(f"##### Step 4 · Stopping criterion\n<span class='rl-pill'>{best.stop_reason}</span>",
                unsafe_allow_html=True)

    st.markdown("##### Result · clusters in two dimensions")
    a, b = st.columns([1, 1])
    color_by = a.radio("Colour by", ["cluster", "true category"], horizontal=True, key="km_color")
    n_groups = cl.k if color_by == "cluster" else len(set(labels))
    opts = ["(none)"] + list(range(n_groups))
    names = ([f"Cluster {k}" for k in range(cl.k)] if color_by == "cluster" else sorted(set(labels)))
    hl = b.selectbox("Highlight one group (others greyed out)", opts,
                     format_func=lambda o: o if o == "(none)" else names[o], key="km_hl")
    fig = _scatter(cl, corpus, "cluster" if color_by == "cluster" else "label", None if hl == "(none)" else hl)
    pts = pd.DataFrame({"doc_id": corpus.doc_ids, "x": cl.coords[:, 0].round(4), "y": cl.coords[:, 1].round(4),
                        "cluster": cl.labels_pred, "true label": labels})
    c.chart_with_table(fig, pts, "km_scatter")
    st.caption("The projection is used only for this picture; K-Means itself runs on all TF-IDF dimensions, so "
               "clusters that overlap here can still be separated in the full space. Marker shapes repeat the colour "
               "coding for colour-blind readers.")

    sizes = pd.Series(cl.labels_pred).value_counts().reindex(range(cl.k), fill_value=0)
    size_df = pd.DataFrame({"cluster": [f"C{k}" for k in range(cl.k)], "documents": sizes.values})
    a, b = st.columns(2)
    with a:
        fig = c.bar(size_df, "cluster", "documents", height=330, title="Cluster distribution (documents per cluster)",
                    text_fmt="%{y}")
        c.chart_with_table(fig, size_df, "km_sizes")
    with b:
        cont = pd.crosstab(pd.Series(cl.labels_pred, name="cluster"), pd.Series(labels, name="category"))
        cont = cont.reindex(range(cl.k), fill_value=0)
        cmap = c.color_map(labels)
        fig = go.Figure()
        for cat in cont.columns:
            fig.add_trace(go.Bar(x=[f"C{k}" for k in cont.index], y=cont[cat], name=cat,
                                 marker=dict(color=cmap[cat], line=dict(color=c.SURFACE, width=2)),
                                 hovertemplate="%{x} · " + cat + ": %{y}<extra></extra>"))
        c.style(fig, 380, "Cluster composition by true category", legend_bottom=True)
        fig.update_layout(barmode="stack")
        c.chart_with_table(fig, cont.reset_index(), "km_comp")

    st.markdown("**Cluster membership**")
    mem = pd.DataFrame({"doc_id": corpus.doc_ids, "title": corpus.titles, "true label": labels,
                        "cluster": cl.labels_pred})
    pick = st.multiselect("Filter clusters", list(range(cl.k)), default=[], key="km_filter")
    c.table(mem[mem["cluster"].isin(pick)] if pick else mem, height=320)
    st.download_button("Download cluster assignments (CSV)", mem.to_csv(index=False).encode("utf-8"),
                       "cluster_assignments.csv", "text/csv")

    st.markdown("**What each cluster is about** (highest-weighted centroid terms)")
    summ = pd.DataFrame([{"cluster": f"C{k}", "size": size, "majority category": lab,
                          "majority share": cnt / size if size else 0.0,
                          "top terms": ", ".join(t for t, _ in cl.top_terms[k][:10])}
                         for k, (lab, cnt, size) in sorted(cl.majority.items())])
    c.table(summ, column_config={"majority share": c.progress_column(max_value=1.0),
                                 "top terms": st.column_config.TextColumn(width="large")})


# ------------------------------------------------------------------------------------------------ evaluation
def _eval_tab() -> None:
    cl, cr, pr = c.clustering(), c.classification(), c.pipeline()
    labels = pr.corpus.labels
    pc = cl.pair_counts
    base = chance_baseline(cl.labels_pred, labels)
    c.explain("Purity and Rand Index (manual implementations)",
              "**Purity** credits each cluster with its most frequent true category. **Rand Index** looks at all "
              "N(N-1)/2 document pairs and counts how often clustering and labels agree (same/same or "
              "different/different). The chance baseline shuffles the cluster ids across documents (same cluster "
              "sizes) 20 times.",
              [r"\mathrm{purity}(\Omega, C) = \frac{1}{N}\sum_k \max_j |\omega_k \cap c_j|",
               r"\mathrm{RI} = \frac{TP + TN}{TP + FP + FN + TN}"], expanded=True)
    m = st.columns(6)
    m[0].metric("Purity", c.f3(cl.purity))
    m[1].metric("Purity by chance", c.f3(base["purity"]))
    m[2].metric("Rand Index", c.f3(cl.rand_index))
    m[3].metric("Rand Index by chance", c.f3(base["rand_index"]))
    m[4].metric("Document pairs", f"{sum(pc.values()):,}")
    m[5].metric("Pairs in agreement", f"{pc['TP'] + pc['TN']:,}", help="TP + TN")

    a, b = st.columns([2, 3])
    with a:
        st.markdown("**Pair confusion matrix**")
        pcm = pd.DataFrame({"": ["same cluster", "different cluster"], "same category": [pc["TP"], pc["FN"]],
                            "different category": [pc["FP"], pc["TN"]]})
        c.table(pcm)
        st.caption("TP = same category & same cluster, FP = different category but same cluster, "
                   "FN = same category but different clusters, TN = different & different.")
        if c.wide_button("Verify: recount the Rand Index pair by pair", key="ri_verify"):
            t = time.perf_counter()
            brute = pair_counts_bruteforce(cl.labels_pred, labels)
            secs = time.perf_counter() - t
            same = brute == pair_counts(cl.labels_pred, labels)
            st.success(f"Brute-force pair loop: {brute} in {secs:.2f}s - identical to the contingency-table "
                       f"method: {same}")
    with b:
        cont = pd.crosstab(pd.Series(cl.labels_pred, name="cluster"), pd.Series(labels, name="category"))
        z = cont.values
        fig = go.Figure(go.Heatmap(z=z / np.maximum(z.sum(axis=1, keepdims=True), 1), x=list(cont.columns),
                                   y=[f"C{k}" for k in cont.index], customdata=z, text=z, texttemplate="%{text}",
                                   colorscale=c.SEQ_BLUE, zmin=0, zmax=1,
                                   colorbar=dict(tickformat=".0%", thickness=12),
                                   hovertemplate="%{y} · %{x}: %{customdata} docs (%{z:.0%} of cluster)<extra></extra>"))
        c.style(fig, 330, "Contingency table: clusters × true categories", legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        c.chart_with_table(fig, cont.reset_index(), "eval_cont")

    st.markdown("##### Naive Bayes (supervised) vs K-Means (unsupervised)")
    rep = cr.report
    n_cls = len(set(labels))
    majority_share = pd.Series(cr.y_true).value_counts(normalize=True).max()
    comp = pd.DataFrame([
        {"method": "Naive Bayes", "metric": "accuracy", "value": rep["accuracy"], "chance level": majority_share},
        {"method": "Naive Bayes", "metric": "macro precision", "value": rep["macro"]["precision"], "chance level": 1 / n_cls},
        {"method": "Naive Bayes", "metric": "macro recall", "value": rep["macro"]["recall"], "chance level": 1 / n_cls},
        {"method": "Naive Bayes", "metric": "macro F1", "value": rep["macro"]["f1"], "chance level": 1 / n_cls},
        {"method": "K-Means", "metric": "purity", "value": cl.purity, "chance level": base["purity"]},
        {"method": "K-Means", "metric": "Rand Index", "value": cl.rand_index, "chance level": base["rand_index"]},
    ])
    a, b = st.columns([3, 2])
    with a:
        names = [f"{r.method} · {r.metric}" for r in comp.itertuples()]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=comp["value"], orientation="h", name="model", marker=dict(color=c.SERIES[0]),
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.add_trace(go.Bar(y=names, x=comp["chance level"], orientation="h", name="chance level",
                             marker=dict(color=c.OTHER), hovertemplate="%{y} chance: %{x:.3f}<extra></extra>"))
        c.style(fig, 360, "Model scores vs chance level")
        fig.update_layout(barmode="group")
        fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=c.GRID)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        c.plot(fig, key="eval_comp")
    with b:
        c.table(comp.round(3), height=260)
    lift_nb = rep["accuracy"] / majority_share if majority_share else 0
    lift_p = cl.purity / base["purity"] if base["purity"] else 0
    mixed = sorted(cl.majority.items(), key=lambda kv: kv[1][1] / max(1, kv[1][2]))[:2]
    themes = " and ".join(f'*"{" / ".join(t for t, _ in cl.top_terms[k][:3])}"* (C{k})' for k, _ in mixed)
    st.markdown(
        f"""
**Interpretation.** Naive Bayes learns from {len(cr.train_idx)} labelled reviews and routes **{c.pct(rep['accuracy'])}**
of unseen reviews to the right department ({lift_nb:.1f}× the majority-class rate, macro-F1 {rep['macro']['f1']:.3f}).
K-Means never sees a label: its {cl.k} clusters are dominated by one department for **{c.pct(cl.purity)}** of the
documents (purity; {lift_p:.1f}× the chance level of {c.pct(base['purity'])}). The Rand Index of
**{cl.rand_index:.3f}** looks high, but a random clustering of the same sizes already scores
**{base['rand_index']:.3f}**, because most of the {sum(pc.values()):,} pairs are "different category, different
cluster" - purity and the chance baseline are more informative here.

* **Supervised vs unsupervised** - Naive Bayes optimises exactly the target (the department), so it can use
  department-specific words (*dog, guitar, filter, game*). K-Means optimises compactness (RSS); it groups reviews by
  whatever dominates the vocabulary - its most mixed clusters are {themes}, themes that cut across departments
  (the Software department also contains many app-store games).
* **Metrics are not interchangeable** - accuracy / F1 compare predictions with labels one-to-one; purity and
  Rand Index have to cope with clusters that have no names, so they reward any grouping that keeps categories
  together. Purity reaches 1 when every document is its own cluster (k = N), so it is only comparable at a fixed k;
  here k equals the number of categories.
* **Use together** - clusters reveal themes analysts did not define in advance; the classifier operationalises
  the categories they already care about.
""")


# ------------------------------------------------------------------------------------------------ insights
def _insights_tab() -> None:
    pr = c.pipeline()
    corpus, terms = pr.corpus, pr.features.terms
    labels = corpus.labels
    meta = corpus.metadata
    st.caption("Patterns for domain analysts - which words characterise each department, and what drives "
               "negative (1-2★) reviews inside each department.")
    rating = pd.to_numeric(meta["rating"], errors="coerce").fillna(3).tolist() if "rating" in meta.columns else None
    df = pd.DataFrame(department_insights(terms, labels, rating))
    st.markdown("**Vocabulary that distinguishes each department** (smoothed log-odds vs all other departments)")
    c.table(df[["category", "documents", "distinctive terms"]],
            column_config={"distinctive terms": st.column_config.TextColumn(width="large")})
    if rating is not None:
        st.markdown("**What 1-2★ reviews talk about in each department** (log-odds vs 4-5★ reviews of the same "
                    "department)")
        c.table(df[["category", "negative reviews", "share negative", "what negative reviews talk about"]],
                column_config={"share negative": c.progress_column(max_value=1.0, fmt="%.2f"),
                               "what negative reviews talk about": st.column_config.TextColumn(width="large")})
        cl = c.clustering()
        cdf = pd.DataFrame(cluster_rating_summary(cl.labels_pred, rating, cl.majority, cl.top_terms))
        st.markdown("**Themes found by K-Means, ranked by dissatisfaction**")
        c.table(cdf, column_config={"share 1-2★": c.progress_column(max_value=1.0, fmt="%.2f"),
                                    "mean rating": c.number_column("%.2f"),
                                    "top terms": st.column_config.TextColumn(width="large")})
        fig = c.bar(cdf.sort_values("share 1-2★", ascending=False), "cluster", "share 1-2★", height=280,
                    title="Share of 1-2★ reviews per cluster", text_fmt="%{y:.0%}", hover_fmt="%{y:.1%}")
        fig.update_yaxes(tickformat=".0%")
        c.plot(fig, key="ins_neg")


def render() -> None:
    c.page_header("Text Mining & Corpus Analytics hub",
                  "Manual Naive Bayes classification, manual K-Means clustering, Purity / Rand Index evaluation and "
                  "analyst insights.")
    c.show_flash()
    if not c.has_labels():
        st.warning("The active corpus has no category labels. Classification and the external clustering metrics "
                   "need a label column - load the bundled sample or map a label column on the ingestion page.")
        return
    tabs = st.tabs(["Naive Bayes classification", "K-Means clustering", "Evaluation & comparison", "Corpus insights"])
    with tabs[0]:
        _nb_tab()
    with tabs[1]:
        _kmeans_tab()
    with tabs[2]:
        _eval_tab()
    with tabs[3]:
        _insights_tab()
