"""Performance analytics: scorecard, timings and the experiments behind the report's inference section."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from textmining.evaluation import chance_baseline
from textmining.experiments import ablation_study, duplicate_study, k_sensitivity, kmeans_variants
from views import common as c


def _compute(name: str) -> pd.DataFrame:
    prep, feat, dedup, mc = c.configs()
    if name in ("ablation", "duplicates"):
        bar = st.progress(0.0, text="Starting ...")
        update = lambda f, t: bar.progress(min(1.0, f), text=t)  # noqa: E731
        if name == "ablation":
            rows = ablation_study(c.corpus(), feat, dedup, mc, progress=update)
        else:
            rows = duplicate_study(c.corpus(), prep, feat, dedup, mc, progress=update)
        bar.empty()
        return pd.DataFrame(rows)
    with st.spinner("Running - this takes a few seconds ..."):
        if name == "kmeans":
            return pd.DataFrame(kmeans_variants(c.pipeline(), mc))
        return pd.DataFrame(k_sensitivity(c.pipeline(), mc))


def _run(name: str, label: str):
    """Experiments run on demand; results stay in the session until the configuration changes."""
    store = st.session_state.experiments
    if c.wide_button(label, key=f"run_{name}", primary=name not in store):
        store[name] = _compute(name)
    return store.get(name)


def _scorecard() -> None:
    pr, cr, cl, rec = c.pipeline(), c.classification(), c.clustering(), c.rec_eval()
    base = chance_baseline(cl.labels_pred, pr.corpus.labels)
    rep = cr.report
    m = st.columns(8)
    m[0].metric("NB accuracy", c.f3(rep["accuracy"]))
    m[1].metric("NB macro-P", c.f3(rep["macro"]["precision"]))
    m[2].metric("NB macro-R", c.f3(rep["macro"]["recall"]))
    m[3].metric("NB macro-F1", c.f3(rep["macro"]["f1"]))
    m[4].metric("K-Means purity", c.f3(cl.purity))
    m[5].metric("K-Means RI", c.f3(cl.rand_index))
    m[6].metric(f"Rec P@{rec['k']}", c.f3(rec["precision_at_k"]))
    m[7].metric("Rec MRR", c.f3(rec["mrr"]))
    st.caption(f"Chance levels: purity {base['purity']:.3f} and Rand Index {base['rand_index']:.3f} for random "
               f"clusterings with the same cluster sizes; accuracy 0.167 for random guessing among 6 balanced classes.")
    timings = dict(pr.timings)
    timings["Naive Bayes train + predict"] = cr.seconds
    timings["K-Means (all restarts) + projection"] = cl.seconds
    tdf = pd.DataFrame({"stage": list(timings), "seconds": [round(v, 3) for v in timings.values()]})
    a, b = st.columns([2, 3])
    with a:
        fig = c.bar(tdf, "stage", "seconds", horizontal=True, height=280, title="Run time per stage (seconds)",
                    text_fmt="%{x:.2f}s")
        c.plot(fig, key="perf_time")
    with b:
        summary = pd.DataFrame([
            {"task": "classification", "method": "Multinomial Naive Bayes (manual)", "metric": "accuracy / macro-F1",
             "value": f"{rep['accuracy']:.3f} / {rep['macro']['f1']:.3f}"},
            {"task": "clustering", "method": f"K-Means k={cl.k} (manual)", "metric": "purity / Rand Index",
             "value": f"{cl.purity:.3f} / {cl.rand_index:.3f}"},
            {"task": "clustering (chance)", "method": "shuffled cluster ids", "metric": "purity / Rand Index",
             "value": f"{base['purity']:.3f} / {base['rand_index']:.3f}"},
            {"task": "recommendation", "method": "content-based cosine Top-K", "metric": f"P@{rec['k']} / MRR",
             "value": f"{rec['precision_at_k']:.3f} / {rec['mrr']:.3f}"},
        ])
        c.table(summary)


def _ablation_section() -> None:
    st.markdown("##### Q1 · Effect of preprocessing on vocabulary, classification, clustering and recommendation")
    st.caption("Every variant runs on the same de-duplicated documents. Naive Bayes: mean of 10 random splits; "
               "K-Means: mean of 10 seeds; recommendation: leave-one-out P@K.")
    df = _run("ablation", "Run preprocessing ablation (7 pipelines, about 30-70 s)")
    if df is None:
        return
    short = df["configuration"].str.split(" ").str[0]
    fig = make_subplots(rows=1, cols=5, subplot_titles=["Vocabulary (all terms)", "NB accuracy", "K-Means purity",
                                                        "K-Means Rand Index", "Recommendation P@5"],
                        horizontal_spacing=0.05)
    cols = ["vocabulary (all terms)", "NB accuracy", "K-Means purity", "K-Means Rand Index", "Rec P@5"]
    sds = [None, "NB accuracy sd", "K-Means purity sd", "K-Means RI sd", None]
    for i, (col, sd) in enumerate(zip(cols, sds), start=1):
        fig.add_trace(go.Bar(x=short, y=df[col], marker=dict(color=c.SERIES[0]),
                             error_y=dict(type="data", array=df[sd], color=c.INK_2, thickness=1) if sd else None,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + col + "</extra>"), row=1, col=i)
    c.style(fig, 320, None, legend=False)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=c.INK)
    c.plot(fig, key="abl_chart")
    show = df[["configuration", "vocabulary (all terms)", "vocabulary (after df pruning)", "mean terms / doc",
               "NB accuracy", "NB macro-F1", "K-Means purity", "K-Means Rand Index", "Rec P@5", "Rec MRR"]]
    c.table(show.round(3))
    a0, a3 = df.iloc[0], df.iloc[min(3, len(df) - 1)]
    ng = df[df["configuration"].str.contains("gram")]
    sd = float(df["K-Means purity sd"].mean())
    note = (f"Vocabulary: {int(a0['vocabulary (all terms)']):,} terms (A0) → {int(a3['vocabulary (all terms)']):,} (A3). "
            f"Naive Bayes accuracy {a0['NB accuracy']:.3f} → {a3['NB accuracy']:.3f} and P@5 {a0['Rec P@5']:.3f} → "
            f"{a3['Rec P@5']:.3f}. K-Means purity ranges {df['K-Means purity'].min():.3f}-{df['K-Means purity'].max():.3f} "
            f"across pipelines while its seed-to-seed sd is about {sd:.3f}, so most clustering differences are within "
            f"noise.")
    if len(ng):
        note += (f" N-grams multiply the vocabulary (up to {int(ng['vocabulary (all terms)'].max()):,} terms) and lower "
                 f"NB accuracy (≤ {ng['NB accuracy'].max():.3f}) and P@5 (≤ {ng['Rec P@5'].max():.3f}): short reviews "
                 f"rarely share the same word pairs.")
    st.markdown(f"<span class='rl-note'>{note}</span>", unsafe_allow_html=True)
    st.download_button("Download ablation table (CSV)", df.to_csv(index=False).encode("utf-8"), "ablation.csv", "text/csv")


def _duplicate_section() -> None:
    st.markdown("##### Q2 · How duplicates distort classification, clustering and recommendation")
    df = _run("duplicates", "Run duplicate study (3 settings × 10 splits, about 15-30 s)")
    if df is None:
        return
    c.table(df.round(3))
    leak = df.iloc[0]
    if leak["test docs with a copy in training (total over splits)"]:
        fig = go.Figure(go.Bar(x=["test reviews whose<br>copy is in training", "all other<br>test reviews"],
                               y=[leak["NB accuracy on those docs"], leak["NB accuracy on other test docs"]],
                               marker=dict(color=[c.SERIES[1], c.SERIES[0]]), texttemplate="%{y:.1%}",
                               textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        c.style(fig, 320, "NB accuracy, plain random split", legend=False)
        fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
        fig.update_xaxes(tickangle=0)
        a, b = st.columns([2, 3])
        with a:
            c.plot(fig, key="dup_leak")
        with b:
            st.markdown(
                f"With duplicates kept and a plain random split, **{int(leak['test docs with a copy in training (total over splits)'])}** "
                f"test reviews (summed over 10 splits) had an identical or near-identical copy in the training set. "
                f"Naive Bayes classified **{leak['NB accuracy on those docs']:.1%}** of them correctly versus "
                f"**{leak['NB accuracy on other test docs']:.1%}** for the other test reviews - the model is partly "
                f"*remembering* the copy, which inflates the reported accuracy. The duplicate-aware split and "
                f"de-duplication remove this leakage. In clustering, copies always share a cluster and so add "
                f"'free' true-positive pairs; in recommendation, without collapsing, "
                f"{leak['near-copies in Top-5 (no collapsing)']:.1%} of Top-5 slots are copies of the query.")


def _kmeans_section() -> None:
    st.markdown("##### Q5 · K-Means robustness: initialisation, distance and number of clusters")
    a, b = st.columns(2)
    with a:
        df = _run("kmeans", "Run K-Means variant study (4 variants × 10 seeds)")
        if df is not None:
            names = df["initialisation"] + "<br>" + df["distance"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=df["purity (single runs, mean)"], name="purity, single run (mean ± sd)",
                                 marker=dict(color=c.SERIES[0]),
                                 error_y=dict(type="data", array=df["purity sd"], color=c.INK_2, thickness=1)))
            best_col = [col for col in df.columns if col.startswith("purity (best of")][0]
            fig.add_trace(go.Bar(x=names, y=df[best_col], name=best_col, marker=dict(color=c.SERIES[1])))
            c.style(fig, 360, "Purity by K-Means variant", legend_bottom=True)
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1])
            fig.update_xaxes(tickangle=0)
            c.plot(fig, key="km_var")
            compact = pd.DataFrame({
                "variant": df["initialisation"] + " · " + df["distance"],
                "purity (mean ± sd)": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(df["purity (single runs, mean)"], df["purity sd"])],
                "Rand Index (mean ± sd)": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(df["Rand Index (single runs, mean)"], df["RI sd"])],
                "best-of-restarts purity / RI": [f"{p:.3f} / {r:.3f}" for p, r in zip(df[best_col],
                                                  df[[col for col in df.columns if col.startswith("Rand Index (best of")][0]])],
            })
            c.table(compact)
    with b:
        df = _run("k", "Run k-sensitivity study (k = 2 ... 12)")
        if df is not None:
            fig = go.Figure()
            for name, color in (("purity", c.SERIES[0]), ("Rand Index", c.SERIES[1])):
                fig.add_trace(go.Scatter(x=df["k"], y=df[name], mode="lines+markers", name=name,
                                         line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=c.SURFACE, width=2))))
            c.style(fig, 300, "Purity and Rand Index vs k", x_title="k", y_title="score")
            fig.update_yaxes(range=[0, 1])
            c.plot(fig, key="k_scores")
            fig = go.Figure(go.Scatter(x=df["k"], y=df["RSS"], mode="lines+markers", line=dict(color=c.SERIES[0], width=2),
                                       marker=dict(size=8, line=dict(color=c.SURFACE, width=2)), name="RSS"))
            c.style(fig, 260, "RSS vs k (elbow plot)", legend=False, x_title="k", y_title="RSS")
            c.plot(fig, key="k_rss")
            st.caption("RSS falls as k grows and the Rand Index rises (more pairs end up in different clusters); purity "
                       "only reaches 1 at k = N and fluctuates for moderate k. None of them can choose k on its own, so "
                       "k is fixed to the number of departments when comparing with Naive Bayes.")


def render() -> None:
    c.page_header("Performance analytics",
                  "All evaluation metrics in one place, plus the experiments that answer the report's inference "
                  "questions.")
    if not c.has_labels():
        st.warning("Performance analytics need a category label field.")
        return
    _scorecard()
    st.divider()
    _ablation_section()
    st.divider()
    _duplicate_section()
    st.divider()
    _kmeans_section()
