"""Text preprocessing configuration, feature engineering (inverted index + TF-IDF) and duplicate analysis."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from textmining.experiments import DEFAULT_DEDUP, DEFAULT_FEAT, DEFAULT_PREP, detection_quality, detection_sweep
from textmining.indexing import cosine_similarity, tf_weight
from textmining.pipeline import DedupConfig, FeatureConfig
from textmining.preprocessing import PreprocessConfig, Preprocessor, backend_names, raw_tokens, vocabulary_funnel
from textmining.stopwords import REVIEW_STOPWORDS
from views import common as c

NORMALIZERS = {"none": "None", "stem": "Porter stemming", "lemma": "WordNet lemmatisation"}
NGRAMS = {"1": (1, 1), "1-2": (1, 2), "1-3": (1, 3)}
POLICIES = {"keep_all": "Keep all documents", "remove_exact": "Remove exact duplicates (keep first)",
            "remove_near": "Remove exact + near-duplicates (keep first of each group)"}


def _config_form() -> None:
    prep, feat, dedup, _ = c.configs()
    with c.card():
        with st.form("prep_form"):
            st.markdown("**Preprocessing configuration**")
            a, b, cc = st.columns(3)
            with a:
                st.caption("CLEANING & TOKENISATION")
                strip_markup = st.checkbox("Remove HTML markup & entities (<br />, &amp;)", prep.strip_markup)
                remove_urls = st.checkbox("Remove URLs and e-mail addresses", prep.remove_urls)
                contractions = st.checkbox("Expand contractions (didn't -> did not)", prep.expand_contractions)
                lowercase = st.checkbox("Lower-case", prep.lowercase)
                numbers = st.checkbox("Drop pure numbers (keeps ps4, mp3)", prep.remove_numbers)
                min_len = st.number_input("Minimum token length", 1, 5, prep.min_token_len)
            with b:
                st.caption("STOP WORDS, NORMALISATION, N-GRAMS")
                stops = st.checkbox("Remove stop words", prep.remove_stopwords)
                negations = st.checkbox("...but keep negations (not, no, never)", prep.keep_negations)
                custom = st.text_area("Extra domain stop words (comma separated)", ", ".join(prep.custom_stopwords),
                                      height=90)
                norm = st.radio("Word normalisation", list(NORMALIZERS), index=list(NORMALIZERS).index(prep.normalizer),
                                format_func=NORMALIZERS.get, horizontal=True)
                ng_key = f"{prep.ngram_min}" if prep.ngram_max == 1 else f"{prep.ngram_min}-{prep.ngram_max}"
                ngram = st.radio("N-grams", list(NGRAMS), index=list(NGRAMS).index(ng_key) if ng_key in NGRAMS else 0,
                                 format_func=lambda k: {"1": "unigrams", "1-2": "uni + bigrams", "1-3": "uni + bi + trigrams"}[k],
                                 horizontal=True)
            with cc:
                st.caption("TF-IDF FEATURES & DUPLICATES")
                tf = st.selectbox("TF weight", ["log", "raw", "binary"], ["log", "raw", "binary"].index(feat.tf_scheme),
                                  format_func={"log": "log: 1 + log10(tf)", "raw": "raw count", "binary": "binary"}.get)
                idf = st.selectbox("IDF weight", ["idf", "smooth", "none"], ["idf", "smooth", "none"].index(feat.idf_scheme),
                                   format_func={"idf": "log10(N / df)", "smooth": "log10((1+N)/(1+df)) + 1",
                                                "none": "none (tf only)"}.get)
                min_df = st.number_input("min_df (documents)", 1, 20, feat.min_df)
                max_df = st.slider("max_df (share of documents)", 0.10, 1.0, float(feat.max_df), 0.05)
                max_feat = st.number_input("max_features (vocabulary cap)", 500, 20000, feat.max_features, step=500)
                policy = st.selectbox("Duplicate handling", list(POLICIES), list(POLICIES).index(dedup.policy),
                                      format_func=POLICIES.get)
                threshold = st.slider("Near-duplicate threshold (cosine)", 0.50, 0.99, float(dedup.threshold), 0.01)
            s1, s2, _ = st.columns([1, 1, 3])
            apply = s1.form_submit_button("Apply configuration", type="primary")
            reset = s2.form_submit_button("Reset to defaults")
        if apply:
            n_min, n_max = NGRAMS[ngram]
            st.session_state.prep_cfg = PreprocessConfig(
                strip_markup=strip_markup, remove_urls=remove_urls, expand_contractions=contractions,
                lowercase=lowercase, remove_numbers=numbers, min_token_len=int(min_len), remove_stopwords=stops,
                keep_negations=negations, custom_stopwords=tuple(w.strip().lower() for w in custom.split(",") if w.strip()),
                normalizer=norm, ngram_min=n_min, ngram_max=n_max)
            st.session_state.feat_cfg = FeatureConfig(min_df=int(min_df), max_df=float(max_df),
                                                      max_features=int(max_feat), tf_scheme=tf, idf_scheme=idf)
            st.session_state.dedup_cfg = DedupConfig(policy=policy, threshold=float(threshold))
            st.session_state.experiments = {}
            c.flash("Configuration applied - every page now uses the new pipeline.")
        if reset:
            st.session_state.prep_cfg, st.session_state.feat_cfg = DEFAULT_PREP, DEFAULT_FEAT
            st.session_state.dedup_cfg = DEFAULT_DEDUP
            st.session_state.experiments = {}
            c.flash("Default configuration restored.")


def _preview_tab(pr) -> None:
    corpus = c.corpus()

    def noisiness(i: int) -> tuple:
        t = corpus.texts[i]
        n_words = len(t.split())
        marks = ("<br" in t) + ("&" in t and ";" in t) + ("'" in t or "’" in t) + any(w.isupper() and len(w) > 2 for w in t.split())
        return (-marks, abs(n_words - 60), i)

    default = min(range(len(corpus)), key=noisiness)
    idx = st.selectbox("Document to trace through the pipeline", list(range(len(corpus))), index=default,
                       format_func=lambda i: f"{corpus.doc_ids[i]} · {c.snippet(corpus.titles[i], 60)}")
    stages = Preprocessor(st.session_state.prep_cfg).stages(corpus.texts[idx])
    st.markdown("<p class='rl-stage'>Raw text (as stored)</p>", unsafe_allow_html=True)
    c.text_block(corpus.texts[idx])
    for name, toks in stages.items():
        if name == "raw tokens":
            continue
        if name == "cleaned text":
            st.markdown("<p class='rl-stage'>1 · cleaned text (markup, URLs removed; contractions expanded; "
                        "lower-cased)</p>", unsafe_allow_html=True)
            c.text_block(toks[0])
            continue
        label = {"tokens": "2 · tokens", "after stop-word removal": "3 · after stop-word removal",
                 "after stemming/lemmatisation": "4 · after stemming / lemmatisation",
                 "index terms": "5 · index terms (with n-grams)"}.get(name, name)
        st.markdown(f"<p class='rl-stage'>{label} <span class='rl-pill'>{len(toks)} tokens</span></p>",
                    unsafe_allow_html=True)
        c.text_block("  ·  ".join(toks))
    st.caption(f"Raw whitespace tokens: {len(stages['raw tokens'])}. Backends - stemmer: "
               f"{backend_names()['stemmer']}; lemmatiser: {backend_names()['lemmatizer']}")


def _vocab_tab(pr) -> None:
    prep = st.session_state.prep_cfg
    texts = pr.corpus.texts
    funnel = pd.DataFrame(vocabulary_funnel(texts, prep))
    raw_counts = Counter(t for txt in texts for t in raw_tokens(txt))
    proc_counts = Counter(t for terms in pr.features.terms for t in terms)
    raw_len = np.array([len(raw_tokens(t)) for t in texts])
    proc_len = np.array([len(t) for t in pr.features.terms])

    def summary(counts: Counter, lengths: np.ndarray) -> dict:
        total = sum(counts.values())
        top10 = sum(n for _, n in counts.most_common(10))
        return {"vocabulary size": len(counts), "total tokens": total, "mean tokens / doc": round(float(lengths.mean()), 1),
                "median tokens / doc": float(np.median(lengths)), "max tokens / doc": int(lengths.max()),
                "hapax terms (freq 1)": sum(1 for n in counts.values() if n == 1),
                "type/token ratio": round(len(counts) / total, 3) if total else 0,
                "share of tokens in top-10 terms": f"{100 * top10 / total:.1f}%" if total else "0%"}

    s = pd.DataFrame({"before (raw tokens)": summary(raw_counts, raw_len),
                      "after (index terms)": summary(proc_counts, proc_len)})
    s.loc["vocabulary after df pruning"] = ["-", len(pr.features.index)]
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Summary statistics**")
        c.table(s.reset_index().rename(columns={"index": "statistic"}).astype(str))
    with right:
        fig = c.bar(funnel, "stage", "vocabulary size", horizontal=True, height=300,
                    title="Vocabulary size after each cumulative step", text_fmt="%{x:,}")
        c.chart_with_table(fig, funnel, "voc_funnel")

    a, b = st.columns(2)
    top_raw = pd.DataFrame(raw_counts.most_common(20), columns=["token", "frequency"])
    top_proc = pd.DataFrame(proc_counts.most_common(20), columns=["term", "frequency"])
    with a:
        fig = c.bar(top_raw, "token", "frequency", horizontal=True, height=480, color=c.SERIES[1],
                    title="Top-20 tokens BEFORE preprocessing")
        c.chart_with_table(fig, top_raw, "voc_top_raw")
    with b:
        fig = c.bar(top_proc, "term", "frequency", horizontal=True, height=480, title="Top-20 terms AFTER preprocessing")
        c.chart_with_table(fig, top_proc, "voc_top_proc")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, counts, color in (("before (raw tokens)", raw_counts, c.SERIES[1]),
                                    ("after (index terms)", proc_counts, c.SERIES[0])):
            freqs = sorted(counts.values(), reverse=True)
            ranks = np.arange(1, len(freqs) + 1)
            fig.add_trace(go.Scatter(x=ranks, y=freqs, mode="lines", name=name, line=dict(color=color, width=2),
                                     hovertemplate="rank %{x}: freq %{y}<extra>" + name + "</extra>"))
        c.style(fig, 340, "Rank-frequency distribution (Zipf), log-log", x_title="rank", y_title="frequency")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        zipf = pd.DataFrame({"rank": [1, 10, 100, 1000],
                             "raw freq": [sorted(raw_counts.values(), reverse=True)[r - 1] if r <= len(raw_counts) else None for r in (1, 10, 100, 1000)],
                             "processed freq": [sorted(proc_counts.values(), reverse=True)[r - 1] if r <= len(proc_counts) else None for r in (1, 10, 100, 1000)]})
        c.chart_with_table(fig, zipf, "voc_zipf")
    with b:
        bins = np.linspace(0, max(raw_len.max(), proc_len.max()) + 1, 31)
        fig = go.Figure()
        rows = {}
        for name, lens, color in (("before (raw tokens)", raw_len, c.SERIES[1]), ("after (index terms)", proc_len, c.SERIES[0])):
            h, e = np.histogram(lens, bins=bins)
            rows[name] = h
            fig.add_trace(go.Scatter(x=(e[:-1] + e[1:]) / 2, y=h, mode="lines", name=name, line=dict(color=color, width=2),
                                     line_shape="hvh", hovertemplate="~%{x:.0f} tokens: %{y} docs<extra>" + name + "</extra>"))
        c.style(fig, 340, "Document length before vs after", x_title="tokens per document", y_title="documents")
        c.chart_with_table(fig, pd.DataFrame({"bin start": bins[:-1].round(0), **rows}), "voc_len")

    df = pd.Series(pr.features.index.df)
    a, b = st.columns(2)
    with a:
        h, e = np.histogram(np.log10(df.values), bins=25)
        dfh = pd.DataFrame({"df from": (10 ** e[:-1]).round(1), "df to": (10 ** e[1:]).round(1), "terms": h})
        fig = go.Figure(go.Bar(x=10 ** ((e[:-1] + e[1:]) / 2), y=h, marker=dict(color=c.SERIES[0]),
                               hovertemplate="df ≈ %{x:.0f}: %{y} terms<extra></extra>"))
        c.style(fig, 300, "Document-frequency distribution of index terms", legend=False, x_title="document frequency (log)",
                y_title="terms")
        fig.update_xaxes(type="log")
        c.chart_with_table(fig, dfh, "voc_df")
    with b:
        idf = pd.Series(pr.features.model.idf)
        h, e = np.histogram(idf.values, bins=25)
        fig = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h, width=(e[1] - e[0]) * 0.85, marker=dict(color=c.SERIES[0]),
                               hovertemplate="idf ≈ %{x:.2f}: %{y} terms<extra></extra>"))
        c.style(fig, 300, "IDF distribution", legend=False, x_title="idf = log10(N / df)", y_title="terms")
        c.chart_with_table(fig, pd.DataFrame({"idf from": e[:-1].round(3), "terms": h}), "voc_idf")


def _index_tab(pr) -> None:
    index, model = pr.features.index, pr.features.model
    corpus = pr.corpus
    X = pr.features.X
    m = st.columns(5)
    m[0].metric("Index terms", f"{len(index):,}")
    m[1].metric("Postings", f"{index.n_postings():,}")
    m[2].metric("Mean postings / term", f"{index.n_postings() / max(1, len(index)):.1f}")
    m[3].metric("Doc-term matrix", f"{X.shape[0]:,} × {X.shape[1]:,}")
    m[4].metric("Non-zero cells", c.pct(float((X > 0).mean()), 2))
    c.explain("How the TF-IDF weights are computed (manual implementation)",
              "1. **Inverted index** - for every term a postings list of `(document, term frequency)` pairs.\n"
              "2. **Weights** - walking each postings list: tf weight × idf.\n"
              "3. **Length normalisation** - each document vector is divided by its Euclidean length, so every "
              "vector has length 1 and the cosine of two documents is simply their dot product.",
              [r"w_{t,d} = \bigl(1 + \log_{10} \mathrm{tf}_{t,d}\bigr) \cdot \log_{10}\frac{N}{\mathrm{df}_t}",
               r"\hat{w}_{t,d} = \frac{w_{t,d}}{\sqrt{\sum_{t'} w_{t',d}^2}} \qquad "
               r"\cos(d_1, d_2) = \sum_t \hat{w}_{t,d_1}\,\hat{w}_{t,d_2}"])

    st.markdown("**Inverted index lookup**")
    a, b = st.columns([2, 3])
    with a:
        query = st.text_input("Term to look up (it is preprocessed the same way as the documents)", "batteries")
        pre = Preprocessor(st.session_state.prep_cfg)
        processed = pre(query) if query.strip() else []
        target = max(processed, key=len) if processed else ""
        if target in index.postings:
            st.markdown(f"Index term **`{target}`** · df = **{index.df[target]}** · cf = {index.cf[target]} · "
                        f"idf = **{model.idf[target]:.4f}**")
        elif query.strip():
            st.warning(f"`{target or query}` is not in the index (removed by preprocessing or df pruning).")
        sample = pd.DataFrame([{"term": t, "df": index.df[t], "idf": round(model.idf[t], 4), "cf": index.cf[t],
                                "postings (doc, tf)": ", ".join(f"({corpus.doc_ids[d]}, {tf})" for d, tf in index.postings[t][:4])
                                + (" ..." if index.df[t] > 4 else "")}
                               for t in index.vocabulary[:: max(1, len(index) // 40)][:40]])
        st.caption("Sample of the index (every n-th term, alphabetical)")
        c.table(sample, height=300)
    with b:
        if target in index.postings:
            rows = []
            for d, tf in index.postings[target]:
                rows.append({"doc_id": corpus.doc_ids[d], "category": (corpus.labels or [""] * len(corpus))[d],
                             "tf": tf, "tf weight": round(tf_weight(tf, pr.feat.tf_scheme), 4),
                             "normalised w(t,d)": round(model.doc_vectors[d].get(target, 0.0), 4),
                             "title": corpus.titles[d]})
            postings = pd.DataFrame(rows).sort_values("normalised w(t,d)", ascending=False)
            st.caption(f"Postings list of '{target}' ({len(postings)} documents, sorted by weight)")
            c.table(postings, height=380, column_config={"normalised w(t,d)": c.progress_column(max_value=1.0)})

    st.markdown("**Document vector inspector**")
    idx = st.selectbox("Document", list(range(len(corpus))), index=0, key="vec_doc",
                       format_func=lambda i: f"{corpus.doc_ids[i]} · {c.snippet(corpus.titles[i], 60)}")
    tf_counts = Counter(t for t in pr.features.terms[idx] if t in model.idf)
    rows = [{"term": t, "tf": n, "tf weight": round(tf_weight(n, pr.feat.tf_scheme), 4), "idf": round(model.idf[t], 4),
             "w = tf_w × idf": round(tf_weight(n, pr.feat.tf_scheme) * model.idf[t], 4),
             "normalised w": round(model.doc_vectors[idx].get(t, 0.0), 4)} for t, n in tf_counts.items()]
    vec = pd.DataFrame(rows).sort_values("normalised w", ascending=False) if rows else pd.DataFrame()
    a, b = st.columns([3, 2])
    with a:
        c.table(vec, height=300, column_config={"normalised w": c.progress_column(max_value=1.0)})
    with b:
        sq = sum(w * w for w in model.doc_vectors[idx].values())
        st.metric("Vector length before normalisation", f"{model.doc_norms[idx]:.4f}")
        st.metric("Σ (normalised w)² after normalisation", f"{sq:.4f}")
        st.code(c.snippet(corpus.texts[idx], 400), language=None)


def _similarity_tab(pr) -> None:
    corpus, model, S = pr.corpus, pr.features.model, pr.features.sim
    labels = corpus.labels or ["all"] * len(corpus)
    fmt = lambda i: f"{corpus.doc_ids[i]} · {labels[i]} · {c.snippet(corpus.titles[i], 45)}"  # noqa: E731
    a, b = st.columns(2)
    i = a.selectbox("Document A", list(range(len(corpus))), index=0, format_func=fmt, key="sim_a")
    order = np.argsort(-S[i])
    j = b.selectbox("Document B", list(range(len(corpus))), index=int(order[1]), format_func=fmt, key="sim_b")
    cos = cosine_similarity(model.doc_vectors[i], model.doc_vectors[j])
    shared = sorted(((t, w * model.doc_vectors[j][t]) for t, w in model.doc_vectors[i].items() if t in model.doc_vectors[j]),
                    key=lambda kv: -kv[1])
    with c.card():
        x, y = st.columns([1, 3])
        x.metric("cosine(A, B)", f"{cos:.4f}")
        y.markdown("**Shared terms and their contribution** w(t,A)·w(t,B): " +
                   (", ".join(f"`{t}` {v:.3f}" for t, v in shared[:10]) or "none"))
    # heatmap of a category-sorted sample
    per = 8
    sample = []
    for lab in sorted(set(labels)):
        sample.extend([k for k, l in enumerate(labels) if l == lab][:per])
    sub = S[np.ix_(sample, sample)]
    names = [f"{corpus.doc_ids[k]} ({labels[k][:10]})" for k in sample]
    fig = go.Figure(go.Heatmap(z=sub, x=names, y=names, colorscale=c.SEQ_BLUE, zmin=0, zmax=1,
                               colorbar=dict(title="cos", thickness=12),
                               hovertemplate="%{y}<br>%{x}<br>cosine %{z:.3f}<extra></extra>"))
    c.style(fig, 560, f"Cosine similarity of {per} documents per category (sorted by category)", legend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed", showgrid=False)
    a, b = st.columns([3, 2])
    with a:
        c.chart_with_table(fig, pd.DataFrame(sub.round(3), index=names, columns=names).reset_index(), "sim_heat")
    with b:
        iu = np.triu_indices(len(labels), k=1)
        lab = np.asarray(labels)
        same = lab[iu[0]] == lab[iu[1]]
        vals = S[iu]
        bins = np.linspace(0, 1, 41)
        fig = go.Figure()
        rows = {}
        for name, mask, color in (("same category", same, c.SERIES[0]), ("different category", ~same, c.SERIES[1])):
            h, e = np.histogram(vals[mask], bins=bins)
            dens = h / max(1, h.sum())
            rows[name] = dens.round(4)
            fig.add_trace(go.Scatter(x=(e[:-1] + e[1:]) / 2, y=dens, mode="lines", name=name, line=dict(color=color, width=2),
                                     line_shape="hvh", hovertemplate="cos ≈ %{x:.2f}: %{y:.1%}<extra>" + name + "</extra>"))
        c.style(fig, 300, "Pairwise cosine: same vs different category", x_title="cosine similarity",
                y_title="share of pairs")
        fig.update_yaxes(tickformat=".0%", type="log")
        c.chart_with_table(fig, pd.DataFrame({"cos from": bins[:-1].round(3), **rows}), "sim_dist")
        m_same = f"{vals[same].mean():.3f}" if same.any() else "n/a"
        m_diff = f"{vals[~same].mean():.3f}" if (~same).any() else "n/a"
        st.markdown(f"<span class='rl-note'>Mean cosine - same category: <b>{m_same}</b>, "
                    f"different category: <b>{m_diff}</b> (over {len(vals):,} document pairs).</span>",
                    unsafe_allow_html=True)


def _duplicates_tab(pr) -> None:
    dup, full = pr.duplicates, pr.full_corpus
    m = st.columns(5)
    m[0].metric("Duplicate groups", len(dup.groups))
    m[1].metric("Exact-copy groups", len(dup.exact_groups))
    m[2].metric(f"Near-duplicate pairs (cos ≥ {dup.threshold:.2f})", len(dup.near_pairs))
    m[3].metric("Documents in groups", dup.n_in_groups)
    m[4].metric("Removed by current policy", len(full) - len(pr.corpus))
    c.explain("How duplicates are detected and handled",
              "* **Exact duplicates** - a hash table of the normalised text (lower-cased, markup and punctuation "
              "removed) finds identical re-posts.\n"
              "* **Near-duplicates** - pairs whose TF-IDF cosine similarity is at least the threshold; the word "
              "3-shingle Jaccard similarity is shown as a second opinion. Pairs are merged into groups (union-find).\n"
              "* **Handling** - by default one document per group is kept, so that a review and its copy can "
              "never sit in both the training and test split (leakage), do not form artificial micro-clusters, "
              "and do not fill the recommendation list with copies. The duplicate study on the *Performance "
              "analytics* page measures these effects.")
    quality = detection_quality(pr)
    if quality:
        st.markdown(f"<span class='rl-note'>Against the dataset's ground-truth column <code>dup_group</code>: "
                    f"precision <b>{quality['precision']:.3f}</b>, recall <b>{quality['recall']:.3f}</b> "
                    f"({quality['correct pairs']} of {quality['true pairs']} true duplicate pairs found, "
                    f"{quality['detected pairs']} pairs flagged).</span>", unsafe_allow_html=True)
    kept = set(pr.kept_indices)
    sim = pr.full_features.sim
    rows = []
    for gi, g in enumerate(dup.groups, start=1):
        for k in g:
            rows.append({"group": f"G{gi:03d}", "doc_id": full.doc_ids[k],
                         "category": (full.labels or [""] * len(full))[k],
                         "cos to first": round(float(sim[g[0], k]), 3), "kept": k in kept,
                         "text": c.snippet(full.texts[k], 150)})
    c.table(pd.DataFrame(rows), height=380, column_config={"text": st.column_config.TextColumn(width="large")})
    if quality is not None:
        if c.wide_button("Run threshold sweep (precision / recall vs threshold)", key="dup_sweep"):
            st.session_state.experiments["dup_sweep"] = pd.DataFrame(detection_sweep(pr))
        sweep = st.session_state.experiments.get("dup_sweep")
        if sweep is not None:
            fig = go.Figure()
            for name, color in (("precision", c.SERIES[0]), ("recall", c.SERIES[1])):
                fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[name], mode="lines+markers", name=name,
                                         line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=c.SURFACE, width=2))))
            c.style(fig, 300, "Near-duplicate detector vs ground truth", x_title="cosine threshold", y_title="score")
            fig.update_yaxes(range=[0, 1.05])
            c.chart_with_table(fig, sweep.round(3), "dup_sweep_chart")


def render() -> None:
    c.page_header("Preprocessing & feature engineering",
                  "Configure the text pipeline, inspect its effect on the vocabulary, and explore the manually built "
                  "inverted index, TF-IDF vectors and cosine similarities.")
    c.show_flash()
    _config_form()
    pr = c.pipeline()
    st.markdown(" ".join(f"<span class='rl-pill'>{k}: {v:.2f}s</span>" for k, v in pr.timings.items()),
                unsafe_allow_html=True)
    tabs = st.tabs(["Pipeline preview", "Vocabulary & term frequencies", "Inverted index & TF-IDF",
                    "Cosine similarity", "Duplicates"])
    with tabs[0]:
        _preview_tab(pr)
    with tabs[1]:
        _vocab_tab(pr)
    with tabs[2]:
        _index_tab(pr)
    with tabs[3]:
        _similarity_tab(pr)
    with tabs[4]:
        _duplicates_tab(pr)
