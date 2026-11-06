"""Data upload / ingestion interface."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from textmining.duplicates import text_hash
from textmining.ingestion import (build_corpus, corpus_statistics, guess_column, load_bundled, noise_profile,
                                  read_table, read_txt_zip)
from views import common as c

SOURCES = ["Bundled sample - JSON (text and metadata stored separately)", "Bundled sample - CSV",
           "Upload CSV / JSON / JSONL file", "Upload ZIP folder of .txt files"]


def _activate(corpus, log, source: str) -> None:
    st.session_state.corpus = corpus
    st.session_state.ingest_log = log
    st.session_state.corpus_source = source
    st.session_state.experiments = {}
    if corpus.labels:
        st.session_state.model_cfg = c.replace(st.session_state.model_cfg, k=0)


def render() -> None:
    c.page_header("Data upload & ingestion",
                  "Load a document collection, map its fields, and inspect the corpus before any processing.")
    c.show_flash()
    with c.card():
        src = st.radio("Data source", SOURCES, horizontal=False, key="ingest_source")
        if src.startswith("Bundled"):
            fmt = "json" if "JSON" in src else "csv"
            st.caption("1,224 Amazon product reviews from 6 departments (Amazon Reviews 2023, McAuley Lab, UCSD), "
                       "sampled reproducibly by `scripts/build_dataset.py`.")
            if c.wide_button("Load bundled sample", key="load_bundled", primary=True):
                corpus, log = load_bundled(fmt)
                _activate(corpus, log, f"bundled sample ({fmt.upper()})")
                c.flash(f"Loaded {len(corpus):,} documents from the bundled {fmt.upper()} file.")
        else:
            is_zip = src.startswith("Upload ZIP")
            up = st.file_uploader("Choose a file", type=["zip"] if is_zip else ["csv", "json", "jsonl"],
                                  key="uploader_zip" if is_zip else "uploader_table")
            if up is not None:
                try:
                    raw = read_txt_zip(up.getvalue()) if is_zip else read_table(up.getvalue(), up.name)
                except Exception as exc:  # show parsing problems to the user instead of crashing
                    st.error(f"Could not read the file: {exc}")
                    raw = None
                if raw is not None and len(raw):
                    st.markdown(f"**Preview** - {len(raw):,} rows, {len(raw.columns)} columns")
                    c.table(raw.head(8).astype(str), height=240)
                    cols = list(raw.columns)
                    none = "(none)"
                    g_text = guess_column(cols, ["text", "content", "body", "review", "review_text", "abstract",
                                                 "complaint", "document"]) or cols[0]
                    g_id = guess_column(cols, ["doc_id", "id", "document_id", "uid"])
                    g_lab = guess_column(cols, ["category", "label", "class", "topic", "product"])
                    g_title = guess_column(cols, ["title", "subject", "headline", "name"])
                    a, b, cc, d = st.columns(4)
                    text_col = a.selectbox("Text column *", cols, index=cols.index(g_text))
                    id_col = b.selectbox("Document ID column", [none] + cols,
                                         index=([none] + cols).index(g_id) if g_id else 0)
                    lab_col = cc.selectbox("Category label column", [none] + cols,
                                           index=([none] + cols).index(g_lab) if g_lab else 0)
                    title_col = d.selectbox("Title column", [none] + cols,
                                            index=([none] + cols).index(g_title) if g_title else 0)
                    if c.wide_button("Ingest this dataset", key="ingest_upload", primary=True):
                        corpus, log = build_corpus(raw, text_col=text_col, id_col=None if id_col == none else id_col,
                                                   label_col=None if lab_col == none else lab_col,
                                                   title_col=None if title_col == none else title_col,
                                                   name=up.name, source_format=up.name.rsplit(".", 1)[-1].upper())
                        _activate(corpus, log, f"uploaded: {up.name}")
                        c.flash(f"Ingested {len(corpus):,} documents from {up.name}."
                                + ("" if corpus.labels else " No label column was mapped - classification and "
                                   "clustering evaluation need labels."))

    corpus = c.corpus()
    log = st.session_state.ingest_log
    stats = corpus_statistics(corpus)
    st.markdown(f"##### Active corpus · {corpus.name}")
    st.markdown(" ".join(f"<span class='rl-pill'>{k}: {v}</span>" for k, v in log.items()), unsafe_allow_html=True)

    m = st.columns(6)
    m[0].metric("Documents", f"{stats['documents']:,}")
    m[1].metric("Categories", stats["categories"])
    m[2].metric("Raw text", f"{stats['raw text size (MB)']:.2f} MB")
    m[3].metric("Raw tokens", f"{stats['total raw tokens']:,}")
    m[4].metric("Raw vocabulary", f"{stats['raw vocabulary (whitespace tokens)']:,}")
    m[5].metric("Metadata fields", stats["metadata fields"])

    hashes = pd.Series([text_hash(t) for t in corpus.texts])
    exact_copies = int(hashes.duplicated().sum())
    st.markdown(f"<span class='rl-note'>Ingestion check: <b>{exact_copies}</b> documents are exact copies of another "
                f"document after normalisation (case, markup and punctuation ignored). Near-duplicates are detected "
                f"after TF-IDF on the <i>Preprocessing & Features</i> page.</span>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Document text", "Metadata", "Corpus statistics", "Noise profile"])
    with t1:
        st.caption("Main text-content field - the only field that is preprocessed and indexed.")
        c.table(pd.DataFrame({"doc_id": corpus.doc_ids, "text": corpus.texts}), height=380)
    with t2:
        st.caption("Metadata kept in a separate table keyed by doc_id (never indexed).")
        c.table(corpus.metadata.reset_index(), height=300)
        schema = []
        for col in corpus.metadata.columns:
            s = corpus.metadata[col]
            non_empty = int((s.astype(str).str.strip() != "").sum())
            role = ("category label" if col == corpus.label_field else "title" if col == corpus.title_field
                    else "metadata")
            schema.append({"field": col, "role": role, "dtype": str(s.dtype), "non-empty": non_empty,
                           "unique values": int(s.nunique()), "example": str(s.iloc[0])[:60] if len(s) else ""})
        st.markdown("**Metadata schema**")
        c.table(pd.DataFrame(schema))
    with t3:
        left, right = st.columns([2, 3])
        with left:
            c.table(pd.DataFrame({"statistic": list(stats), "value": [str(v) for v in stats.values()]}), height=420)
        with right:
            if corpus.labels:
                counts = pd.Series(corpus.labels).value_counts().rename_axis("category").reset_index(name="documents")
                fig = c.bar(counts, "category", "documents", horizontal=True, height=320,
                            title="Category labels (used for classification & clustering evaluation)", text_fmt="%{x}")
                c.chart_with_table(fig, counts, "ing_labels")
    with t4:
        noise = noise_profile(corpus)
        fig = c.bar(noise.sort_values("share of corpus", ascending=False), "noise pattern", "share of corpus",
                    horizontal=True, height=320, title="Realistic noise in the raw text (share of documents)",
                    text_fmt="%{x:.1%}")
        fig.update_xaxes(tickformat=".0%")
        c.chart_with_table(fig, noise.assign(**{"share of corpus": noise["share of corpus"].round(3)}), "ing_noise")
        examples = [t for t in corpus.texts if "<br" in t][:2]
        if examples:
            st.caption("Examples of raw review text with markup:")
        for ex in examples:
            c.text_block(ex[:400])

    with st.expander("Export the active corpus"):
        flat = corpus.metadata.reset_index()
        flat.insert(1, "text", corpus.texts)
        st.download_button("Download as CSV", flat.to_csv(index=False).encode("utf-8"), "corpus.csv", "text/csv")
        nested = [{"doc_id": d, "text": t, "metadata": {k: (v.item() if hasattr(v, "item") else v)
                                                        for k, v in corpus.metadata.loc[d].to_dict().items()}}
                  for d, t in zip(corpus.doc_ids, corpus.texts)]
        st.download_button("Download as JSON", json.dumps(nested, ensure_ascii=False, indent=1).encode("utf-8"),
                           "corpus.json", "application/json")
