"""
ReviewLens - Streamlit Text Mining & Document Intelligence System
BITS WILP - Information Retrieval (AIMLCZG537 / DSECLZG537), Assignment 2

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="ReviewLens - Text Mining & Document Intelligence", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")

from views import common  # noqa: E402
from views import dashboard, ingestion, mining, performance, preprocessing, recommend  # noqa: E402

CSS = """
<style>
:root { --rl-ink:#0b0b0b; --rl-ink2:#52514e; --rl-muted:#898781; --rl-line:#e1e0d9; --rl-accent:#2a78d6; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
.rl-header h2 { margin: 0 0 .15rem 0; font-weight: 650; letter-spacing: -.01em; color: var(--rl-ink); }
.rl-header p { margin: 0 0 1rem 0; color: var(--rl-ink2); font-size: 1.0rem; }
.rl-brand { font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; color: var(--rl-ink); margin-bottom: 0; }
.rl-brand span { color: var(--rl-accent); }
.rl-tag { color: var(--rl-ink2); font-size: .82rem; margin-top: .1rem; }
.rl-step { display:flex; gap:.6rem; align-items:flex-start; padding:.45rem 0; border-bottom:1px solid var(--rl-line); }
.rl-step:last-child { border-bottom: none; }
.rl-step b { color: var(--rl-ink); }
.rl-step small { color: var(--rl-ink2); }
.rl-num { min-width:1.6rem; height:1.6rem; border-radius:50%; background:#eaf2fc; color:var(--rl-accent);
          font-weight:700; font-size:.8rem; display:flex; align-items:center; justify-content:center; }
.rl-pill { display:inline-block; padding:.1rem .55rem; border-radius:999px; background:#f0efec; color:var(--rl-ink2);
           font-size:.78rem; margin:.1rem .25rem .1rem 0; }
.rl-note { color: var(--rl-ink2); font-size: .9rem; }
.rl-text { white-space: pre-wrap; word-break: break-word; background:#f3f2ee; border-radius:8px; padding:.7rem .9rem;
           color: var(--rl-ink); font-size:.9rem; line-height:1.45; margin-bottom:.6rem; }
.rl-text.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.82rem; }
.rl-stage { margin:.35rem 0 .15rem 0; color: var(--rl-ink); font-weight:600; font-size:.92rem; }
div[data-testid="stMetricValue"] { font-size: 1.65rem; }
div[data-testid="stMetricLabel"] p { color: var(--rl-ink2); }
</style>
"""

PAGES = {
    "Dashboard": dashboard.render,
    "Data Ingestion": ingestion.render,
    "Preprocessing & Features": preprocessing.render,
    "Text Mining Hub": mining.render,
    "Recommendations": recommend.render,
    "Performance Analytics": performance.render,
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown("<p class='rl-brand'>Review<span>Lens</span></p>"
                    "<p class='rl-tag'>Text Mining & Document Intelligence</p>", unsafe_allow_html=True)
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        corpus = common.corpus()
        prep, feat, dedup, mc = common.configs()
        st.caption("ACTIVE CORPUS")
        st.markdown(f"**{len(corpus):,} documents** · {len(set(corpus.labels or [])) or 'no'} categories  \n"
                    f"<span class='rl-note'>{st.session_state.corpus_source}</span>", unsafe_allow_html=True)
        st.caption("PIPELINE")
        st.markdown(f"<span class='rl-note'>{prep.describe()}<br>min_df={feat.min_df}, tf={feat.tf_scheme}, "
                    f"idf={feat.idf_scheme}, L2-normalised<br>duplicates: {dedup.policy.replace('_', ' ')} "
                    f"(cos ≥ {dedup.threshold:.2f})</span>", unsafe_allow_html=True)
        st.caption("MODELS")
        k = mc.k or len(set(corpus.labels or [])) or 2
        st.markdown(f"<span class='rl-note'>Naive Bayes α={mc.alpha}, test={int(mc.test_size * 100)}%<br>"
                    f"K-Means k={k}, {mc.init}, {mc.metric}, {mc.n_init} restarts<br>"
                    f"Top-K recommender K={mc.top_k}</span>", unsafe_allow_html=True)
        st.divider()
        st.caption("All TF-IDF, Naive Bayes, K-Means, Purity and Rand Index code is implemented manually "
                   "in the `textmining` package.")
    return page


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    common.init_state()
    page = sidebar()
    PAGES[page]()


main()
