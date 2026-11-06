"""Shared helpers for the Streamlit views: state, caching, compatibility shims, chart styling."""
from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st

from textmining.experiments import DEFAULT_DEDUP, DEFAULT_FEAT, DEFAULT_MODEL, DEFAULT_PREP
from textmining.ingestion import Corpus, load_bundled
from textmining.pipeline import (ClassificationResult, ClusteringResult, DedupConfig, FeatureConfig, ModelConfig,
                                 PipelineResult, run_classification, run_clustering, run_pipeline,
                                 run_recommendation_eval)
from textmining.preprocessing import PreprocessConfig

APP_NAME = "ReviewLens"
APP_TAGLINE = "Text Mining & Document Intelligence for product-review analytics"

# ------------------------------------------------------------------------------------------------ palette
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b5b3ab"
SEQ_BLUE = [[0.0, "#f4f8fe"], [0.15, "#cde2fb"], [0.35, "#9ec5f4"], [0.55, "#5598e7"], [0.75, "#256abf"],
            [1.0, "#0d366b"]]
SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x", "cross", "star", "triangle-down",
           "pentagon", "hexagon", "star-diamond", "hourglass"]
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def color_map(values: Sequence[str]) -> Dict[str, str]:
    """Fixed-order categorical colours (sorted names -> slots). More than 8 values: the rest share 'other' grey."""
    uniq = sorted({str(v) for v in values})
    return {v: (SERIES[i] if i < len(SERIES) else OTHER) for i, v in enumerate(uniq)}


def cluster_color(k: int) -> str:
    return SERIES[k] if k < len(SERIES) else OTHER


# ------------------------------------------------------------------------------------------------ compat
def _wide_kwargs(fn) -> dict:
    """Full-width rendering across Streamlit versions (old: use_container_width, new: width='stretch')."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    w = params.get("width")
    if w is not None and isinstance(w.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


_PLOT_KW = _wide_kwargs(st.plotly_chart)
_DF_KW = _wide_kwargs(st.dataframe)
_BTN_KW = _wide_kwargs(st.button)
_PLOTLY_VERSION = tuple(int(p) for p in plotly.__version__.split(".")[:2] if p.isdigit())


def plot(fig: go.Figure, key: Optional[str] = None) -> None:
    kwargs = dict(_PLOT_KW)
    if key:
        kwargs["key"] = key
    st.plotly_chart(fig, theme=None, config={"displaylogo": False, "responsive": True}, **kwargs)


def table(df: pd.DataFrame, height: Optional[int] = None, column_config: Optional[dict] = None,
          hide_index: bool = True) -> None:
    kwargs = dict(_DF_KW)
    if height:
        kwargs["height"] = height
    if column_config:
        kwargs["column_config"] = column_config
    try:
        st.dataframe(df, hide_index=hide_index, **kwargs)
    except TypeError:
        st.dataframe(df, **kwargs)


def card():
    """Bordered container where supported (Streamlit >= 1.29), plain container otherwise."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def wide_button(label: str, key: Optional[str] = None, primary: bool = False, help: Optional[str] = None) -> bool:
    kwargs = {"key": key, "help": help}
    if primary:
        kwargs["type"] = "primary"
    try:
        return st.button(label, **kwargs, **_BTN_KW)
    except TypeError:
        return st.button(label, key=key)


def chart_with_table(fig: go.Figure, df: pd.DataFrame, key: str, label: str = "Show data table") -> None:
    """Every chart has a table twin (accessibility: values never live only in colour / tooltips)."""
    plot(fig, key=key)
    with st.expander(label):
        table(df)


def number_column(fmt: str = "%.3f", label: Optional[str] = None):
    try:
        return st.column_config.NumberColumn(label=label, format=fmt)
    except AttributeError:
        return None


def progress_column(label: Optional[str] = None, max_value: float = 1.0, fmt: str = "%.3f"):
    try:
        return st.column_config.ProgressColumn(label=label, min_value=0.0, max_value=max_value, format=fmt)
    except AttributeError:
        return None


# ------------------------------------------------------------------------------------------------ figures
def style(fig: go.Figure, height: int = 360, title: Optional[str] = None, legend: bool = True,
          x_title: Optional[str] = None, y_title: Optional[str] = None, legend_bottom: bool = False) -> go.Figure:
    top = (34 if title else 12) + (30 if legend and not legend_bottom else 0)
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left", y=1, yanchor="top",
                   yref="container", pad=dict(t=8)) if title else None,
        margin=dict(l=10, r=16, t=top, b=10 if not (legend and legend_bottom) else 10),
        showlegend=legend,
        legend=(dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title_text="",
                     font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)") if legend_bottom else
                dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                     font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)")),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        bargap=0.35, bargroupgap=0.12,
    )
    if _PLOTLY_VERSION >= (5, 19):
        fig.update_layout(barcornerradius=4)
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2), zeroline=False, automargin=True,
                     title_text=x_title)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2), automargin=True, title_text=y_title)
    return fig


def bar(df: pd.DataFrame, x: str, y: str, color: str = SERIES[0], horizontal: bool = False, height: int = 340,
        title: Optional[str] = None, text_fmt: Optional[str] = None, x_title: Optional[str] = None,
        y_title: Optional[str] = None, hover_fmt: str = "%{y}") -> go.Figure:
    """Single-series bar chart: one colour for every bar (colour never re-encodes the length)."""
    if horizontal:
        trace = go.Bar(x=df[y], y=df[x], orientation="h", marker=dict(color=color, line=dict(width=0)),
                       hovertemplate="%{y}: %{x}<extra></extra>")
    else:
        trace = go.Bar(x=df[x], y=df[y], marker=dict(color=color, line=dict(width=0)),
                       hovertemplate="%{x}: " + hover_fmt + "<extra></extra>")
    if text_fmt:
        trace.update(texttemplate=text_fmt, textposition="outside", cliponaxis=False,
                     textfont=dict(color=INK_2, size=12))
    fig = go.Figure(trace)
    style(fig, height, title, legend=False, x_title=x_title, y_title=y_title)
    vmax = float(pd.to_numeric(df[y], errors="coerce").max() or 0)
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        if text_fmt and vmax > 0:
            fig.update_xaxes(range=[0, vmax * 1.18])
    elif text_fmt and vmax > 0:
        fig.update_yaxes(range=[0, vmax * 1.15])
    return fig


# ------------------------------------------------------------------------------------------------ state
def init_state() -> None:
    ss = st.session_state
    if "corpus" not in ss:
        corpus, log = load_bundled("json")
        ss.corpus, ss.ingest_log, ss.corpus_source = corpus, log, "bundled sample (JSON)"
    ss.setdefault("prep_cfg", DEFAULT_PREP)
    ss.setdefault("feat_cfg", DEFAULT_FEAT)
    ss.setdefault("dedup_cfg", DEFAULT_DEDUP)
    ss.setdefault("model_cfg", DEFAULT_MODEL)
    ss.setdefault("projection", "documents")
    ss.setdefault("experiments", {})


def corpus() -> Corpus:
    return st.session_state.corpus


def configs():
    ss = st.session_state
    return ss.prep_cfg, ss.feat_cfg, ss.dedup_cfg, ss.model_cfg


def set_model_cfg(**changes) -> None:
    st.session_state.model_cfg = replace(st.session_state.model_cfg, **changes)
    st.session_state.experiments = {}          # experiment results depend on the model settings


def pipeline_key() -> str:
    prep, feat, dedup, _ = configs()
    return f"{corpus().fingerprint()}|{prep!r}|{feat!r}|{dedup!r}"


# ------------------------------------------------------------------------------------------------ cached compute
@st.cache_resource(show_spinner=False, max_entries=3)
def _cached_pipeline(key: str, _corpus: Corpus, _prep: PreprocessConfig, _feat: FeatureConfig,
                     _dedup: DedupConfig) -> PipelineResult:
    return run_pipeline(_corpus, _prep, _feat, _dedup)


@st.cache_resource(show_spinner=False, max_entries=6)
def _cached_classification(key: str, _pr: PipelineResult, _mc: ModelConfig) -> ClassificationResult:
    return run_classification(_pr, _mc)


@st.cache_resource(show_spinner=False, max_entries=6)
def _cached_clustering(key: str, _pr: PipelineResult, _mc: ModelConfig, projection: str) -> ClusteringResult:
    return run_clustering(_pr, _mc, projection=projection)


@st.cache_resource(show_spinner=False, max_entries=6)
def _cached_rec_eval(key: str, _pr: PipelineResult, _mc: ModelConfig) -> Dict[str, float]:
    return run_recommendation_eval(_pr, _mc)


def pipeline() -> PipelineResult:
    prep, feat, dedup, _ = configs()
    with st.spinner("Preprocessing, building the inverted index and TF-IDF vectors ..."):
        return _cached_pipeline(pipeline_key(), corpus(), prep, feat, dedup)


def has_labels() -> bool:
    return bool(corpus().labels)


def classification() -> ClassificationResult:
    pr = pipeline()
    mc = st.session_state.model_cfg
    key = f"{pipeline_key()}|nb|{mc.test_size}|{mc.seed}|{mc.alpha}|{mc.group_split}"
    with st.spinner("Training the Naive Bayes classifier ..."):
        return _cached_classification(key, pr, mc)


def clustering() -> ClusteringResult:
    pr = pipeline()
    mc = st.session_state.model_cfg
    proj = st.session_state.projection
    key = f"{pipeline_key()}|km|{mc.k}|{mc.init}|{mc.metric}|{mc.n_init}|{mc.max_iter}|{mc.tol}|{mc.seed}"
    with st.spinner("Running K-Means ..."):
        return _cached_clustering(key, pr, mc, proj)


def rec_eval() -> Dict[str, float]:
    pr = pipeline()
    mc = st.session_state.model_cfg
    key = f"{pipeline_key()}|rec|{mc.top_k}|{mc.collapse_duplicates}"
    return _cached_rec_eval(key, pr, mc)


# ------------------------------------------------------------------------------------------------ misc
def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def flash(message: str) -> None:
    """Show a message after the next rerun (used after a settings form updates the configuration)."""
    st.session_state["_flash"] = message
    rerun()


def show_flash() -> None:
    msg = st.session_state.pop("_flash", None)
    if msg:
        st.success(msg)


def pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def f3(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='rl-header'><h2>{title}</h2><p>{subtitle}</p></div>", unsafe_allow_html=True)


def explain(title: str, body_md: str, latex: Optional[List[str]] = None, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.markdown(body_md)
        for formula in latex or []:
            st.latex(formula)


def text_block(text: str, mono: bool = True) -> None:
    """Wrapped, HTML-escaped text box (raw review text keeps its markup visible, e.g. <br />)."""
    import html as _html
    cls = "rl-text mono" if mono else "rl-text"
    st.markdown(f"<div class='{cls}'>{_html.escape(text)}</div>", unsafe_allow_html=True)


def snippet(text: str, n: int = 160) -> str:
    t = " ".join(text.replace("<br />", " ").split())
    return t if len(t) <= n else t[: n - 1] + "…"
