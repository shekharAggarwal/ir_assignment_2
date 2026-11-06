"""
Data ingestion: CSV / JSON / JSON-lines / ZIP folder of .txt files -> Corpus.

The Corpus keeps the main text content and the metadata logically separate:
    corpus.texts     list of document texts (the only field that is preprocessed and indexed)
    corpus.metadata  DataFrame indexed by doc_id with title, label, date, source, ... (never indexed)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BUNDLED_JSON = os.path.join(DATA_DIR, "amazon_reviews_corpus.json")
BUNDLED_CSV = os.path.join(DATA_DIR, "amazon_reviews_corpus.csv")
BUNDLED_NAME = "Amazon Reviews 2023 - 6 product departments (bundled sample)"


@dataclass
class Corpus:
    name: str
    doc_ids: List[str]
    texts: List[str]
    metadata: pd.DataFrame                 # index = doc_id, no text column
    label_field: Optional[str] = None
    title_field: Optional[str] = None
    source_format: str = ""

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def labels(self) -> Optional[List[str]]:
        if not self.label_field or self.label_field not in self.metadata.columns:
            return None
        return [str(v) for v in self.metadata[self.label_field].tolist()]

    @property
    def titles(self) -> List[str]:
        if self.title_field and self.title_field in self.metadata.columns:
            return [str(t) if str(t).strip() else self.texts[i][:60] for i, t in
                    enumerate(self.metadata[self.title_field].tolist())]
        return [t[:60].replace("\n", " ") + ("..." if len(t) > 60 else "") for t in self.texts]

    def subset(self, indices: Sequence[int]) -> "Corpus":
        idx = list(indices)
        return Corpus(self.name, [self.doc_ids[i] for i in idx], [self.texts[i] for i in idx],
                      self.metadata.iloc[idx].copy(), self.label_field, self.title_field, self.source_format)

    def fingerprint(self) -> str:
        h = hashlib.md5()
        for d, t in zip(self.doc_ids, self.texts):
            h.update(d.encode("utf-8"))
            h.update(t.encode("utf-8"))
        h.update(str(self.label_field).encode())
        if self.labels:
            h.update("|".join(self.labels).encode("utf-8"))
        return h.hexdigest()


# ------------------------------------------------------------------------------------------------ readers
def _flatten_json_records(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for k, v in r.items():
            if isinstance(v, dict):          # nested {"metadata": {...}} -> top-level columns
                row.update(v)
        rows.append(row)
    return pd.DataFrame(rows)


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV / JSON / JSONL file into a DataFrame (all columns as strings or numbers)."""
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), keep_default_na=False)
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return _flatten_json_records(records)
    if name.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict):
            for key in ("documents", "data", "records", "items"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
            else:
                obj = [obj]
        return _flatten_json_records(obj)
    raise ValueError("Unsupported file type - please upload .csv, .json, .jsonl or a .zip of .txt files")


def read_txt_zip(data: bytes) -> pd.DataFrame:
    """ZIP of a folder of .txt files. The parent folder of each file is used as its category label."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt") or "__MACOSX" in info.filename:
                continue
            raw = zf.read(info)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            parts = [p for p in info.filename.split("/") if p]
            label = parts[-2] if len(parts) >= 2 else "unlabelled"
            rows.append({"doc_id": "/".join(parts), "text": text, "category": label,
                         "title": os.path.splitext(parts[-1])[0], "source": "uploaded zip"})
    if not rows:
        raise ValueError("No .txt files found in the ZIP archive")
    return pd.DataFrame(rows)


def guess_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def build_corpus(df: pd.DataFrame, text_col: str, id_col: Optional[str] = None, label_col: Optional[str] = None,
                 title_col: Optional[str] = None, name: str = "uploaded dataset",
                 source_format: str = "") -> Tuple[Corpus, Dict[str, int]]:
    """Validate a table and split it into text + metadata. Returns (corpus, ingestion log counts)."""
    log = {"rows read": int(len(df))}
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    empty = df[text_col].str.strip().eq("")
    log["rows dropped (empty text)"] = int(empty.sum())
    df = df[~empty].reset_index(drop=True)
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).str.strip().tolist()
    else:
        ids = [f"D{i + 1:05d}" for i in range(len(df))]
        log["ids generated"] = len(ids)
    seen: Dict[str, int] = {}
    unique_ids = []
    renamed = 0
    for d in ids:
        if d in seen:
            seen[d] += 1
            unique_ids.append(f"{d}_{seen[d]}")
            renamed += 1
        else:
            seen[d] = 0
            unique_ids.append(d)
    log["duplicate ids renamed"] = renamed
    meta_cols = [c for c in df.columns if c not in (text_col, id_col)]
    meta = df[meta_cols].copy()
    meta.index = pd.Index(unique_ids, name="doc_id")
    if label_col and label_col in meta.columns:
        meta[label_col] = meta[label_col].astype(str).replace({"": "unlabelled", "nan": "unlabelled"})
    log["documents loaded"] = len(unique_ids)
    corpus = Corpus(name=name, doc_ids=unique_ids, texts=df[text_col].tolist(), metadata=meta,
                    label_field=label_col if label_col in meta.columns else None,
                    title_field=title_col if title_col in meta.columns else None, source_format=source_format)
    return corpus, log


def load_bundled(fmt: str = "json") -> Tuple[Corpus, Dict[str, int]]:
    if fmt == "csv":
        df = pd.read_csv(BUNDLED_CSV, keep_default_na=False)
        source = "CSV"
    else:
        with open(BUNDLED_JSON, encoding="utf-8") as fh:
            df = _flatten_json_records(json.load(fh))
        source = "JSON (text and metadata stored separately)"
    return build_corpus(df, text_col="text", id_col="doc_id", label_col="category", title_col="title",
                        name=BUNDLED_NAME, source_format=source)


# ------------------------------------------------------------------------------------------------ statistics
_WS = re.compile(r"\S+")


def corpus_statistics(corpus: Corpus) -> Dict[str, object]:
    lengths = np.array([len(_WS.findall(t)) for t in corpus.texts])
    chars = np.array([len(t) for t in corpus.texts])
    vocab = set()
    for t in corpus.texts:
        vocab.update(_WS.findall(t))
    labels = corpus.labels
    stats: Dict[str, object] = {
        "documents": len(corpus),
        "categories": len(set(labels)) if labels else 0,
        "raw text size (MB)": round(float(chars.sum()) / 1e6, 3),
        "total raw tokens": int(lengths.sum()),
        "raw vocabulary (whitespace tokens)": len(vocab),
        "mean tokens / doc": round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        "median tokens / doc": float(np.median(lengths)) if len(lengths) else 0.0,
        "min tokens / doc": int(lengths.min()) if len(lengths) else 0,
        "max tokens / doc": int(lengths.max()) if len(lengths) else 0,
        "metadata fields": len(corpus.metadata.columns),
    }
    if "date" in corpus.metadata.columns:
        dates = pd.to_datetime(corpus.metadata["date"], errors="coerce").dropna()
        if len(dates):
            stats["date range"] = f"{dates.min().date()} to {dates.max().date()}"
    return stats


def noise_profile(corpus: Corpus) -> pd.DataFrame:
    """How much realistic noise the raw text contains (share of documents with each pattern)."""
    patterns = {
        "HTML markup (<br />, tags)": r"<[^>]+>",
        "HTML entities (&amp; &#34;)": r"&[#a-zA-Z0-9]+;",
        "ALL-CAPS words (3+ letters)": r"\b[A-Z]{3,}\b",
        "repeated punctuation (!!, ??, ...)": r"[!?]{2,}|\.{3,}",
        "digits / model numbers": r"\d",
        "non-ASCII (emoji, curly quotes)": r"[^\x00-\x7F]",
        "URLs": r"https?://|www\.",
    }
    s = pd.Series(corpus.texts)
    rows = [{"noise pattern": k, "documents": int(s.str.contains(p, regex=True).sum()),
             "share of corpus": float(s.str.contains(p, regex=True).mean())} for k, p in patterns.items()]
    return pd.DataFrame(rows)
