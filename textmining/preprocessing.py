"""
Configurable text-preprocessing pipeline.

Order of operations (each step can be switched on/off from the Streamlit UI):
    raw text -> markup/entity removal -> URL/e-mail removal -> contraction expansion -> lower-casing
             -> tokenisation -> number / short-token filtering -> stop-word removal
             -> stemming OR lemmatisation -> n-gram generation  => list of index terms

Tokenisation, stop-word removal and n-gram generation are implemented here by hand. Stemming uses NLTK's
Porter stemmer and lemmatisation uses NLTK WordNet when available; if NLTK or its WordNet data are missing
(for example on an offline machine) a small built-in fallback is used and the UI reports which one is active.
"""
from __future__ import annotations

import html
import re
import socket
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .stopwords import ENGLISH_STOPWORDS, NEGATIONS, REVIEW_STOPWORDS

TAG_RE = re.compile(r"<[^>]+>")                         # HTML / XML tags such as <br />
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)           # runs of letters/digits; punctuation & emoji are dropped
RAW_TOKEN_RE = re.compile(r"\S+")                        # "before preprocessing": plain whitespace split

CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b", re.I), "can not"),
    (re.compile(r"\bwon['’]t\b", re.I), "will not"),
    (re.compile(r"\bain['’]t\b", re.I), "is not"),
    (re.compile(r"n['’]t\b", re.I), " not"),
    (re.compile(r"['’]re\b", re.I), " are"),
    (re.compile(r"['’]ve\b", re.I), " have"),
    (re.compile(r"['’]ll\b", re.I), " will"),
    (re.compile(r"['’]d\b", re.I), " would"),
    (re.compile(r"['’]m\b", re.I), " am"),
    (re.compile(r"['’]s\b", re.I), ""),
]


@dataclass(frozen=True)
class PreprocessConfig:
    """All switches of the preprocessing pipeline (frozen so it can be used as a cache key)."""
    strip_markup: bool = True          # remove <br />, other tags and HTML entities (&amp; &#34;)
    remove_urls: bool = True
    expand_contractions: bool = True   # didn't -> did not (keeps the negation as its own token)
    lowercase: bool = True
    remove_numbers: bool = True        # drop tokens made only of digits (keeps "ps4", "mp3")
    min_token_len: int = 2
    remove_stopwords: bool = True
    keep_negations: bool = True        # keep not/no/never even when stop words are removed
    custom_stopwords: Tuple[str, ...] = REVIEW_STOPWORDS
    normalizer: str = "stem"           # "none" | "stem" (Porter) | "lemma" (WordNet)
    ngram_min: int = 1
    ngram_max: int = 1                 # 1 = unigrams; 2 adds bigrams such as "not work"

    def describe(self) -> str:
        parts = []
        if self.strip_markup:
            parts.append("strip markup")
        if self.lowercase:
            parts.append("lower-case")
        if self.remove_stopwords:
            parts.append("stop-words" + (" (keep negations)" if self.keep_negations else ""))
        parts.append({"none": "no stemming", "stem": "Porter stemming", "lemma": "lemmatisation"}[self.normalizer])
        parts.append({(1, 1): "unigrams", (1, 2): "uni+bigrams", (1, 3): "uni+bi+trigrams"}.get(
            (self.ngram_min, self.ngram_max), f"{self.ngram_min}-{self.ngram_max}-grams"))
        return ", ".join(parts)


# ------------------------------------------------------------------------------------------------ backends
def _simple_stem(word: str) -> str:
    """Fallback suffix stripper used only if NLTK is not installed."""
    for suffix in ("ational", "ization", "fulness", "ousness", "iveness", "ments", "ment", "ness", "ings", "ing",
                   "edly", "ies", "ied", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                stem += "i"
            return stem
    return word


_IRREGULAR = {"was": "be", "were": "be", "is": "be", "are": "be", "been": "be", "has": "have", "had": "have",
              "did": "do", "does": "do", "done": "do", "went": "go", "gone": "go", "bought": "buy", "made": "make",
              "got": "get", "gotten": "get", "took": "take", "taken": "take", "came": "come", "gave": "give",
              "given": "give", "found": "find", "kept": "keep", "left": "leave", "sent": "send", "broke": "break",
              "broken": "break", "children": "child", "men": "man", "women": "woman", "feet": "foot",
              "teeth": "tooth", "mice": "mouse", "geese": "goose"}


def _simple_lemmatize(word: str) -> str:
    """Fallback dictionary + plural rules lemmatiser used only if WordNet data are unavailable."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _internet_available(host: str = "raw.githubusercontent.com", timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


_BACKENDS: Dict[str, Tuple[Callable[[str], str], str]] = {}


def get_stemmer() -> Tuple[Callable[[str], str], str]:
    if "stem" not in _BACKENDS:
        try:
            from nltk.stem.porter import PorterStemmer
            stemmer = PorterStemmer()
            stemmer.stem("running")
            _BACKENDS["stem"] = (stemmer.stem, "NLTK PorterStemmer")
        except Exception:  # NLTK missing -> keep the app usable
            _BACKENDS["stem"] = (_simple_stem, "built-in suffix stripper (NLTK not installed)")
    return _BACKENDS["stem"]


def get_lemmatizer() -> Tuple[Callable[[str], str], str]:
    if "lemma" not in _BACKENDS:
        try:
            import nltk
            from nltk.corpus import wordnet
            from nltk.stem import WordNetLemmatizer
            try:
                wordnet.ensure_loaded()
            except LookupError:
                if not _internet_available():
                    raise
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                wordnet.ensure_loaded()
            lem = WordNetLemmatizer()

            def lemmatize(word: str) -> str:
                as_verb = lem.lemmatize(word, "v")          # running -> run, was -> be
                return as_verb if as_verb != word else lem.lemmatize(word, "n")  # batteries -> battery

            lemmatize("batteries")
            _BACKENDS["lemma"] = (lemmatize, "NLTK WordNet lemmatizer")
        except Exception:
            _BACKENDS["lemma"] = (_simple_lemmatize, "built-in rule-based lemmatizer (WordNet data unavailable)")
    return _BACKENDS["lemma"]


def backend_names() -> Dict[str, str]:
    return {"stemmer": get_stemmer()[1], "lemmatizer": get_lemmatizer()[1]}


# ------------------------------------------------------------------------------------------------ pipeline
def clean_text(text: str, cfg: PreprocessConfig) -> str:
    """Character-level cleaning before tokenisation."""
    if cfg.strip_markup:
        text = html.unescape(text)          # &amp; -> &, &#34; -> "
        text = TAG_RE.sub(" ", text)        # <br /> -> space
    if cfg.remove_urls:
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
    if cfg.expand_contractions:
        for pattern, repl in CONTRACTIONS:
            text = pattern.sub(repl, text)
    if cfg.lowercase:
        text = text.lower()
    return text


def tokenize(text: str, cfg: PreprocessConfig) -> List[str]:
    """Regex tokeniser: keeps runs of letters/digits, drops punctuation, emoji and very short tokens."""
    tokens = TOKEN_RE.findall(text)
    out = []
    for tok in tokens:
        if len(tok) < cfg.min_token_len:
            continue
        if cfg.remove_numbers and tok.isdigit():
            continue
        out.append(tok)
    return out


def stopword_set(cfg: PreprocessConfig) -> frozenset:
    stops = set(ENGLISH_STOPWORDS) | {w.strip().lower() for w in cfg.custom_stopwords if w.strip()}
    if cfg.keep_negations:
        stops -= NEGATIONS
    return frozenset(stops)


def make_ngrams(tokens: Sequence[str], n_min: int, n_max: int) -> List[str]:
    """Contiguous n-grams joined by a space, e.g. ['not', 'work'] -> 'not work'."""
    grams: List[str] = []
    for n in range(max(1, n_min), max(n_min, n_max) + 1):
        if n == 1:
            grams.extend(tokens)
        else:
            grams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return grams


class Preprocessor:
    """Applies a PreprocessConfig to documents; caches stems/lemmas per distinct token for speed."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.stops = stopword_set(cfg)
        if cfg.normalizer == "stem":
            self._norm_fn, self.backend = get_stemmer()
        elif cfg.normalizer == "lemma":
            self._norm_fn, self.backend = get_lemmatizer()
        else:
            self._norm_fn, self.backend = None, "none"
        self._memo: Dict[str, str] = {}

    def normalize(self, token: str) -> str:
        if self._norm_fn is None:
            return token
        hit = self._memo.get(token)
        if hit is None:
            hit = self._norm_fn(token)
            self._memo[token] = hit
        return hit

    def stages(self, text: str) -> Dict[str, List[str]]:
        """Intermediate output of each step (used by the 'pipeline preview' in the UI)."""
        cfg = self.cfg
        cleaned = clean_text(text, cfg)
        tokens = tokenize(cleaned, cfg)
        if cfg.remove_stopwords:
            no_stop = [t for t in tokens if t.lower() not in self.stops]
        else:
            no_stop = tokens
        normalized = [self.normalize(t) for t in no_stop]
        terms = make_ngrams(normalized, cfg.ngram_min, cfg.ngram_max)
        return {"raw tokens": RAW_TOKEN_RE.findall(text), "cleaned text": [cleaned], "tokens": tokens,
                "after stop-word removal": no_stop, "after stemming/lemmatisation": normalized, "index terms": terms}

    def __call__(self, text: str) -> List[str]:
        cfg = self.cfg
        tokens = tokenize(clean_text(text, cfg), cfg)
        if cfg.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stops]
        tokens = [self.normalize(t) for t in tokens]
        return make_ngrams(tokens, cfg.ngram_min, cfg.ngram_max)

    def process_corpus(self, texts: Sequence[str]) -> List[List[str]]:
        return [self(t) for t in texts]


def raw_tokens(text: str) -> List[str]:
    """Baseline 'before preprocessing' tokens: whitespace split, case and punctuation untouched."""
    return RAW_TOKEN_RE.findall(text)


def vocabulary_funnel(texts: Sequence[str], cfg: PreprocessConfig) -> List[Dict[str, object]]:
    """Vocabulary size / token count after each cumulative preprocessing step (evidence for Q1)."""
    pre = Preprocessor(cfg)
    rows = []

    def add(stage: str, docs_tokens: List[List[str]]):
        counts = Counter(t for doc in docs_tokens for t in doc)
        n_tokens = sum(counts.values())
        rows.append({"stage": stage, "vocabulary size": len(counts), "total tokens": n_tokens,
                     "type/token ratio": round(len(counts) / n_tokens, 4) if n_tokens else 0.0,
                     "hapax terms (freq = 1)": sum(1 for c in counts.values() if c == 1)})

    add("0. raw whitespace tokens", [raw_tokens(t) for t in texts])
    tok_cfg = PreprocessConfig(strip_markup=cfg.strip_markup, remove_urls=cfg.remove_urls,
                               expand_contractions=cfg.expand_contractions, lowercase=False,
                               remove_numbers=cfg.remove_numbers, min_token_len=cfg.min_token_len)
    tokenized = [tokenize(clean_text(t, tok_cfg), tok_cfg) for t in texts]
    add("1. cleaned + tokenised (case kept)", tokenized)
    lowered = [[t.lower() for t in doc] for doc in tokenized] if cfg.lowercase else tokenized
    add("2. + lower-casing" if cfg.lowercase else "2. (lower-casing off)", lowered)
    no_stop = [[t for t in doc if t.lower() not in pre.stops] for doc in lowered] if cfg.remove_stopwords else lowered
    add("3. + stop-word removal" if cfg.remove_stopwords else "3. (stop-word removal off)", no_stop)
    normed = [[pre.normalize(t) for t in doc] for doc in no_stop]
    label = {"none": "4. (no stemming / lemmatisation)", "stem": "4. + Porter stemming",
             "lemma": "4. + lemmatisation"}[cfg.normalizer]
    add(label, normed)
    add(f"5. + n-grams ({cfg.ngram_min}-{cfg.ngram_max})", [make_ngrams(d, cfg.ngram_min, cfg.ngram_max) for d in normed])
    return rows
