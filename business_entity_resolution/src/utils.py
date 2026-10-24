"""Shared helpers: TSV I/O, text normalisation, parallel map, metric.

Everything here is country-agnostic: no city/state/country lists, only generic
string rules (legal-suffix stripping, accent folding, abbreviation expansion) and
script-level transliteration derived from Unicode character names.
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]  # business_entity_resolution/
COLS = ["entity_id", "business_name", "business_address", "country"]

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
_T0 = time.time()


def get_logger(name: str = "er") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def default_n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------- #
# paths + TSV I/O
# --------------------------------------------------------------------------- #
def resolve_data_dir(arg: str | None = None) -> Path:
    """Find the folder that holds train/ and test/.

    Order: explicit arg, $ER_DATA_DIR, ./DB, <repo>/DB, <repo>/Dataset/student_resource/dataset.
    """
    cands: List[Path] = []
    if arg:
        cands.append(Path(arg))
    if os.environ.get("ER_DATA_DIR"):
        cands.append(Path(os.environ["ER_DATA_DIR"]))
    cands += [
        Path.cwd() / "DB",
        ROOT.parent / "DB",
        ROOT.parent / "Dataset" / "student_resource" / "dataset",
        Path.cwd() / "dataset",
    ]
    for base in (ROOT.parent, ROOT.parent.parent, Path.cwd()):
        if base.is_dir():
            cands.extend(base.glob("*/student_resource/dataset"))
            cands.extend(base.glob("*/*/student_resource/dataset"))
    for c in cands:
        if (c / "train").is_dir() or (c / "test").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate the data directory (needs train/ and test/). "
        "Pass --data-dir or set ER_DATA_DIR. Tried: " + ", ".join(map(str, cands))
    )


def read_tsv(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read a TSV as all-string columns. quote_char=None matters: names contain quotes."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema_length=0,
        columns=list(columns) if columns else None,
        missing_utf8_is_empty_string=True,
    )
    return df.with_columns(pl.all().fill_null(""))


def scan_tsv(path: str | Path) -> pl.LazyFrame:
    """Lazy / streaming reader for files that should not be fully materialised."""
    return pl.scan_csv(
        path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True
    )


def load_split(data_dir: Path, split: str) -> Dict[str, pl.DataFrame]:
    """Return {'s1','s2','s3'[, 'gt']} for split in {'train','test'}."""
    out = {}
    for k in (1, 2, 3):
        out[f"s{k}"] = read_tsv(data_dir / split / f"{split}_source{k}.tsv")
    gt = data_dir / split / f"{split}_ground_truth.tsv"
    if gt.exists():
        out["gt"] = read_tsv(gt)
    return out


def write_id_lists(path: str | Path, header: Sequence[str], s1_ids: Sequence[str], lists: Sequence[str]) -> None:
    """Write a 2-col TSV (id, comma-joined ids) with no quoting. Empty list -> empty cell."""
    df = pl.DataFrame({header[0]: list(s1_ids), header[1]: list(lists)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never")


# --------------------------------------------------------------------------- #
# parallel map (spawn: polars is not fork-safe)
# --------------------------------------------------------------------------- #
def parallel_map(
    fn: Callable,
    items: Iterable,
    n_jobs: int,
    max_inflight: int | None = None,
    initializer: Callable | None = None,
    initargs: tuple = (),
) -> Iterator:
    """Ordered map with bounded in-flight tasks (keeps memory flat for huge inputs)."""
    if n_jobs <= 1:
        if initializer:
            initializer(*initargs)
        for x in items:
            yield fn(x)
        return
    max_inflight = max_inflight or n_jobs * 2
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=initializer, initargs=initargs) as ex:
        it = iter(items)
        pending: deque = deque()
        for x in itertools.islice(it, max_inflight):
            pending.append(ex.submit(fn, x))
        while pending:
            res = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(fn, nxt))
            yield res


# --------------------------------------------------------------------------- #
# transliteration of Indic scripts (from Unicode names; no lookup tables of words)
# --------------------------------------------------------------------------- #
_VOWEL_LETTER = {
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e",
    "AI": "ai", "O": "o", "OO": "o", "AU": "au", "R": "ri", "L": "li", "RR": "ri", "LL": "li",
}
_CONS_SPECIAL = {"CA": "ch", "CHA": "ch", "SSA": "sh", "SHA": "sh", "JNYA": "gy", "KSSA": "ksh",
                 "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n", "LLA": "l",
                 "LLLA": "l", "RRA": "r", "NNNA": "n", "TRA": "tr"}
_SCHWA_DELETE = {"DEVANAGARI", "GUJARATI", "GURMUKHI", "BENGALI", "ORIYA", "ODIA"}


@lru_cache(maxsize=None)
def _indic_info(ch: str):
    """('C'|'V'|'S'|'H'|'M'|'D'|'X', latin, script). None when not an Indic char."""
    cp = ord(ch)
    if not (0x0900 <= cp <= 0x0DFF):
        return None
    name = unicodedata.name(ch, "")
    parts = name.split()
    if len(parts) < 2:
        return ("X", "", "")
    script = parts[0]
    if "DIGIT" in parts:
        return ("D", str(unicodedata.digit(ch, "")), script)
    if "VIRAMA" in parts or "PULLI" in parts:
        return ("H", "", script)
    if "VOWEL" in parts and "SIGN" in parts:
        last = [p for p in parts if p not in ("SHORT", "LONG", "CANDRA")][-1]
        return ("S", _VOWEL_LETTER.get(last, ""), script)
    if "SIGN" in parts:
        if parts[-1] in ("ANUSVARA", "CANDRABINDU", "ANUSVARA"):
            return ("M", "n", script)
        if parts[-1] == "VISARGA":
            return ("M", "h", script)
        return ("X", "", script)
    if "LETTER" in parts:
        last = [p for p in parts if p not in ("SHORT", "LONG", "CANDRA")][-1]
        if last in _VOWEL_LETTER:
            return ("V", _VOWEL_LETTER[last], script)
        if last in _CONS_SPECIAL:
            return ("C", _CONS_SPECIAL[last], script)
        lat = last.lower()
        if len(lat) > 1 and lat.endswith("a"):
            lat = lat[:-1]
        return ("C", lat, script)
    return ("X", "", script)


def transliterate_indic(s: str) -> str:
    """Rough phonetic Latin form of Indic-script text; other characters pass through."""
    if s.isascii():
        return s
    out: List[str] = []
    inherent = False  # True while the last emitted char is an un-modified inherent 'a'
    script = ""

    def flush_word():
        nonlocal inherent
        if inherent and script in _SCHWA_DELETE and out and out[-1].endswith("a"):
            out[-1] = out[-1][:-1]
        inherent = False

    for ch in s:
        info = _indic_info(ch)
        if info is None:
            flush_word()
            out.append(ch)
            continue
        kind, lat, script = info
        if kind == "C":
            out.append(lat + "a")
            inherent = True
        elif kind == "V":
            out.append(lat)
            inherent = False
        elif kind == "S":
            if inherent and out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1] + lat
            else:
                out.append(lat)
            inherent = False
        elif kind == "H":
            if inherent and out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
            inherent = False
        elif kind == "M":
            out.append(lat)
            inherent = False
        elif kind == "D":
            out.append(lat)
            inherent = False
    flush_word()
    return "".join(out)


_TRANS = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ı": "i"})


def strip_accents(s: str) -> str:
    if s.isascii():
        return s
    s = s.translate(_TRANS)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def nonlatin_fraction(s: str) -> float:
    """Fraction of alphabetic characters outside Latin scripts (script-agnostic signal)."""
    if s.isascii():
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) >= 0x250) / len(letters)


# --------------------------------------------------------------------------- #
# name normalisation
# --------------------------------------------------------------------------- #
_LEGAL_CANON = {
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp", "lllp": "llp", "lp": "lp", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "plc": "plc", "pc": "pc", "pllc": "pllc", "sarl": "sarl",
    "eurl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "sci": "sci", "snc": "snc", "scp": "scp",
    "selarl": "selarl", "gmbh": "gmbh", "opc": "opc", "pty": "pty",
}
_NAME_NOISE = {"shri", "sri", "shree", "smt", "mr", "mrs", "ms"}
_NAME_STOP = {"and", "et", "of", "the", "de", "du", "des", "le", "la", "les"}

_RE_DBA = re.compile(r"\b(?:d/?b/?a|doing business as|trading as|t/a)\b")
_RE_MS = re.compile(r"\bm/s\b")
_RE_DOTTED = re.compile(r"(?<![a-z0-9])(?:[a-z]\.){2,}[a-z]?")
_RE_DOMAIN = re.compile(r"\b([a-z0-9][a-z0-9\-]*)\.(?:co\.in|co\.uk|com|net|org|in|co|fr|us|io|biz|info)\b")
_RE_NONWORD = re.compile(r"[\W_]+", re.UNICODE)


def _base(s: str) -> str:
    """NFKC, lowercase, transliterate Indic, fold accents."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = transliterate_indic(s)
    s = strip_accents(s)
    return s.replace("'", "").replace("’", "").replace("&", " and ")


def normalize_name(raw: str) -> Tuple[str, str]:
    """Return (core_name, legal_suffix_signature).

    core_name: lowercase tokens in original order, de-duplicated, with legal suffixes,
               honorific wrappers, stop words and phone-number tokens removed.
    """
    s = _base(raw)
    parts = _RE_DBA.split(s)
    if len(parts) > 1 and parts[-1].strip():
        s = parts[-1]
    s = _RE_MS.sub(" ", s)
    s = _RE_DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _RE_DOMAIN.sub(r"\1", s)
    toks = [t for t in _RE_NONWORD.split(s) if t]
    toks = [t for t in toks if not (t.isdigit() and len(t) >= 7)]
    sfx = [_LEGAL_CANON[t] for t in toks if t in _LEGAL_CANON]
    core = [t for t in toks if t not in _LEGAL_CANON and t not in _NAME_NOISE and t not in _NAME_STOP]
    if not core:
        core = [t for t in toks if t not in _NAME_NOISE] or toks
    seen: Set[str] = set()
    ded = [t for t in core if not (t in seen or seen.add(t))]
    return " ".join(ded), " ".join(sorted(set(sfx)))


def sorted_token_key(core_name: str) -> str:
    """Token-sort generator: squashed alphabetically-sorted tokens (order/space invariant)."""
    return "".join(sorted(core_name.split()))


_SKEL_SUBS = (("ph", "f"), ("sh", "s"), ("ch", "C"), ("ck", "k"), ("c", "k"), ("q", "k"), ("w", "v"), ("z", "s"), ("x", "ks"))


def phonetic_skeleton(core_name: str) -> str:
    """Consonant skeleton to bridge transliteration/spelling variants across scripts."""
    out = []
    for tok in core_name.split():
        for a, b in _SKEL_SUBS:
            tok = tok.replace(a, b)
        tok = re.sub(r"[aeiouyh]", "", tok)
        tok = re.sub(r"(.)\1+", r"\1", tok)
        out.append(tok)
    return "".join(out)


# --------------------------------------------------------------------------- #
# address normalisation
# --------------------------------------------------------------------------- #
_ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "ter": "terrace", "sq": "square", "fl": "floor",
    "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite", "r": "rue", "imp": "impasse",
    "ctr": "center", "centre": "center", "mt": "mount",
}
_ADDR_DROP = {"no", "h", "hno", "unit", "apt", "apartment", "suite", "ste", "flat", "po", "box", "pmb", "and"}
_ORD = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
        "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th"}
_RE_POBOX = re.compile(r"\b(?:p\.? ?o\.? ?box|pmb)\s*#?\s*\d+")


def normalize_address(raw: str) -> str:
    s = _base(raw)
    s = _RE_POBOX.sub(" ", s)
    toks = [t for t in _RE_NONWORD.split(s) if t]
    out: List[str] = []
    seen: Set[str] = set()
    for t in toks:
        if t in _ADDR_DROP:
            continue
        t = _ADDR_ABBR.get(t, t)
        t = _ORD.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


def normalize_country(raw: str) -> str:
    return strip_accents(raw.strip().lower())


# --------------------------------------------------------------------------- #
# frame preparation (parallel)
# --------------------------------------------------------------------------- #
PREP_COLS = ["nm", "sfx", "nm_key", "skel", "ad", "nl"]


def _prep_chunk(args):
    names, addrs = args
    nm, sfx, key, skel, ad, nl = [], [], [], [], [], []
    for n, a in zip(names, addrs):
        core, sg = normalize_name(n)
        nm.append(core)
        sfx.append(sg)
        key.append(sorted_token_key(core))
        skel.append(phonetic_skeleton(core))
        ad.append(normalize_address(a))
        nl.append(nonlatin_fraction(n))
    return nm, sfx, key, skel, ad, nl


def prepare_frame(df: pl.DataFrame, n_jobs: int = 1, chunk: int = 50_000) -> pl.DataFrame:
    """Add normalised columns (nm, sfx, nm_key, skel, ad, nl, ctry) to a source frame."""
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    chunks = ((names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk))
    parts = list(parallel_map(_prep_chunk, chunks, n_jobs))
    cols = {c: [] for c in PREP_COLS}
    for p in parts:
        for c, v in zip(PREP_COLS, p):
            cols[c].extend(v)
    out = df.with_columns(
        [pl.Series(c, cols[c], dtype=pl.Float32 if c == "nl" else pl.String) for c in PREP_COLS]
        + [pl.col("country").str.strip_chars().str.to_lowercase().alias("ctry")]
    )
    return out


def load_or_prepare(df_loader: Callable[[], pl.DataFrame], cache: Path | None, n_jobs: int, log=None) -> pl.DataFrame:
    if cache is not None and cache.exists():
        if log:
            log.info(f"  cache hit {cache.name}")
        return pl.read_parquet(cache)
    df = prepare_frame(df_loader(), n_jobs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache)
    return df


# --------------------------------------------------------------------------- #
# metric: macro F_0.5 with singleton rules
# --------------------------------------------------------------------------- #
def f05_from_counts(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 from counts.

    F0.5 = 1.25*P*R / (0.25*P + R) = 1.25*tp / (0.25*n_true + n_pred).
    n_true == n_pred == 0 -> 1.0 (correct singleton);  n_true == 0 < n_pred -> 0.0 (tp=0 gives 0).
    """
    tp = np.asarray(tp, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    denom = 0.25 * n_true + n_pred
    f = np.divide(1.25 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, f)


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]]) -> float:
    """Exact macro-averaged F0.5 over every key of `truth` (missing pred == empty)."""
    scores = []
    for k, t in truth.items():
        p = pred.get(k, set())
        tp = len(p & t)
        scores.append(float(f05_from_counts(np.array([tp]), np.array([len(p)]), np.array([len(t)]))[0]))
    return float(np.mean(scores)) if scores else 0.0
