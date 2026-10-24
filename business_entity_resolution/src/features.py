"""Stage 2 features for (S1, candidate) pairs.

Only relative similarities are used (edit distances, token overlaps, n-gram Jaccard, digit
overlap, length ratios, script indicators, blocking-context statistics). No vocabulary,
city/state lists or country identity enter the model, so it transfers to unseen countries.

Missing address on either side -> address features are NaN (LightGBM handles NaN natively).
"""
from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import BLK_FEATURES
from .utils import parallel_map

NAN = float("nan")

NAME_FEATURES = [
    "nm_ratio", "nm_token_sort", "nm_token_set", "nm_partial", "nm_jw", "nm_lev",
    "nm_sq_ratio", "nm_sq_partial", "nm_tok_jacc", "nm_tok_overlap", "nm_cov1", "nm_cov2",
    "nm_fz_cov1", "nm_fz_cov2", "nm_c3_jacc", "nm_first_jw", "nm_longest_sim",
    "nm_len_ratio", "nm_ntok_diff", "nm_ntok_ratio", "nm_prefix", "nm_skel_ratio",
    "nm_skel_partial", "nm_exact", "nm_contains", "nm_nl1", "nm_nl2",
    "sfx_equal", "sfx_both", "sfx_conflict", "sfx_has1", "sfx_has2",
]
ADDR_FEATURES = [
    "ad_missing1", "ad_missing2", "ad_ratio", "ad_token_sort", "ad_token_set", "ad_partial",
    "ad_tok_jacc", "ad_tok_overlap", "ad_cov1", "ad_cov2", "ad_c3_jacc", "ad_len_ratio",
    "ad_ntok_diff", "ad_abbr_hits", "ad_alpha_jacc", "ad_alpha_fz_cov1", "ad_alpha_fz_cov2",
    "num_jacc", "num_overlap", "num_equal", "num_any", "num_n1", "num_n2",
]
CTX_FEATURES = ["ctx_nm_gap", "ctx_sq_gap", "ctx_ad_gap", "ctx_n_high_nm", "ctx_nm_rank"]
PAIR_FEATURES = NAME_FEATURES + ADDR_FEATURES
FEATURE_NAMES = BLK_FEATURES + PAIR_FEATURES + CTX_FEATURES


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _grams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jacc(a: set, b: set) -> float:
    if not a or not b:
        return NAN
    return len(a & b) / len(a | b)


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return NAN
    return len(a & b) / min(len(a), len(b))


def _fz_cover(t1: List[str], t2: List[str], cutoff: int = 85) -> Tuple[float, float]:
    """Fraction of tokens on each side that have a >=cutoff fuzzy partner on the other."""
    if not t1 or not t2:
        return NAN, NAN
    m = process.cdist(t1, t2, scorer=fuzz.ratio, dtype=np.uint8)
    return float((m.max(axis=1) >= cutoff).mean()), float((m.max(axis=0) >= cutoff).mean())


def _is_subseq(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(c in it for c in short)


def _abbr_hits(t1: List[str], t2: List[str]) -> float:
    """Short alphabetic tokens (2-3 chars) on one side that abbreviate tokens on the other
    (subsequence of a longer token, or initials of 2-3 consecutive tokens), e.g. 'tn' ~ 'tamil nadu'.
    Purely structural; no abbreviation dictionary."""
    s1, s2 = set(t1), set(t2)
    hits = 0
    for a, b, sb in ((t1, t2, s2), (t2, t1, s1)):
        longs = [x for x in b if len(x) >= 4 and x.isalpha()]
        initials = set()
        for n in (2, 3):
            for i in range(len(b) - n + 1):
                initials.add("".join(w[0] for w in b[i:i + n]))
        for t in a:
            if 2 <= len(t) <= 3 and t.isalpha() and t not in sb:
                if t in initials or any(u[0] == t[0] and _is_subseq(t, u) for u in longs):
                    hits += 1
    return float(min(hits, 3))


# --------------------------------------------------------------------------- #
# one pair
# --------------------------------------------------------------------------- #
def _pair_row(n1, n2, sf1, sf2, k1, k2, a1, a2, l1, l2) -> List[float]:
    f: List[float] = []
    # ---------------- name ----------------
    t1, t2 = n1.split(), n2.split()
    if n1 and n2:
        ts1, ts2 = " ".join(sorted(t1)), " ".join(sorted(t2))
        sq1, sq2 = "".join(sorted(t1)), "".join(sorted(t2))
        set1, set2 = set(t1), set(t2)
        c1, c2 = _grams(sq1), _grams(sq2)
        cov1 = len(set1 & set2) / len(set1)
        cov2 = len(set1 & set2) / len(set2)
        fz1, fz2 = _fz_cover(t1, t2)
        lg1, lg2 = max(t1, key=len), max(t2, key=len)
        pre = 0
        for x, y in zip(sq1, sq2):
            if x != y:
                break
            pre += 1
        f += [
            fuzz.ratio(n1, n2) / 100, fuzz.token_sort_ratio(n1, n2) / 100, fuzz.token_set_ratio(n1, n2) / 100,
            fuzz.partial_ratio(n1, n2) / 100, JaroWinkler.similarity(ts1, ts2), Levenshtein.normalized_similarity(ts1, ts2),
            fuzz.ratio(sq1, sq2) / 100, fuzz.partial_ratio(sq1, sq2) / 100,
            len(set1 & set2) / len(set1 | set2), len(set1 & set2) / min(len(set1), len(set2)), cov1, cov2,
            fz1, fz2, _jacc(c1, c2), JaroWinkler.similarity(t1[0], t2[0]), fuzz.ratio(lg1, lg2) / 100,
            min(len(sq1), len(sq2)) / max(len(sq1), len(sq2)), float(abs(len(t1) - len(t2))),
            min(len(t1), len(t2)) / max(len(t1), len(t2)), pre / min(len(sq1), len(sq2)),
        ]
        if k1 and k2:
            f += [fuzz.ratio(k1, k2) / 100, fuzz.partial_ratio(k1, k2) / 100]
        else:
            f += [NAN, NAN]
        f += [float(n1 == n2), float(sq1 == sq2 or (len(sq1) >= 4 and len(sq2) >= 4 and (sq1 in sq2 or sq2 in sq1)))]
    else:
        f += [NAN] * 25
    f += [l1, l2]
    # legal suffix signature
    s1s, s2s = set(sf1.split()), set(sf2.split())
    both = bool(s1s) and bool(s2s)
    f += [float(s1s == s2s), float(both), float(both and not (s1s & s2s)), float(bool(s1s)), float(bool(s2s))]

    # ---------------- address ----------------
    m1, m2 = not a1, not a2
    f += [float(m1), float(m2)]
    if m1 or m2:
        f += [NAN] * (len(ADDR_FEATURES) - 2)
        return f
    u1, u2 = a1.split(), a2.split()
    st1, st2 = set(u1), set(u2)
    sa1, sa2 = " ".join(sorted(u1)), " ".join(sorted(u2))
    al1, al2 = [t for t in u1 if not t.isdigit()], [t for t in u2 if not t.isdigit()]
    nu1, nu2 = {t for t in u1 if any(ch.isdigit() for ch in t)}, {t for t in u2 if any(ch.isdigit() for ch in t)}
    afz1, afz2 = _fz_cover(al1, al2)
    inter = len(st1 & st2)
    f += [
        fuzz.ratio(a1, a2) / 100, fuzz.token_sort_ratio(a1, a2) / 100, fuzz.token_set_ratio(a1, a2) / 100,
        fuzz.partial_ratio(a1, a2) / 100, inter / len(st1 | st2), inter / min(len(st1), len(st2)),
        inter / len(st1), inter / len(st2), _jacc(_grams(sa1), _grams(sa2)),
        min(len(a1), len(a2)) / max(len(a1), len(a2)), float(abs(len(u1) - len(u2))),
        _abbr_hits(al1, al2), _jacc(set(al1), set(al2)), afz1, afz2,
    ]
    # digit / number overlap
    if nu1 and nu2:
        f += [_jacc(nu1, nu2), _overlap(nu1, nu2), float(nu1 == nu2), float(bool(nu1 & nu2))]
    else:
        f += [NAN, NAN, NAN, NAN]
    f += [float(len(nu1)), float(len(nu2))]
    return f


def _feature_chunk(args) -> np.ndarray:
    cols = args
    rows = [_pair_row(*r) for r in zip(*cols)]
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(PAIR_FEATURES))


# --------------------------------------------------------------------------- #
# group context (within one S1's candidate list)
# --------------------------------------------------------------------------- #
def _group_starts(s1_idx: np.ndarray) -> np.ndarray:
    if len(s1_idx) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([[0], np.flatnonzero(np.diff(s1_idx) != 0) + 1])


def _ctx_features(pair_x: np.ndarray, s1_idx: np.ndarray) -> np.ndarray:
    """Gap to the best candidate of the same S1, computed on name/address similarity."""
    n = len(s1_idx)
    out = np.full((n, len(CTX_FEATURES)), np.nan, dtype=np.float32)
    if n == 0:
        return out
    starts = _group_starts(s1_idx)
    sizes = np.diff(np.concatenate([starts, [n]]))
    gid = np.repeat(np.arange(len(starts)), sizes)

    def gap(col: str) -> np.ndarray:
        v = pair_x[:, PAIR_FEATURES.index(col)]
        vv = np.nan_to_num(v, nan=-1.0)
        mx = np.maximum.reduceat(vv, starts)
        return np.where(np.isnan(v), np.nan, v - mx[gid])

    out[:, 0] = gap("nm_token_sort")
    out[:, 1] = gap("nm_sq_ratio")
    out[:, 2] = gap("ad_token_set")
    nm = np.nan_to_num(pair_x[:, PAIR_FEATURES.index("nm_token_sort")], nan=0.0)
    out[:, 3] = np.add.reduceat((nm >= 0.9).astype(np.float32), starts)[gid]
    # rank of the candidate's name similarity within its group (0 = best)
    order = np.lexsort((-nm, gid))
    rank = np.empty(n, dtype=np.float32)
    pos = np.arange(n) - np.repeat(starts, sizes)
    rank[order] = pos
    out[:, 4] = rank
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def iter_group_chunks(s1_idx: np.ndarray, max_pairs: int) -> Iterator[Tuple[int, int]]:
    """Yield [a, b) row ranges of a table sorted by s1_idx, never splitting an S1's group."""
    n = len(s1_idx)
    starts = np.concatenate([_group_starts(s1_idx), [n]])  # group boundaries incl. the end
    a = 0
    while a < n:
        # largest group boundary <= a + max_pairs; if none beyond a, take the next boundary
        j = int(np.searchsorted(starts, a + max_pairs, side="right")) - 1
        b = int(starts[j])
        if b <= a:
            b = int(starts[np.searchsorted(starts, a, side="right")])
        yield a, b
        a = b


def compute_features(
    pairs: pl.DataFrame,
    s1: pl.DataFrame,
    pool: pl.DataFrame,
    n_jobs: int = 1,
    task_size: int = 20_000,
) -> np.ndarray:
    """Feature matrix (n_pairs x len(FEATURE_NAMES)) for a pair table sorted by s1_idx.

    ``pairs`` must contain s1_idx, pool_idx and the BLK_FEATURES columns; groups of one S1
    must be contiguous (they are, after ``add_blocking_context``).
    """
    n = pairs.height
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    s1_idx = pairs["s1_idx"].to_numpy()
    pool_idx = pairs["pool_idx"].to_numpy()
    blk = pairs.select(BLK_FEATURES).to_numpy().astype(np.float32)

    s1_side = s1.select(["nm", "sfx", "skel", "ad", "nl"]).gather(s1_idx)
    pool_side = pool.select(["nm", "sfx", "skel", "ad", "nl"]).gather(pool_idx)

    def cols(a: int, b: int):
        l = lambda df, c: df[c][a:b].to_list()  # noqa: E731
        return (l(s1_side, "nm"), l(pool_side, "nm"), l(s1_side, "sfx"), l(pool_side, "sfx"),
                l(s1_side, "skel"), l(pool_side, "skel"), l(s1_side, "ad"), l(pool_side, "ad"),
                l(s1_side, "nl"), l(pool_side, "nl"))

    tasks = (cols(a, min(a + task_size, n)) for a in range(0, n, task_size))
    parts = list(parallel_map(_feature_chunk, tasks, n_jobs))
    pair_x = np.concatenate(parts, axis=0)
    ctx = _ctx_features(pair_x, s1_idx)
    return np.concatenate([blk, pair_x, ctx], axis=1)
