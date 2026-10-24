"""Stage 1: candidate generation (blocking).

Per country partition (exact match on the normalised country string; an open set of
labels, nothing hard-coded) every Source-1 record is scored against the pooled S2+S3
records with three sparse TF-IDF channels:

    score = 0.3*cos(name)  +  0.6*cos(address)  +  0.1*cos(phonetic skeleton)

* name    : char 3-grams of the *sorted, squashed* core-name tokens (inverted index over sorted
            name tokens; invariant to word order / spacing, tolerant to typos and ``brand.com``).
* address : word unigrams + adjacent bigrams. Addresses are far more reliable than names in this
            data (names are often garbled/aliased); bigrams make truncated addresses
            (``a 302 navi mumbai``) selective even though every single token is common.
* skeleton: char 4-grams of a consonant skeleton; bridges phonetic spellings and Indic->Latin
            transliteration (``lakshmi eksaports`` ~ ``laxmi exports``) and helps when the
            address is missing.

Very frequent n-grams/tokens are pruned (max_df) to bound posting-list length. Top-K per
(S1, source) are kept, so each S1 gets up to 2*K candidates.

Memory: each channel is vectorised, weighted, written to disk as memmap and freed; workers
share the matrices through the page cache and add the three per-channel products per chunk.

Candidate tables use positional indices: ``s1_idx`` into the S1 frame and ``pool_idx`` into
the pooled frame (S2 rows first, then S3 rows).

Measured on real train data (full-size pools, 400 queries/country, recall@15 per source):
original name-heavy 2-channel design 0.75 (India) / 0.88 (US)  ->  this design 0.91 / 0.96.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .utils import default_n_jobs, get_logger, parallel_map

log = get_logger()

# channel name -> (pool/S1 column, analyzer, ngram)
CHANNELS: Dict[str, Tuple[str, str, int]] = {
    "name": ("nm_key", "char", 3),
    "addr": ("ad", "word", 2),
    "skel": ("skel", "char", 4),
}


@dataclass
class BlockingConfig:
    top_k_per_source: int = 15       # -> up to 2*K = 30 candidates per S1
    w_name: float = 0.3
    w_addr: float = 0.6
    w_skel: float = 0.1
    hash_dim: int = 1 << 22
    max_df_frac_name: float = 0.003  # prune n-grams present in > this fraction of the pool ...
    max_df_frac_addr: float = 0.005
    max_df_frac_skel: float = 0.003
    max_df_floor: int = 1000         # ... but never below this many documents
    min_score: float = 0.10          # drop hopeless candidates
    chunk_size: int = 1000           # S1 rows per task
    flops_budget: int = 12_000_000   # bound on sparse-product size per sub-chunk (memory)
    vec_chunk: int = 200_000
    n_jobs: int = 0                  # 0 -> cpu_count-1

    def weights(self) -> Dict[str, float]:
        return {"name": self.w_name, "addr": self.w_addr, "skel": self.w_skel}

    def max_df(self) -> Dict[str, float]:
        return {"name": self.max_df_frac_name, "addr": self.max_df_frac_addr, "skel": self.max_df_frac_skel}

    def to_dict(self) -> dict:
        return asdict(self)


def build_pool(s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    """Pooled S2+S3 frame. pool_idx < len(s2) means S2, otherwise S3."""
    return pl.concat([s2, s3], how="vertical")


# --------------------------------------------------------------------------- #
# vectorisation
# --------------------------------------------------------------------------- #
def _vec_task(args) -> sp.csr_matrix:
    analyzer, ngram, docs, dim = args
    if analyzer == "char":
        hv = HashingVectorizer(analyzer="char", ngram_range=(ngram, ngram), lowercase=False, norm=None,
                               alternate_sign=False, n_features=dim, dtype=np.float32)
        return hv.transform(["^" + d + "$" for d in docs])
    hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, ngram), lowercase=False,
                           norm=None, alternate_sign=False, n_features=dim, dtype=np.float32)
    return hv.transform(docs)


def _vectorise(series: pl.Series, analyzer: str, ngram: int, cfg: BlockingConfig, n_jobs: int) -> sp.csr_matrix:
    """Vectorise a polars string Series in chunks (never materialises the full Python list)."""
    tasks = ((analyzer, ngram, series.slice(i, cfg.vec_chunk).to_list(), cfg.hash_dim)
             for i in range(0, len(series), cfg.vec_chunk))
    mats = list(parallel_map(_vec_task, tasks, n_jobs))
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


def _idf(pool_x: sp.csr_matrix, frac: float, floor: int) -> np.ndarray:
    n, f = pool_x.shape
    df = np.bincount(pool_x.indices, minlength=f)
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    max_df = max(frac * n, floor)
    idf[(df == 0) | (df > max_df)] = 0.0
    return idf


def _weight(x: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Sublinear tf * idf, dropped n-grams removed, rows L2-normalised (in place)."""
    x = x.tocsr()
    x.data = ((1.0 + np.log(x.data)) * idf[x.indices]).astype(np.float32)
    x.eliminate_zeros()
    return normalize(x, norm="l2", copy=False)


# --------------------------------------------------------------------------- #
# worker side
# --------------------------------------------------------------------------- #
_G: Dict[str, object] = {}
_CH = list(CHANNELS)


def _load_csr(d: Path, name: str, shape: Tuple[int, int]) -> sp.csr_matrix:
    data = np.load(d / f"{name}_data.npy", mmap_mode="r")
    idx = np.load(d / f"{name}_indices.npy", mmap_mode="r")
    ptr = np.load(d / f"{name}_indptr.npy", mmap_mode="r")
    return sp.csr_matrix((data, idx, ptr), shape=shape, copy=False)


def _save_csr(d: Path, name: str, m: sp.csr_matrix) -> None:
    m = m.tocsr()
    np.save(d / f"{name}_data.npy", m.data)
    np.save(d / f"{name}_indices.npy", m.indices)
    np.save(d / f"{name}_indptr.npy", m.indptr)


def _init_worker(d: str, shapes: dict, n2c: int, cfg_dict: dict) -> None:
    dd = Path(d)
    _G["P"] = {c: _load_csr(dd, f"p_{c}", shapes[f"p_{c}"]) for c in _CH}
    _G["PT"] = _load_csr(dd, "pt_all", shapes["pt_all"])
    _G["lens"] = np.diff(np.asarray(_G["PT"].indptr)).astype(np.int64)
    _G["n2c"] = n2c
    _G["cfg"] = cfg_dict


def _topk(c: sp.csr_matrix, n2c: int, k: int, min_score: float):
    ip, ind, dat = c.indptr, c.indices, c.data
    rows_o, cols_o, sc_o = [], [], []
    for i in range(c.shape[0]):
        s, e = ip[i], ip[i + 1]
        if s == e:
            continue
        cols, vals = ind[s:e], dat[s:e]
        keep = vals >= min_score
        if not keep.any():
            continue
        cols, vals = cols[keep], vals[keep]
        m2 = cols < n2c
        for mask in (m2, ~m2):
            cc, vv = cols[mask], vals[mask]
            if cc.size == 0:
                continue
            if cc.size > k:
                sel = np.argpartition(vv, -k)[-k:]
                cc, vv = cc[sel], vv[sel]
            rows_o.append(np.full(cc.size, i, dtype=np.int32))
            cols_o.append(cc.astype(np.int64))
            sc_o.append(vv)
    if not rows_o:
        e = np.empty(0)
        return e.astype(np.int32), e.astype(np.int64), e.astype(np.float32)
    return np.concatenate(rows_o), np.concatenate(cols_o), np.concatenate(sc_o).astype(np.float32)


def _block_task(args):
    qs, offset = args                       # qs: {channel: csr chunk}
    cfg = _G["cfg"]
    w = {"name": cfg["w_name"], "addr": cfg["w_addr"], "skel": cfg["w_skel"]}
    n2c, P, PT, lens = _G["n2c"], _G["P"], _G["PT"], _G["lens"]
    n_rows = qs["name"].shape[0]
    # weights go on the (small) query side, so ONE product over the stacked index yields
    # w_name*cos_name + w_addr*cos_addr + w_skel*cos_skel with no merging of sparse results
    qc = sp.hstack([qs[c] * np.float32(w[c]) for c in _CH], format="csr")
    # upper bound of intermediate products per row -> memory-bounded sub-chunks
    cs = np.concatenate([[0], np.cumsum(lens[qc.indices])])
    flops = cs[qc.indptr[1:]] - cs[qc.indptr[:-1]]
    bounds, start, acc = [], 0, 0
    for i, f in enumerate(flops):
        if acc + f > cfg["flops_budget"] and i > start:
            bounds.append((start, i))
            start, acc = i, 0
        acc += f
    bounds.append((start, n_rows))

    rows, cols, scs = [], [], []
    for a, b in bounds:
        if b <= a:
            continue
        tot = qc[a:b] @ PT
        r_, c_, s_ = _topk(tot, n2c, cfg["top_k_per_source"], cfg["min_score"])
        rows.append(r_ + a)
        cols.append(c_)
        scs.append(s_)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32),
             np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32))
    if not rows:
        return empty
    rows, cols, scs = np.concatenate(rows), np.concatenate(cols), np.concatenate(scs)
    cos = [np.asarray(qs[c][rows].multiply(P[c][cols]).sum(axis=1)).ravel().astype(np.float32) for c in _CH]
    return (rows.astype(np.int64) + offset, cols, scs, *cos)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def generate_candidates(
    s1: pl.DataFrame,
    pool: pl.DataFrame,
    n2: int,
    cfg: BlockingConfig,
    work_dir: Path,
    query_idx: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return raw candidate pairs (s1_idx, pool_idx, blk_score, blk_name_cos, blk_addr_cos, blk_skel_cos).

    s1/pool must carry the columns produced by ``utils.prepare_frame``.
    ``query_idx`` optionally restricts which S1 rows are queried (default: all).
    """
    n_jobs = cfg.n_jobs or default_n_jobs()
    tmp_root = Path(work_dir) / "blocking_tmp"
    s1_ctry = s1["ctry"].to_numpy()
    pool_ctry = pool["ctry"].to_numpy()
    all_q = np.arange(s1.height) if query_idx is None else np.asarray(query_idx)
    out_frames: List[pl.DataFrame] = []
    schema = {"s1_idx": pl.Int32, "pool_idx": pl.Int32, "blk_score": pl.Float32,
              "blk_name_cos": pl.Float32, "blk_addr_cos": pl.Float32, "blk_skel_cos": pl.Float32}

    for ci, ctry in enumerate(sorted(set(s1_ctry[all_q].tolist()))):
        q_idx = all_q[s1_ctry[all_q] == ctry]
        p_idx = np.flatnonzero(pool_ctry == ctry)
        if len(q_idx) == 0 or len(p_idx) == 0:
            log.info(f"[blocking] country={ctry!r}: {len(q_idx)} queries, {len(p_idx)} pool rows -> skipped")
            continue
        n2c = int((p_idx < n2).sum())
        log.info(f"[blocking] country={ctry!r}: {len(q_idx):,} S1 queries vs {len(p_idx):,} pool rows (S2={n2c:,})")

        tmp = tmp_root / f"c{ci}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        shapes: Dict[str, Tuple[int, int]] = {}
        Q: Dict[str, sp.csr_matrix] = {}
        max_df = cfg.max_df()
        for name, (col, analyzer, ngram) in CHANNELS.items():
            P = _vectorise(pool[col].gather(p_idx), analyzer, ngram, cfg, n_jobs)
            Qm = _vectorise(s1[col].gather(q_idx), analyzer, ngram, cfg, n_jobs)
            idf = _idf(P, max_df[name], cfg.max_df_floor)
            P, Qm = _weight(P, idf), _weight(Qm, idf)
            _save_csr(tmp, f"p_{name}", P)
            shapes[f"p_{name}"] = (len(p_idx), cfg.hash_dim)
            _save_csr(tmp, f"pt_{name}", P.T.tocsr())
            del P
            Q[name] = Qm
            log.info(f"[blocking]   channel '{name}' indexed")
        # stack the per-channel transposes along the feature axis -> one index for one product
        pt_all = sp.vstack([_load_csr(tmp, f"pt_{c}", (cfg.hash_dim, len(p_idx))) for c in CHANNELS], format="csr")
        _save_csr(tmp, "pt_all", pt_all)
        shapes["pt_all"] = pt_all.shape
        del pt_all
        for c in CHANNELS:
            for suffix in ("data", "indices", "indptr"):
                (tmp / f"pt_{c}_{suffix}.npy").unlink()

        tasks = (({c: Q[c][i:i + cfg.chunk_size] for c in CHANNELS}, i) for i in range(0, len(q_idx), cfg.chunk_size))
        parts = list(parallel_map(
            _block_task, tasks, n_jobs,
            initializer=_init_worker, initargs=(str(tmp), shapes, n2c, cfg.to_dict()),
        ))
        shutil.rmtree(tmp, ignore_errors=True)

        rows = np.concatenate([p[0] for p in parts])
        cols = np.concatenate([p[1] for p in parts])
        df = pl.DataFrame({
            "s1_idx": q_idx[rows].astype(np.int32),
            "pool_idx": p_idx[cols].astype(np.int32),
            "blk_score": np.concatenate([p[2] for p in parts]),
            "blk_name_cos": np.concatenate([p[3] for p in parts]),
            "blk_addr_cos": np.concatenate([p[4] for p in parts]),
            "blk_skel_cos": np.concatenate([p[5] for p in parts]),
        })
        log.info(f"[blocking] country={ctry!r}: {df.height:,} candidate pairs "
                 f"({df.height / max(1, len(q_idx)):.1f} per S1)")
        out_frames.append(df)

    if not out_frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(out_frames)


# --------------------------------------------------------------------------- #
# context features derived from the whole candidate table
# --------------------------------------------------------------------------- #
BLK_FEATURES = [
    "blk_score", "blk_name_cos", "blk_addr_cos", "blk_skel_cos", "blk_rank", "blk_gap_top", "blk_gap_src",
    "blk_n", "blk_n_close", "blk_claims", "blk_rival_max", "blk_rival_gap", "src_is_s3",
]


def add_blocking_context(pairs: pl.DataFrame, n2: int) -> pl.DataFrame:
    """Add rank / gap / competition features. Result sorted by (s1_idx, -blk_score).

    Competition features ('rival') look at OTHER S1 entities that also list the same S2/S3
    record as a candidate. Because each S2/S3 record truly belongs to at most one S1 entity,
    a strong rival is direct evidence against the pair. They must be computed on the FULL
    candidate table (all queried S1 entities) to match test-time statistics.
    """
    p = pairs.with_columns((pl.col("pool_idx") >= n2).cast(pl.Int8).alias("src_is_s3"))
    p = p.sort(["s1_idx", "blk_score"], descending=[False, True])
    p = p.with_columns(
        pl.col("blk_score").rank("ordinal", descending=True).over(["s1_idx", "src_is_s3"]).cast(pl.Int16).alias("blk_rank"),
        (pl.col("blk_score") - pl.col("blk_score").max().over("s1_idx")).alias("blk_gap_top"),
        (pl.col("blk_score") - pl.col("blk_score").max().over(["s1_idx", "src_is_s3"])).alias("blk_gap_src"),
        pl.len().over("s1_idx").cast(pl.Int16).alias("blk_n"),
        (pl.col("blk_name_cos") >= 0.8).sum().over("s1_idx").cast(pl.Int16).alias("blk_n_close"),
        pl.len().over("pool_idx").cast(pl.Int16).alias("blk_claims"),
        pl.col("blk_score").rank("ordinal", descending=True).over("pool_idx").cast(pl.Int16).alias("_prk"),
    )
    top = (
        p.filter(pl.col("_prk") <= 2)
        .group_by("pool_idx")
        .agg(
            pl.col("blk_score").filter(pl.col("_prk") == 1).first().alias("_top1"),
            pl.col("blk_score").filter(pl.col("_prk") == 2).first().alias("_top2"),
        )
    )
    p = p.join(top, on="pool_idx", how="left")
    p = p.with_columns(
        pl.when(pl.col("_prk") == 1).then(pl.col("_top2").fill_null(0.0)).otherwise(pl.col("_top1"))
        .cast(pl.Float32).alias("blk_rival_max")
    ).with_columns((pl.col("blk_score") - pl.col("blk_rival_max")).alias("blk_rival_gap"))
    return p.drop(["_prk", "_top1", "_top2"]).sort(["s1_idx", "blk_score"], descending=[False, True])


def blocking_recall(pairs: pl.DataFrame, gt_pairs: pl.DataFrame, s1_idx_subset: np.ndarray | None = None) -> Dict[str, float]:
    """Candidate recall (the ceiling for stage 2) and mean candidate count.

    gt_pairs: columns s1_idx, pool_idx for every true match.
    """
    g = gt_pairs
    if s1_idx_subset is not None:
        g = g.filter(pl.col("s1_idx").is_in(pl.Series(s1_idx_subset)))
    hit = g.join(pairs.select("s1_idx", "pool_idx"), on=["s1_idx", "pool_idx"], how="inner").height
    n_q = len(s1_idx_subset) if s1_idx_subset is not None else pairs["s1_idx"].n_unique()
    return {
        "true_pairs": float(g.height),
        "found_pairs": float(hit),
        "recall": hit / max(1, g.height),
        "avg_candidates": pairs.height / max(1, n_q),
    }
