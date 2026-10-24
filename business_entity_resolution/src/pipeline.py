"""Glue shared by training and inference: data prep, chunked scoring, decision rule, scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl

from .blocking import build_pool
from .features import compute_features, iter_group_chunks
from .utils import f05_from_counts, get_logger, load_split, prepare_frame

log = get_logger()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _explode_gt(gt: pl.DataFrame) -> pl.DataFrame:
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
    )


def _dev_subsample(data: Dict[str, pl.DataFrame], frac: float, seed: int, full_pool: bool = False) -> Dict[str, pl.DataFrame]:
    """Consistent small sample for smoke tests.

    full_pool=True samples only S1 and keeps the ENTIRE S2/S3 pool: realistic candidate
    density (and realistic difficulty), but indexing cost of the full data.

    train: sample S1 groups, keep all their matches, and sample the unmatched S2/S3 at the
           same rate (so candidate density stays roughly proportional).
    test : sample S1 / S2 / S3 independently at `frac` (no labels to keep consistent).
    """
    rng = np.random.default_rng(seed)
    s1 = data["s1"].filter(pl.Series(rng.random(data["s1"].height) < frac))
    out = {"s1": s1}
    if "gt" in data:
        gt = data["gt"].filter(pl.col("source1_entity_id").is_in(s1["entity_id"]))
        matched = _explode_gt(gt)["matched_entity_ids"]
        out["gt"] = gt
        for k in ("s2", "s3"):
            df = data[k]
            if full_pool:
                out[k] = df
                continue
            keep = df["entity_id"].is_in(matched) | pl.Series(rng.random(df.height) < frac)
            out[k] = df.filter(keep)
    else:
        for k in ("s2", "s3"):
            out[k] = data[k] if full_pool else data[k].filter(pl.Series(rng.random(data[k].height) < frac))
    return out


def prepare_split(
    data_dir: Path,
    split: str,
    work_dir: Path,
    n_jobs: int,
    dev_frac: float = 0.0,
    seed: int = 13,
    dev_full_pool: bool = False,
) -> dict:
    """Load + normalise a split. Returns dict(s1, pool, n2, [gt_pairs, n_true])."""
    tag = f"{split}" + (f"_dev{dev_frac}" + ("fp" if dev_full_pool else "") if dev_frac else "")
    cache = Path(work_dir) / "prepared"
    cache.mkdir(parents=True, exist_ok=True)
    files = {k: cache / f"{tag}_{k}.parquet" for k in ("s1", "s2", "s3")}
    if all(f.exists() for f in files.values()) and (split == "test" or (cache / f"{tag}_gt.parquet").exists()):
        log.info(f"[data] loading prepared '{tag}' from cache {cache}")
        frames = {k: pl.read_parquet(f) for k, f in files.items()}
        if split == "train":
            frames["gt"] = pl.read_parquet(cache / f"{tag}_gt.parquet")
    else:
        log.info(f"[data] reading raw '{split}' TSVs from {data_dir / split}")
        raw = load_split(data_dir, split)
        if dev_frac:
            raw = _dev_subsample(raw, dev_frac, seed, dev_full_pool)
        frames = {}
        for k in ("s1", "s2", "s3"):
            log.info(f"[data] normalising {k}: {raw[k].height:,} rows")
            # raw text is not needed after normalisation; dropping it roughly halves RAM
            frames[k] = prepare_frame(raw[k], n_jobs).drop(["business_name", "business_address", "country"])
            frames[k].write_parquet(files[k])
            raw[k] = None
        if "gt" in raw:
            frames["gt"] = raw["gt"]
            raw["gt"].write_parquet(cache / f"{tag}_gt.parquet")
        del raw

    s1 = frames["s1"]
    if s1["entity_id"].n_unique() != s1.height:
        log.info("[data] WARNING: duplicate S1 entity ids found")
    n2 = frames["s2"].height
    pool = build_pool(frames["s2"], frames["s3"])
    gt_frame = frames.get("gt")
    del frames
    out = {"s1": s1, "pool": pool, "n2": n2}
    if gt_frame is not None:
        g = _explode_gt(gt_frame)
        s1_map = s1.select("entity_id").with_row_index("s1_idx")
        pool_map = pool.select("entity_id").with_row_index("pool_idx")
        gp = (
            g.join(s1_map, left_on="source1_entity_id", right_on="entity_id", how="inner")
            .join(pool_map, left_on="matched_entity_ids", right_on="entity_id", how="inner")
            .select(pl.col("s1_idx").cast(pl.Int32), pl.col("pool_idx").cast(pl.Int32))
        )
        n_true = np.zeros(s1.height, dtype=np.int32)
        cnt = g.join(s1_map, left_on="source1_entity_id", right_on="entity_id").group_by("s1_idx").len()
        n_true[cnt["s1_idx"].to_numpy()] = cnt["len"].to_numpy()
        out["gt_pairs"], out["n_true"] = gp, n_true
    log.info(f"[data] S1={s1.height:,}  S2={n2:,}  S3={pool.height - n2:,}")
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def predict_pairs(booster, pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, n_jobs: int,
                  chunk_pairs: int = 2_000_000) -> np.ndarray:
    """Chunked feature extraction + prediction (chunks never split an S1's candidate group)."""
    probs = np.empty(pairs.height, dtype=np.float32)
    s1_idx = pairs["s1_idx"].to_numpy()
    for a, b in iter_group_chunks(s1_idx, chunk_pairs):
        x = compute_features(pairs[a:b], s1, pool, n_jobs)
        probs[a:b] = booster.predict(x, num_threads=n_jobs)
        log.info(f"[predict] {b:,}/{pairs.height:,} pairs")
    return probs


def decide(pairs: pl.DataFrame, tau: float, one_to_one: bool = True) -> pl.DataFrame:
    """Select pairs with prob >= tau.

    With one_to_one, each S2/S3 record is assigned to the S1 entity that scores it
    highest. The S1 entity that loses a contested candidate keeps its next candidate
    that is still free and above tau, or is left unmatched when none remain.
    """
    from .postprocess import resolve_one_to_one
    return resolve_one_to_one(pairs, tau, one_to_one)


def macro_f05_selection(sel: pl.DataFrame, n_true: np.ndarray, eval_s1: np.ndarray) -> float:
    """Macro F0.5 over `eval_s1` entities. `sel` needs s1_idx and a 0/1 `label` column."""
    n_pred = np.zeros(len(n_true), dtype=np.int64)
    tp = np.zeros(len(n_true), dtype=np.int64)
    if sel.height:
        g = sel.group_by("s1_idx").agg(pl.len().alias("n"), pl.col("label").sum().alias("tp"))
        n_pred[g["s1_idx"].to_numpy()] = g["n"].to_numpy()
        tp[g["s1_idx"].to_numpy()] = g["tp"].to_numpy()
    return float(f05_from_counts(tp[eval_s1], n_pred[eval_s1], n_true[eval_s1]).mean())


def id_lists(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """One row per S1 (in S1 file order): comma-joined candidate/matched ids ('' when none)."""
    if pairs.height:
        cid = pool["entity_id"].gather(pairs["pool_idx"].to_numpy())
        g = (
            pairs.select("s1_idx").with_columns(cid.alias("cid"))
            .group_by("s1_idx", maintain_order=True)
            .agg(pl.col("cid").unique(maintain_order=True).str.join(","))
        )
    else:
        g = pl.DataFrame(schema={"s1_idx": pl.UInt32, "cid": pl.String})
    base = s1.select("entity_id").with_row_index("s1_idx")
    out = base.join(g.with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="left", maintain_order="left")
    return out.select("entity_id", pl.col("cid").fill_null(""))
