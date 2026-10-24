"""Test-time pipeline: blocking -> candidate_pairs.tsv -> features -> LightGBM -> matching_results.tsv"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .blocking import BlockingConfig, add_blocking_context, generate_candidates
from .pipeline import decide, id_lists, predict_pairs, prepare_split
from .postprocess import apply_hard_vetoes
from .utils import default_n_jobs, get_logger, write_id_lists

log = get_logger()

CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")


@dataclass
class InferConfig:
    data_dir: Path
    work_dir: Path
    model_path: Path
    output_dir: Path
    n_jobs: int = 0
    dev_frac: float = 0.0
    dev_full_pool: bool = False
    tau: float | None = None         # override the tau stored in the model artifact
    one_to_one: bool | None = None


def run_inference(cfg: InferConfig) -> None:
    t0 = time.time()
    n_jobs = cfg.n_jobs or default_n_jobs()
    with open(cfg.model_path, "rb") as fh:
        art = pickle.load(fh)
    tau = cfg.tau if cfg.tau is not None else art["tau"]
    one_to_one = art["one_to_one"] if cfg.one_to_one is None else cfg.one_to_one
    log.info(f"[infer] model={cfg.model_path.name} tau={tau:.2f} one_to_one={one_to_one}")

    data = prepare_split(cfg.data_dir, "test", cfg.work_dir, n_jobs, cfg.dev_frac, dev_full_pool=cfg.dev_full_pool)
    s1, pool, n2 = data["s1"], data["pool"], data["n2"]

    # ---- stage 1: blocking -> candidate_pairs.tsv -----------------------------------------
    bcfg = BlockingConfig(**art["blocking_cfg"])
    bcfg.n_jobs = n_jobs
    raw = generate_candidates(s1, pool, n2, bcfg, cfg.work_dir)
    pairs = add_blocking_context(raw, n2)
    del raw
    cand = id_lists(pairs, s1, pool)
    out_c = cfg.output_dir / "candidate_pairs.tsv"
    write_id_lists(out_c, CAND_HEADER, cand["entity_id"], cand["cid"])
    log.info(f"[infer] wrote {out_c}  ({pairs.height:,} candidate pairs, "
             f"{(cand['cid'] == '').sum():,} S1 entities without candidates)")

    # ---- stage 2: score candidates -> matching_results.tsv ---------------------------------
    probs = predict_pairs(art["booster"], pairs, s1, pool, n_jobs)
    pairs = pairs.with_columns(pl.Series("prob", probs))
    pairs = apply_hard_vetoes(pairs, s1, pool)
    sel = decide(pairs, tau, one_to_one)
    match = id_lists(sel, s1, pool)
    out_m = cfg.output_dir / "matching_results.tsv"
    write_id_lists(out_m, MATCH_HEADER, match["entity_id"], match["cid"])
    n_pos = (match["cid"] != "").sum()
    log.info(f"[infer] wrote {out_m}  ({sel.height:,} matched pairs; "
             f"{s1.height - n_pos:,} S1 entities predicted singleton)  total {time.time() - t0:.0f}s")
