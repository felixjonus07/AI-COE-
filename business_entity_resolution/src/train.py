"""Train the LightGBM pair classifier and tune the decision threshold tau.

Flow
  1. load + normalise train split
  2. run blocking for ALL S1 entities (so competition/'rival' features have test-like statistics)
  3. label candidate pairs from the ground truth, report candidate recall (ceiling)
  4. sample S1 entities, split 80/20 GROUPED by source1 entity id
  5. features -> LightGBM (binary, scale_pos_weight ~1) -> validation probabilities
  6. hard vetoes, then sweep tau in [0.70, 0.98] maximising macro F0.5 on validation
     (singletons included, one-to-one conflict resolution applied exactly as at inference)
  7. save models/lgbm_er.pkl
"""
from __future__ import annotations

import gc
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import BlockingConfig, add_blocking_context, blocking_recall, generate_candidates
from .features import FEATURE_NAMES, compute_features, iter_group_chunks
from .pipeline import decide, macro_f05_selection, prepare_split
from .postprocess import apply_hard_vetoes, sweep_tau
from .utils import default_n_jobs, get_logger

log = get_logger()


@dataclass
class TrainConfig:
    data_dir: Path
    work_dir: Path
    model_path: Path
    n_jobs: int = 0
    dev_frac: float = 0.0            # >0: smoke-test on a small consistent sample
    dev_full_pool: bool = False      # with dev_frac: sample only S1, index the full S2/S3 pool
    n_train_s1: int = 300_000        # S1 entities used for the classifier (0 = all)
    val_frac: float = 0.2
    seed: int = 13
    scale_pos_weight: float = 1.0    # conservative: 1.0 - 1.5
    tau_min: float = 0.70
    tau_max: float = 0.98
    tau_step: float = 0.01
    one_to_one: bool = True
    num_boost_round: int = 3000
    early_stopping: int = 100
    optuna_trials: int = 0
    blocking: BlockingConfig = field(default_factory=BlockingConfig)


BASE_PARAMS = dict(
    objective="binary",
    metric=["binary_logloss", "average_precision"],
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
)


def build_matrix(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, n_jobs: int,
                 chunk_pairs: int = 1_000_000) -> np.ndarray:
    x = np.empty((pairs.height, len(FEATURE_NAMES)), dtype=np.float32)
    s1_idx = pairs["s1_idx"].to_numpy()
    for a, b in iter_group_chunks(s1_idx, chunk_pairs):
        x[a:b] = compute_features(pairs[a:b], s1, pool, n_jobs)
        log.info(f"[features] {b:,}/{pairs.height:,} pairs")
    return x


def _tune(x_tr, y_tr, x_va, y_va, base: dict, n_trials: int, seed: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(y_tr), size=min(len(y_tr), 1_500_000), replace=False)
    vi = rng.choice(len(y_va), size=min(len(y_va), 500_000), replace=False)
    dtr = lgb.Dataset(x_tr[ti], y_tr[ti], feature_name=FEATURE_NAMES, free_raw_data=False)
    dva = lgb.Dataset(x_va[vi], y_va[vi], reference=dtr, free_raw_data=False)

    def objective(trial: "optuna.Trial") -> float:
        p = dict(base)
        p.update(
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 20.0, log=True),
            metric="average_precision",
        )
        bst = lgb.train(p, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        return bst.best_score["valid_0"]["average_precision"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log.info(f"[optuna] best AP={study.best_value:.5f} params={study.best_params}")
    return study.best_params


def find_tau(val: pl.DataFrame, n_true: np.ndarray, val_ids: np.ndarray, cfg: TrainConfig) -> Tuple[float, List[Tuple[float, float]]]:
    """Macro F0.5 sweep. 1-to-1 assignment is applied at every threshold."""
    return sweep_tau(val, n_true, val_ids, cfg.tau_min, cfg.tau_max, cfg.tau_step, cfg.one_to_one)


def assemble_splits(cfg: TrainConfig) -> dict:
    """Blocking, grouped split, and feature matrices shared by training and tuning."""
    t0 = time.time()
    n_jobs = cfg.n_jobs or default_n_jobs()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_split(cfg.data_dir, "train", cfg.work_dir, n_jobs, cfg.dev_frac, cfg.seed, cfg.dev_full_pool)
    s1, pool, n2, gt_pairs, n_true = data["s1"], data["pool"], data["n2"], data["gt_pairs"], data["n_true"]

    # ---- blocking over ALL S1 (test-like competition statistics) ---------------------
    bcfg = cfg.blocking
    bcfg.n_jobs = bcfg.n_jobs or n_jobs
    pairs_cache = cfg.work_dir / f"train_pairs_dev{cfg.dev_frac}{'fp' if cfg.dev_full_pool else ''}.parquet"
    if pairs_cache.exists():
        log.info(f"[train] loading cached candidate pairs {pairs_cache.name}")
        pairs = pl.read_parquet(pairs_cache)
    else:
        raw_pairs = generate_candidates(s1, pool, n2, bcfg, cfg.work_dir)
        pairs = add_blocking_context(raw_pairs, n2)
        del raw_pairs
        pairs.write_parquet(pairs_cache)
    rec = blocking_recall(pairs, gt_pairs)
    log.info(f"[train] blocking: recall={rec['recall']:.4f} avg_candidates={rec['avg_candidates']:.1f} "
             f"true_pairs={int(rec['true_pairs']):,}")

    lab = gt_pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    pairs = (
        pairs.join(lab, on=["s1_idx", "pool_idx"], how="left", maintain_order="left")
        .with_columns(pl.col("label").fill_null(0))
        .sort(["s1_idx", "blk_score"], descending=[False, True])
    )

    # ---- grouped split by source1 entity ----------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    all_q = np.unique(pairs["s1_idx"].to_numpy())
    if cfg.n_train_s1 and len(all_q) > cfg.n_train_s1:
        all_q = rng.choice(all_q, size=cfg.n_train_s1, replace=False)
    # S1 entities with no candidates at all still count for the metric (they are in val 'eval' set)
    is_val = rng.random(len(all_q)) < cfg.val_frac
    tr_ids, va_ids = np.sort(all_q[~is_val]), np.sort(all_q[is_val])
    assign = np.full(s1.height, -1, dtype=np.int8)
    assign[tr_ids], assign[va_ids] = 0, 1
    pair_split = assign[pairs["s1_idx"].to_numpy()]
    tr_pairs, va_pairs = pairs.filter(pl.Series(pair_split == 0)), pairs.filter(pl.Series(pair_split == 1))
    del pairs
    log.info(f"[train] S1 train={len(tr_ids):,} val={len(va_ids):,}  pairs train={tr_pairs.height:,} "
             f"val={va_pairs.height:,}  pos-rate={tr_pairs['label'].mean():.3f}")

    # ---- features ------------------------------------------------------------------------
    x_tr = build_matrix(tr_pairs, s1, pool, n_jobs)
    y_tr = tr_pairs["label"].to_numpy().astype(np.float32)
    del tr_pairs
    gc.collect()
    x_va = build_matrix(va_pairs, s1, pool, n_jobs)
    y_va = va_pairs["label"].to_numpy().astype(np.float32)
    va_pairs = va_pairs.select(["s1_idx", "pool_idx", "label"])
    gc.collect()
    log.info(f"[train] feature matrices built  ({time.time() - t0:.0f}s elapsed)")
    return {
        "n_jobs": n_jobs, "s1": s1, "pool": pool, "n_true": n_true, "va_ids": va_ids,
        "va_pairs": va_pairs, "x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "y_va": y_va,
        "rec": rec, "bcfg": bcfg, "t0": t0,
    }


def run_train(cfg: TrainConfig) -> dict:
    bundle = assemble_splits(cfg)
    n_jobs = bundle["n_jobs"]
    s1, pool, n_true = bundle["s1"], bundle["pool"], bundle["n_true"]
    va_ids, va_pairs = bundle["va_ids"], bundle["va_pairs"]
    x_tr, y_tr, x_va, y_va = bundle["x_tr"], bundle["y_tr"], bundle["x_va"], bundle["y_va"]
    rec, bcfg, t0 = bundle["rec"], bundle["bcfg"], bundle["t0"]

    # ---- LightGBM ------------------------------------------------------------------------
    params = dict(BASE_PARAMS, scale_pos_weight=cfg.scale_pos_weight, num_threads=n_jobs, seed=cfg.seed)
    if cfg.optuna_trials > 0:
        params.update(_tune(x_tr, y_tr, x_va, y_va, params, cfg.optuna_trials, cfg.seed))
    dtr = lgb.Dataset(x_tr, y_tr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(x_va, y_va, reference=dtr)
    bst = lgb.train(
        params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dtr, dva], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(cfg.early_stopping, first_metric_only=True), lgb.log_evaluation(50)],
    )
    log.info(f"[train] best_iteration={bst.best_iteration}")

    # ---- tau -----------------------------------------------------------------------------
    va_pairs = va_pairs.with_columns(pl.Series("prob", bst.predict(x_va, num_iteration=bst.best_iteration,
                                                                   num_threads=n_jobs).astype(np.float32)))
    va_pairs = apply_hard_vetoes(va_pairs, s1, pool)
    tau, sweep = find_tau(va_pairs, n_true, va_ids, cfg)
    for t, f in sweep[::5] + ([sweep[-1]] if (len(sweep) - 1) % 5 else []):
        log.info(f"[tau] {t:.2f} -> macro F0.5 = {f:.5f}")
    if tau in (sweep[0][0], sweep[-1][0]):
        log.info(f"[tau] WARNING: optimum {tau:.2f} lies on the edge of the search range")
    f_best = dict(sweep)[tau]
    f_no_1to1 = macro_f05_selection(decide(va_pairs, tau, False), n_true, va_ids)
    oracle = macro_f05_selection(va_pairs.filter(pl.col("label") == 1).with_columns(pl.lit(1.0).alias("prob")),
                                 n_true, va_ids)
    log.info(f"[train] val macro F0.5 = {f_best:.5f} @ tau={tau:.2f} "
             f"(without one-to-one: {f_no_1to1:.5f}; blocking ceiling: {oracle:.5f})")

    imp = sorted(zip(FEATURE_NAMES, bst.feature_importance("gain")), key=lambda kv: -kv[1])[:15]
    log.info("[train] top features (gain): " + ", ".join(f"{n}={v:.0f}" for n, v in imp))

    artifact = dict(
        booster=bst, feature_names=FEATURE_NAMES, tau=float(tau), one_to_one=cfg.one_to_one,
        best_iteration=bst.best_iteration, params=params, blocking_cfg=bcfg.to_dict(),
        metrics=dict(val_macro_f05=f_best, val_no_one_to_one=f_no_1to1, blocking_ceiling=oracle,
                     blocking_recall=rec["recall"], avg_candidates=rec["avg_candidates"], tau_sweep=sweep),
    )
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.model_path, "wb") as fh:
        pickle.dump(artifact, fh)
    bst.save_model(str(cfg.model_path.with_suffix(".txt")), num_iteration=bst.best_iteration)
    log.info(f"[train] saved {cfg.model_path}  (total {time.time() - t0:.0f}s)")
    return artifact
