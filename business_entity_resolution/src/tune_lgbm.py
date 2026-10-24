"""Memory-safe CPU search over LightGBM regularization, then a Macro F0.5 tau sweep.

The search never rebuilds histograms from Pandas or Polars. Feature matrices are
binned once with ``max_bin=127`` and written with ``Dataset.save_binary``; each
Optuna trial loads those files. Early stopping and the Optuna objective use
validation Macro F0.5 (singletons included, global 1-to-1 assignment, hard vetoes).

Search space
  lambda_l1          [0.5, 10.0]
  lambda_l2          [0.5, 10.0]
  feature_fraction   [0.65, 0.85]
  min_data_in_leaf   [50, 200]
  num_leaves         [63, 127]

    python -m src.tune_lgbm --trials 15
    python main.py --mode tune --optuna-trials 15
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blocking import BlockingConfig  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.pipeline import decide, macro_f05_selection  # noqa: E402
from src.postprocess import sweep_tau, sweep_tau_arrays, veto_mask  # noqa: E402
from src.train import TrainConfig, assemble_splits  # noqa: E402
from src.utils import get_logger, resolve_data_dir  # noqa: E402

log = get_logger()

MAX_BIN = 127


def _dataset_params() -> dict:
    return {"max_bin": MAX_BIN, "feature_pre_filter": False}


def _save_binary(x: np.ndarray, y: np.ndarray, path: Path, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """Bin once and persist. The caller still owns ``x`` until it deletes it."""
    ds = lgb.Dataset(
        x, label=y, reference=reference, feature_name=FEATURE_NAMES,
        params=_dataset_params(), free_raw_data=False,
    )
    ds.construct()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_binary(str(path))
    log.info(f"[tune] saved binary dataset {path.name}  rows={x.shape[0]:,}  cols={x.shape[1]}")
    return ds


def _load_binary(path: Path, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(str(path), reference=reference, params=_dataset_params(), free_raw_data=True)


def _predict(bst: lgb.Booster, x: np.ndarray, n_jobs: int, rows: int = 500_000) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.float32)
    iteration = bst.best_iteration or bst.current_iteration()
    for a in range(0, x.shape[0], rows):
        b = min(a + rows, x.shape[0])
        block = np.ascontiguousarray(x[a:b])
        out[a:b] = bst.predict(block, num_iteration=iteration, num_threads=n_jobs)
    return out


def _base_params(cfg: TrainConfig, n_jobs: int) -> dict:
    return dict(
        objective="binary",
        metric="None",
        learning_rate=0.05,
        bagging_fraction=0.8,
        bagging_freq=1,
        max_bin=MAX_BIN,
        force_col_wise=True,
        verbose=-1,
        seed=cfg.seed,
        bagging_seed=cfg.seed,
        feature_fraction_seed=cfg.seed,
        num_threads=n_jobs,
        scale_pos_weight=cfg.scale_pos_weight,
    )


def run_tune(cfg: TrainConfig, n_trials: int = 15, trial_rounds: int = 800, trial_early_stopping: int = 40) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = assemble_splits(cfg)
    n_jobs = bundle["n_jobs"]
    s1, pool, n_true = bundle["s1"], bundle["pool"], bundle["n_true"]
    va_ids, va_pairs = bundle["va_ids"], bundle["va_pairs"]
    x_tr, y_tr, x_va, y_va = bundle["x_tr"], bundle["y_tr"], bundle["x_va"], bundle["y_va"]
    rec, bcfg, t0 = bundle["rec"], bundle["bcfg"], bundle["t0"]
    del bundle

    feature_gb = (x_tr.nbytes + x_va.nbytes) / 1e9
    log.info(f"[tune] feature matrices {feature_gb:.2f} GB  (max_bin={MAX_BIN})")
    if feature_gb > 10.0:
        log.info("[tune] WARNING: matrices are large for 16 GB RAM; rerun with a smaller --n-train-s1")

    bin_dir = cfg.work_dir / "lgb_bin"
    train_bin, valid_bin = bin_dir / "train.bin", bin_dir / "valid.bin"
    dtr = _save_binary(x_tr, y_tr, train_bin)
    _save_binary(x_va, y_va, valid_bin, reference=dtr)
    del dtr, x_tr, y_tr
    gc.collect()

    veto = veto_mask(va_pairs, s1, pool)
    del s1, pool
    gc.collect()
    s1_idx = va_pairs["s1_idx"].to_numpy()
    pool_idx = va_pairs["pool_idx"].to_numpy()
    label = va_pairs["label"].to_numpy()

    va_npy = bin_dir / "x_va.npy"
    np.save(va_npy, np.ascontiguousarray(x_va))
    del x_va
    gc.collect()

    def _feval(preds: np.ndarray, dataset: lgb.Dataset):
        _tau, score = sweep_tau_arrays(
            np.asarray(preds, dtype=np.float64), s1_idx, pool_idx, label, n_true, va_ids,
            tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_step=cfg.tau_step,
            one_to_one=cfg.one_to_one, veto=veto,
        )
        return "macro_f05", score, True

    base = _base_params(cfg, n_jobs)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(base)
        params.update(
            lambda_l1=trial.suggest_float("lambda_l1", 0.5, 10.0),
            lambda_l2=trial.suggest_float("lambda_l2", 0.5, 10.0),
            feature_fraction=trial.suggest_float("feature_fraction", 0.65, 0.85),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 200),
            num_leaves=trial.suggest_int("num_leaves", 63, 127),
        )
        dtrain = _load_binary(train_bin)
        dvalid = _load_binary(valid_bin, reference=dtrain)
        bst = lgb.train(
            params, dtrain, num_boost_round=trial_rounds, valid_sets=[dvalid], valid_names=["val"],
            feval=_feval,
            callbacks=[lgb.early_stopping(trial_early_stopping, verbose=False), lgb.log_evaluation(0)],
        )
        score = float(bst.best_score["val"]["macro_f05"])
        log.info(f"[optuna] trial {trial.number} macro F0.5={score:.5f} "
                 f"trees={bst.best_iteration} params={trial.params}")
        del bst, dtrain, dvalid
        gc.collect()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials)
    log.info(f"[optuna] best macro F0.5={study.best_value:.5f} params={study.best_params}")

    params = dict(base)
    params.update(study.best_params)
    dtrain = _load_binary(train_bin)
    dvalid = _load_binary(valid_bin, reference=dtrain)
    bst = lgb.train(
        params, dtrain, num_boost_round=cfg.num_boost_round, valid_sets=[dvalid], valid_names=["val"],
        feval=_feval,
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False), lgb.log_evaluation(50)],
    )
    del dtrain, dvalid
    gc.collect()
    log.info(f"[tune] final best_iteration={bst.best_iteration}")

    x_va = np.load(va_npy, mmap_mode="r")
    prob = _predict(bst, x_va, n_jobs)
    del x_va
    prob[veto] = 0.0
    va_pairs = va_pairs.with_columns(pl.Series("prob", prob))
    tau, sweep = sweep_tau(
        va_pairs, n_true, va_ids, cfg.tau_min, cfg.tau_max, cfg.tau_step, cfg.one_to_one,
    )
    for t, f in sweep[::5] + ([sweep[-1]] if (len(sweep) - 1) % 5 else []):
        log.info(f"[tau] {t:.2f} -> macro F0.5 = {f:.5f}")
    if tau in (sweep[0][0], sweep[-1][0]):
        log.info(f"[tau] WARNING: optimum {tau:.2f} lies on the edge of [{cfg.tau_min:.2f}, {cfg.tau_max:.2f}]")
    f_best = dict(sweep)[tau]
    f_no_1to1 = macro_f05_selection(decide(va_pairs, tau, False), n_true, va_ids)
    oracle = macro_f05_selection(
        va_pairs.filter(pl.col("label") == 1).with_columns(pl.lit(1.0).alias("prob")),
        n_true, va_ids,
    )
    log.info(f"[tune] val macro F0.5 = {f_best:.5f} @ tau={tau:.2f} "
             f"(without one-to-one: {f_no_1to1:.5f}; blocking ceiling: {oracle:.5f})")

    artifact = dict(
        booster=bst, feature_names=FEATURE_NAMES, tau=float(tau), one_to_one=cfg.one_to_one,
        best_iteration=bst.best_iteration, params=params, blocking_cfg=bcfg.to_dict(),
        metrics=dict(
            val_macro_f05=f_best, val_no_one_to_one=f_no_1to1, blocking_ceiling=oracle,
            blocking_recall=rec["recall"], avg_candidates=rec["avg_candidates"], tau_sweep=sweep,
            optuna_best=float(study.best_value), optuna_params=study.best_params,
        ),
    )
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.model_path, "wb") as fh:
        pickle.dump(artifact, fh)
    bst.save_model(str(cfg.model_path.with_suffix(".txt")), num_iteration=bst.best_iteration)
    log.info(f"[tune] saved {cfg.model_path}")
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--work-dir", default=str(ROOT / "work"))
    ap.add_argument("--model-path", default=str(ROOT / "models" / "lgbm_er.pkl"))
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.0)
    ap.add_argument("--dev-full-pool", action="store_true")
    ap.add_argument("--n-train-s1", type=int, default=300_000)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0)
    ap.add_argument("--tau-min", type=float, default=0.70)
    ap.add_argument("--tau-max", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--trial-rounds", type=int, default=800)
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--no-one-to-one", action="store_true")
    a = ap.parse_args()
    cfg = TrainConfig(
        data_dir=resolve_data_dir(a.data_dir),
        work_dir=Path(a.work_dir),
        model_path=Path(a.model_path),
        n_jobs=a.n_jobs,
        dev_frac=a.dev_frac,
        dev_full_pool=a.dev_full_pool,
        n_train_s1=a.n_train_s1,
        seed=a.seed,
        scale_pos_weight=a.scale_pos_weight,
        tau_min=a.tau_min,
        tau_max=a.tau_max,
        one_to_one=not a.no_one_to_one,
        num_boost_round=a.num_boost_round,
        early_stopping=a.early_stopping,
        blocking=BlockingConfig(top_k_per_source=a.top_k, chunk_size=a.chunk_size, n_jobs=a.n_jobs),
    )
    run_tune(cfg, n_trials=a.trials, trial_rounds=a.trial_rounds)


if __name__ == "__main__":
    main()
