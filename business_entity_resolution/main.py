"""End-to-end runner.

    python main.py --mode train          # blocking + LightGBM + tau  -> models/lgbm_er.pkl
    python main.py --mode tune           # regularized search + tau sweep -> models/lgbm_er.pkl
    python main.py --mode test           # blocking + scoring          -> output/*.tsv
    python main.py --mode all            # train, then test
    python main.py --mode validate       # only run the official submission validator

Add --dev-frac 0.01 for a fast smoke test on a 1% sample.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.blocking import BlockingConfig  # noqa: E402
from src.inference import InferConfig, run_inference  # noqa: E402
from src.train import TrainConfig, run_train  # noqa: E402
from src.tune_lgbm import run_tune  # noqa: E402
from src.utils import get_logger, resolve_data_dir  # noqa: E402

log = get_logger()


def find_validator() -> Path | None:
    direct = [
        ROOT.parent / "Dataset" / "student_resource" / "utils" / "validate_submission.py",
        ROOT.parent / "utils" / "validate_submission.py",
        Path.cwd() / "utils" / "validate_submission.py",
        ROOT / "utils" / "validate_submission.py",
        ROOT.parent.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py",
    ]
    for p in direct:
        if p.exists():
            return p
    for base in (ROOT.parent.parent, ROOT.parent, Path.cwd()):
        if not base.is_dir():
            continue
        for pat in ("*/student_resource/utils/validate_submission.py",
                    "*/*/student_resource/utils/validate_submission.py"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def run_validator(output_dir: Path, data_dir: Path) -> int:
    v = find_validator()
    if v is None:
        log.info("[validate] utils/validate_submission.py not found - skipping")
        return 0
    cmd = [sys.executable, str(v), "--matching", str(output_dir / "matching_results.tsv"),
           "--candidate", str(output_dir / "candidate_pairs.tsv"), "--test-dir", str(data_dir / "test")]
    log.info("[validate] " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "test", "all", "validate", "tune"], default="all")
    ap.add_argument("--data-dir", default=None, help="folder containing train/ and test/ (default: auto-detect)")
    ap.add_argument("--output-dir", default=str(ROOT.parent / "output"))
    ap.add_argument("--work-dir", default=str(ROOT / "work"), help="cache for normalised data / candidate pairs")
    ap.add_argument("--model-path", default=str(ROOT / "models" / "lgbm_er.pkl"))
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--dev-frac", type=float, default=0.0, help="smoke-test on this fraction of the data")
    ap.add_argument("--dev-full-pool", action="store_true", help="with --dev-frac: sample S1 only, keep the full S2/S3 pool")
    # training
    ap.add_argument("--n-train-s1", type=int, default=300_000, help="S1 entities for the classifier (0 = all)")
    ap.add_argument("--scale-pos-weight", type=float, default=1.0)
    ap.add_argument("--tau-min", type=float, default=0.70)
    ap.add_argument("--tau-max", type=float, default=0.98)
    ap.add_argument("--trial-rounds", type=int, default=800, help="boosting rounds per Optuna trial in --mode tune")
    ap.add_argument("--no-one-to-one", action="store_true", help="disable S2/S3 conflict resolution")
    ap.add_argument("--optuna-trials", type=int, default=0)
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    # blocking
    ap.add_argument("--top-k", type=int, default=15, help="candidates per source per S1 (total <= 2*k)")
    ap.add_argument("--chunk-size", type=int, default=1000, help="S1 rows per blocking task (lower = less RAM)")
    # inference
    ap.add_argument("--tau", type=float, default=None, help="override tau stored in the model")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()

    data_dir = resolve_data_dir(a.data_dir)
    output_dir, work_dir, model_path = Path(a.output_dir), Path(a.work_dir), Path(a.model_path)
    log.info(f"data_dir={data_dir}  output_dir={output_dir}  model={model_path}")

    if a.mode == "tune":
        bcfg = BlockingConfig(top_k_per_source=a.top_k, chunk_size=a.chunk_size, n_jobs=a.n_jobs)
        run_tune(TrainConfig(
            data_dir=data_dir, work_dir=work_dir, model_path=model_path, n_jobs=a.n_jobs, dev_frac=a.dev_frac, dev_full_pool=a.dev_full_pool,
            n_train_s1=a.n_train_s1, seed=a.seed, scale_pos_weight=a.scale_pos_weight, tau_min=a.tau_min,
            tau_max=a.tau_max, one_to_one=not a.no_one_to_one, num_boost_round=a.num_boost_round,
            early_stopping=100, blocking=bcfg,
        ), n_trials=a.optuna_trials or 15, trial_rounds=a.trial_rounds)
    if a.mode in ("train", "all"):
        bcfg = BlockingConfig(top_k_per_source=a.top_k, chunk_size=a.chunk_size, n_jobs=a.n_jobs)
        run_train(TrainConfig(
            data_dir=data_dir, work_dir=work_dir, model_path=model_path, n_jobs=a.n_jobs, dev_frac=a.dev_frac, dev_full_pool=a.dev_full_pool,
            n_train_s1=a.n_train_s1, seed=a.seed, scale_pos_weight=a.scale_pos_weight, tau_min=a.tau_min,
            tau_max=a.tau_max, one_to_one=not a.no_one_to_one, num_boost_round=a.num_boost_round,
            optuna_trials=a.optuna_trials, blocking=bcfg,
        ))
    if a.mode in ("test", "all"):
        run_inference(InferConfig(
            data_dir=data_dir, work_dir=work_dir, model_path=model_path, output_dir=output_dir,
            n_jobs=a.n_jobs, dev_frac=a.dev_frac, dev_full_pool=a.dev_full_pool, tau=a.tau,
            one_to_one=False if a.no_one_to_one else None,
        ))
    if a.mode in ("test", "all", "validate") and not a.no_validate and a.dev_frac == 0:
        rc = run_validator(output_dir, data_dir)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
