"""Q1 — Severity benchmark.

Runs 5-fold CV for GLM, XGBoost, and TabPFN on French and Belgian MTPL
severity data.  Records per-fold and pooled Gamma deviance in res/results.csv
and prints the Table 1 summary to the log.

Usage:
    python scripts/run_q1_severity.py
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_setup import setup_logging
from src.data.loaders import load_dataset
from src.data.contracts import validate_dataset
from src.data.cv import make_cv_splits, get_fold
from src.data.preprocessing import encode_features, get_raw_features, get_targets
from src.methods.glm_model import make_glm
from src.methods.xgboost_model import make_xgboost
from src.methods.tabpfn_model import make_tabpfn
from src.utils.metrics import gamma_deviance, mae, pooled_gamma_deviance, rmse
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q1_severity.yaml"
ERROR_METRICS_PATH = PROJECT_ROOT / "res" / "results_error_metrics.csv"
logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    """Return encoding family for GLM/XGBoost; TabPFN uses raw features."""
    if model_name == "glm":
        return "glm"
    if model_name == "xgboost":
        return "tree"
    return "tabpfn"  # raw unencoded features


def run_dataset(dataset: str, cfg: dict) -> None:
    """Run all models on one dataset for the severity task."""
    task = cfg["task"]  # "sev"
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]
    tabpfn_max = cfg["tabpfn"]["max_train_size"]

    logger.info("=" * 70)
    logger.info("Dataset: %s  |  Task: %s", dataset, task)
    logger.info("=" * 70)

    # ── Load & validate ───────────────────────────────────────────────────────
    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)

    # ── CV splits ─────────────────────────────────────────────────────────────
    splits = make_cv_splits(df, dataset, cfg)

    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    for model_name in cfg["models"]:
        family = _model_family(model_name)
        logger.info("--- Model: %s ---", model_name)

        fold_deviances: list[float] = []
        all_y: list[np.ndarray] = []
        all_mu: list[np.ndarray] = []
        all_w: list[np.ndarray] = []
        result_rows: list[dict] = []
        error_rows: list[dict] = []

        for fold in range(n_folds):
            train_df, test_df = get_fold(df, splits, fold)

            if family == "tabpfn":
                X_train = get_raw_features(train_df, dataset, feat_cfg)
                X_test = get_raw_features(test_df, dataset, feat_cfg)
            else:
                X_train, X_test = encode_features(train_df, test_df, dataset, family, feat_cfg)
            y_train, w_train, log_exp_train = get_targets(train_df, dataset, task)
            y_test, w_test, log_exp_test = get_targets(test_df, dataset, task)

            # Instantiate model
            if model_name == "glm":
                model = make_glm(task)
            elif model_name == "xgboost":
                model = make_xgboost(task)
            else:
                model = make_tabpfn(task, max_train_size=tabpfn_max)

            # Fit + predict with wall-clock timing
            fold_seed = cv_seed + fold
            t0 = time.perf_counter()
            fit_kwargs: dict = dict(
                sample_weight=w_train,
                log_exposure=log_exp_train,
            )
            if model_name == "tabpfn":
                fit_kwargs["fold_seed"] = fold_seed
            model.fit(X_train, y_train, **fit_kwargs)
            mu_test = model.predict(X_test, log_exposure=log_exp_test)
            elapsed = time.perf_counter() - t0

            # Evaluate
            dev = gamma_deviance(y_test, mu_test, sample_weight=w_test)
            fold_rmse = rmse(y_test, mu_test)
            fold_mae = mae(y_test, mu_test)
            fold_deviances.append(dev)
            all_y.append(y_test)
            all_mu.append(mu_test)
            all_w.append(w_test)

            logger.info(
                "  Fold %d: gamma_deviance=%.6f  rmse=%.6f  mae=%.6f  time=%.1fs",
                fold, dev, fold_rmse, fold_mae, elapsed,
            )

            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "gamma_deviance", dev,
            ))
            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "fit_predict_seconds", elapsed,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "rmse", fold_rmse,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "mae", fold_mae,
            ))

        # Pooled OOF score
        pooled = pooled_gamma_deviance(all_y, all_mu, all_w)
        result_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "gamma_deviance", pooled,
        ))

        mean_dev = float(np.mean(fold_deviances))
        std_dev = float(np.std(fold_deviances, ddof=1))
        logger.info(
            "  %s | %s | %s : mean=%.6f  std=%.6f  pooled=%.6f",
            dataset, model_name, task, mean_dev, std_dev, pooled,
        )

        append_results(result_rows)
        append_results(error_rows, output_path=ERROR_METRICS_PATH)


def main() -> None:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q1 severity benchmark")
    logger.info("Config: %s", CFG_PATH)

    for dataset in cfg["datasets"]:
        run_dataset(dataset, cfg)

    logger.info("Q1 complete — results appended to res/results.csv")


if __name__ == "__main__":
    main()
