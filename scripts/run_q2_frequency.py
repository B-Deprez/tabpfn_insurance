"""Q2 — Frequency ceiling.

Evaluates TabPFN at subsample sizes {2k, 5k, 10k} and compares against
GLM and XGBoost baselines on both French and Belgian MTPL frequency data.

Subsampling is nested: the script draws max(subsample_sizes) training rows
per fold (seed = cv_seed + fold) and reuses the first N rows for each
smaller size (2k ⊂ 5k ⊂ 10k), keeping the comparison fair.

Usage:
    python scripts/run_q2_frequency.py
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
from src.methods.tabpfn_model import TabPFNFreq
from src.utils.metrics import (
    exposure_weighted_rmse_rate,
    poisson_deviance,
    pooled_poisson_deviance,
)
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q2_frequency.yaml"
ERROR_METRICS_PATH = PROJECT_ROOT / "res" / "results_error_metrics.csv"
logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    return "glm" if model_name == "glm" else "tree"


def run_baselines(
    dataset: str,
    cfg: dict,
    df,
    splits: list,
    feat_cfg: dict,
) -> None:
    """Run GLM and XGBoost on the full training fold (no subsampling)."""
    task = cfg["task"]
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]

    for model_name in ("glm", "xgboost"):
        family = _model_family(model_name)
        logger.info("--- Baseline: %s ---", model_name)

        fold_deviances: list[float] = []
        all_y: list[np.ndarray] = []
        all_mu: list[np.ndarray] = []
        all_e: list[np.ndarray] = []
        result_rows: list[dict] = []
        error_rows: list[dict] = []

        for fold in range(n_folds):
            train_df, test_df = get_fold(df, splits, fold)

            X_train, X_test = encode_features(train_df, test_df, dataset, family, feat_cfg)
            y_train, w_train, log_exp_train = get_targets(train_df, dataset, task)
            y_test, w_test, log_exp_test = get_targets(test_df, dataset, task)

            model = make_glm(task) if model_name == "glm" else make_xgboost(task)

            t0 = time.perf_counter()
            model.fit(X_train, y_train, sample_weight=w_train, log_exposure=log_exp_train)
            mu_test = model.predict(X_test, log_exposure=log_exp_test)
            elapsed = time.perf_counter() - t0

            dev = poisson_deviance(y_test, mu_test, w_test, sample_weight=w_test)
            y_rate_test = y_test / np.maximum(w_test, 1e-10)
            fold_rmse_rate = exposure_weighted_rmse_rate(y_rate_test, mu_test, w_test)
            fold_deviances.append(dev)
            all_y.append(y_test)
            all_mu.append(mu_test)
            all_e.append(w_test)

            logger.info(
                "  Fold %d: poisson_deviance=%.6f  exposure_weighted_rmse_rate=%.6f  time=%.1fs",
                fold, dev, fold_rmse_rate, elapsed,
            )

            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "poisson_deviance", dev,
            ))
            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "fit_predict_seconds", elapsed,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "exposure_weighted_rmse_rate", fold_rmse_rate,
            ))

        pooled = pooled_poisson_deviance(all_y, all_mu, all_e, all_e)
        result_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "poisson_deviance", pooled,
        ))

        logger.info(
            "  %s | %s | %s : mean=%.6f  std=%.6f  pooled=%.6f",
            dataset, model_name, task,
            float(np.mean(fold_deviances)),
            float(np.std(fold_deviances, ddof=1)),
            pooled,
        )
        append_results(result_rows)
        append_results(error_rows, output_path=ERROR_METRICS_PATH)


def run_tabpfn_subsample(
    dataset: str,
    cfg: dict,
    df,
    splits: list,
    feat_cfg: dict,
) -> None:
    """Run TabPFN at each subsample size, with nested subsampling."""
    task = cfg["task"]
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]
    subsample_sizes: list[int] = sorted(cfg["tabpfn"]["subsample_sizes"])
    max_size = subsample_sizes[-1]

    logger.info("--- TabPFN subsample sweep: sizes=%s ---", subsample_sizes)

    for fold in range(n_folds):
        train_df, test_df = get_fold(df, splits, fold)

        # TabPFN-2.6 uses raw unencoded features directly
        X_train_full = get_raw_features(train_df, dataset, feat_cfg)
        X_test = get_raw_features(test_df, dataset, feat_cfg)
        y_train_full, w_train_full, _ = get_targets(train_df, dataset, task)
        y_test, w_test, _ = get_targets(test_df, dataset, task)

        y_arr = y_train_full
        w_arr = w_train_full

        # Draw the master subsample (nested: smaller sizes take first N rows)
        fold_seed = cv_seed + fold
        rng = np.random.default_rng(fold_seed)
        n_available = len(X_train_full)
        master_size = min(max_size, n_available)
        master_idx = rng.choice(n_available, size=master_size, replace=False)

        for size in subsample_sizes:
            actual_size = min(size, master_size)
            idx = master_idx[:actual_size]

            X_sub = X_train_full.iloc[idx].reset_index(drop=True)
            y_sub = y_arr[idx]
            w_sub = w_arr[idx]

            # Strategy B: rate as response
            y_rate_sub = y_sub / np.maximum(w_sub, 1e-10)

            from tabpfn import TabPFNRegressor
            from src.methods.tabpfn_model import _detect_device
            device = _detect_device()

            t0 = time.perf_counter()
            model = TabPFNRegressor(device=device)
            model.fit(X_sub, y_rate_sub)
            mu_test = model.predict(X_test)
            elapsed = time.perf_counter() - t0

            dev = poisson_deviance(y_test, mu_test, w_test, sample_weight=w_test)
            y_rate_test = y_test / np.maximum(w_test, 1e-10)
            fold_rmse_rate = exposure_weighted_rmse_rate(y_rate_test, mu_test, w_test)
            logger.info(
                "  Fold %d | size=%d: poisson_deviance=%.6f  exposure_weighted_rmse_rate=%.6f  time=%.1fs",
                fold, actual_size, dev, fold_rmse_rate, elapsed,
            )

            # Store model name as "tabpfn_<size>" for plot differentiation
            model_label = f"tabpfn_{actual_size}"
            rows = [
                build_result_row(
                    experiment_id, dataset, model_label, task, fold,
                    "poisson_deviance", dev,
                ),
                build_result_row(
                    experiment_id, dataset, model_label, task, fold,
                    "fit_predict_seconds", elapsed,
                ),
            ]
            error_rows = [
                build_result_row(
                    experiment_id, dataset, model_label, task, fold,
                    "exposure_weighted_rmse_rate", fold_rmse_rate,
                ),
            ]
            append_results(rows)
            append_results(error_rows, output_path=ERROR_METRICS_PATH)


def run_dataset(dataset: str, cfg: dict) -> None:
    """Run baseline and TabPFN experiments on one dataset."""
    task = cfg["task"]
    logger.info("=" * 70)
    logger.info("Dataset: %s  |  Task: %s", dataset, task)
    logger.info("=" * 70)

    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)
    splits = make_cv_splits(df, dataset, cfg)
    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    run_baselines(dataset, cfg, df, splits, feat_cfg)
    run_tabpfn_subsample(dataset, cfg, df, splits, feat_cfg)


def main() -> None:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q2 frequency ceiling experiment")

    for dataset in cfg["datasets"]:
        run_dataset(dataset, cfg)

    logger.info("Q2 complete — results appended to res/results.csv")


if __name__ == "__main__":
    main()
