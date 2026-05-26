"""Q2 — Frequency ceiling.

Evaluates TabPFN at subsample sizes {2k, 5k, 10k} and compares against
GLM and XGBoost baselines on both French and Belgian MTPL frequency data.

Subsampling is nested: the script draws max(subsample_sizes) training rows
per fold (seed = cv_seed + fold) and reuses the first N rows for each
smaller size (2k ⊂ 5k ⊂ 10k), keeping the comparison fair.

Per-fold and pooled Poisson deviance are written to
``res/results_frequency.csv``; the exposure-weighted RMSE-on-rate metric is
written to ``res/results_error_frequency.csv``.

Usage:
    python scripts/run_q2_frequency.py
"""

from __future__ import annotations

import gc
import sys
import time
import logging
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:  # CPU-only environments without torch
    torch = None


def _release_gpu() -> None:
    """Drop Python garbage and return CUDA buffers to the caching allocator.

    Called between TabPFN fits to prevent fragmentation-driven OOM on long
    multi-fold / multi-size sweeps. Safe no-op when torch or CUDA is absent.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

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
from src.methods.tabpfn_model import (
    TabPFNFreq,
    _VERSION_MAX_TRAIN_SIZE,
    _make_regressor,
)
from src.utils.metrics import (
    exposure_weighted_rmse_rate,
    poisson_deviance,
    pooled_poisson_deviance,
)
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q2_frequency.yaml"
RESULTS_PATH = PROJECT_ROOT / "res" / "results_frequency.csv"
ERROR_METRICS_PATH = PROJECT_ROOT / "res" / "results_error_frequency.csv"
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

            del model, train_df, test_df, X_train, X_test, mu_test
            del y_train, w_train, log_exp_train, log_exp_test, y_rate_test

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
        append_results(result_rows, output_path=RESULTS_PATH)
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
    # Parse subsample_sizes: integers feed the nested-subsample sweep, while
    # the "full" sentinel triggers an additional pass on the entire training
    # fold (no subsampling).  Putting "full" last keeps memory growth monotonic.
    subsample_sizes_raw = cfg["tabpfn"]["subsample_sizes"]
    numeric_sizes: list[int] = sorted(
        s for s in subsample_sizes_raw if isinstance(s, int)
    )
    include_full: bool = "full" in subsample_sizes_raw
    sizes_to_run: list[int | str] = [*numeric_sizes]
    if include_full:
        sizes_to_run.append("full")

    # Accept either a list or a single string for backward compat.
    tabpfn_versions: list[str] = cfg["tabpfn"].get(
        "tabpfn_versions",
        [cfg["tabpfn"].get("tabpfn_version", "v3")],
    )
    max_size = numeric_sizes[-1] if numeric_sizes else 0

    logger.info(
        "--- TabPFN subsample sweep: sizes=%s, versions=%s ---",
        sizes_to_run, tabpfn_versions,
    )

    # Hoist device detection out of the per-size loop. The regressor is built
    # via the shared factory so the v2_6/v3 dispatch matches the other scripts.
    from src.methods.tabpfn_model import _detect_device
    device = _detect_device()

    for fold in range(n_folds):
        train_df, test_df = get_fold(df, splits, fold)

        # TabPFN-2.6 uses raw unencoded features directly
        X_train_full = get_raw_features(train_df, dataset, feat_cfg)
        X_test = get_raw_features(test_df, dataset, feat_cfg)
        y_train_full, w_train_full, _ = get_targets(train_df, dataset, task)
        y_test, w_test, _ = get_targets(test_df, dataset, task)
        del train_df, test_df

        # Strategy B: rate response. Compute once per fold; slice for each size.
        y_rate_full = y_train_full / np.maximum(w_train_full, 1e-10)

        # Draw the master subsample (nested: smaller sizes take first N rows).
        # When "full" is requested we permute the entire fold so that the
        # smaller subsamples remain strict subsets of the full pass.
        fold_seed = cv_seed + fold
        rng = np.random.default_rng(fold_seed)
        n_available = len(X_train_full)
        master_size = n_available if include_full else min(max_size, n_available)
        master_idx = rng.choice(n_available, size=master_size, replace=False)

        y_rate_test = y_test / np.maximum(w_test, 1e-10)

        for size in sizes_to_run:
            if size == "full":
                actual_size = n_available
                model_label = "tabpfn_full"
                X_sub = X_train_full
                y_rate_sub = y_rate_full
            else:
                actual_size = min(size, master_size)
                model_label = f"tabpfn_{actual_size}"
                idx = master_idx[:actual_size]
                X_sub = X_train_full.iloc[idx].reset_index(drop=True)
                y_rate_sub = y_rate_full[idx]

            # v2.5 and v2.6 have fixed pretraining ceilings (50k and 100k rows
            # respectively); the "full" data point exists to exercise v3's
            # expanded context size, so skip the older versions whenever the
            # subsample exceeds their supported limit.
            versions_for_size = [
                v for v in tabpfn_versions
                if not (
                    v in _VERSION_MAX_TRAIN_SIZE
                    and actual_size > _VERSION_MAX_TRAIN_SIZE[v]
                )
            ]
            skipped = [v for v in tabpfn_versions if v not in versions_for_size]
            if skipped:
                logger.info(
                    "  Fold %d | size=%s (n=%d): skipping %s (>100k pretraining limit)",
                    fold, model_label, actual_size, skipped,
                )

            # Both versions fit the same subsample, so per-version timing
            # and deviance can be compared directly.
            for current_version in versions_for_size:
                t0 = time.perf_counter()
                model = _make_regressor(current_version, device)
                model.fit(X_sub, y_rate_sub)
                mu_test = model.predict(X_test)
                elapsed = time.perf_counter() - t0

                dev = poisson_deviance(y_test, mu_test, w_test, sample_weight=w_test)
                fold_rmse_rate = exposure_weighted_rmse_rate(y_rate_test, mu_test, w_test)
                size_label = "full" if size == "full" else str(actual_size)
                logger.info(
                    "  Fold %d | size=%s (n=%d) | %s: poisson_deviance=%.6f  "
                    "exposure_weighted_rmse_rate=%.6f  time=%.1fs",
                    fold, size_label, actual_size, current_version,
                    dev, fold_rmse_rate, elapsed,
                )

                # Model label encodes the size (or "full"); the per-version
                # axis lives in the ``tabpfn_version`` column.
                rows = [
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "poisson_deviance", dev, tabpfn_version=current_version,
                    ),
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "fit_predict_seconds", elapsed,
                        tabpfn_version=current_version,
                    ),
                ]
                error_rows = [
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "exposure_weighted_rmse_rate", fold_rmse_rate,
                        tabpfn_version=current_version,
                    ),
                ]
                append_results(rows, output_path=RESULTS_PATH)
                append_results(error_rows, output_path=ERROR_METRICS_PATH)

                # Critical: release the fitted TabPFN and its GPU buffers
                # before the next (size, version) iteration. Without this,
                # the CUDA caching allocator fragments across the
                # sizes × versions × n_folds fits and eventually segfaults.
                del model, mu_test
                _release_gpu()

            del X_sub, y_rate_sub

        # End-of-fold cleanup before loading the next fold's data.
        del X_train_full, X_test, y_train_full, w_train_full, y_test, w_test
        del y_rate_full, y_rate_test, master_idx
        _release_gpu()


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

    del df, splits, feat_cfg
    _release_gpu()


def main() -> None:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q2 frequency ceiling experiment")

    for dataset in cfg["datasets"]:
        run_dataset(dataset, cfg)
        _release_gpu()

    logger.info(
        "Q2 complete — deviances appended to %s, error metrics to %s",
        RESULTS_PATH, ERROR_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
