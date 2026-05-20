"""Q4 — fretelematic binary classification benchmark.

Runs 5-fold CV for LogisticGLM, BinaryXGBoost, and TabPFNClf on the
fretelematic telematics dataset (1,177 policies).  Records per-fold and
pooled AUC-ROC in ``res/results_fretelematic.csv``.

Usage:
    python scripts/run_q4_fretelematic.py
"""

from __future__ import annotations

import gc
import sys
import time
import logging
from pathlib import Path

import numpy as np
import yaml

import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

try:
    import torch
except ImportError:  # CPU-only environments without torch
    torch = None


def _release_gpu() -> None:
    """Drop Python garbage and return CUDA buffers to the caching allocator."""
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
from src.methods.tabpfn_model import make_tabpfn
from src.utils.metrics import auc_roc, pooled_auc_roc
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q4_fretelematic.yaml"
RESULTS_PATH = PROJECT_ROOT / "res" / "results_fretelematic.csv"
logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    """Return encoding family for GLM/XGBoost; TabPFN uses raw features."""
    if model_name == "glm":
        return "glm"
    if model_name == "xgboost":
        return "tree"
    return "tabpfn"


def _expand_models(models: list[str], tabpfn_versions: list[str]) -> list[tuple[str, str]]:
    """Flatten ``models`` into (model_name, tabpfn_version) pairs.

    For ``"tabpfn"`` one pair is emitted per entry in ``tabpfn_versions``.
    Non-TabPFN models get a single pair with an empty version string.
    """
    pairs: list[tuple[str, str]] = []
    for m in models:
        if m == "tabpfn":
            for v in tabpfn_versions:
                pairs.append((m, v))
        else:
            pairs.append((m, ""))
    return pairs


def run_dataset(dataset: str, cfg: dict) -> None:
    """Run all models on the fretelematic dataset for binary classification."""
    task = cfg["task"]  # "clf"
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]
    tabpfn_max = cfg["tabpfn"]["max_train_size"]
    # Accept either a list or a single string for backward compat.
    tabpfn_versions: list[str] = cfg["tabpfn"].get(
        "tabpfn_versions",
        [cfg["tabpfn"].get("tabpfn_version", "v3")],
    )

    logger.info("=" * 70)
    logger.info("Dataset: %s  |  Task: %s", dataset, task)
    logger.info("=" * 70)

    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)

    splits = make_cv_splits(df, dataset, cfg)

    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    for model_name, row_version in _expand_models(cfg["models"], tabpfn_versions):
        family = _model_family(model_name)
        descr = f"{model_name} ({row_version})" if row_version else model_name
        logger.info("--- Model: %s ---", descr)

        fold_aucs: list[float] = []
        all_y: list[np.ndarray] = []
        all_p: list[np.ndarray] = []
        result_rows: list[dict] = []

        for fold in range(n_folds):
            train_df, test_df = get_fold(df, splits, fold)

            if family == "tabpfn":
                X_train = get_raw_features(train_df, dataset, feat_cfg)
                X_test = get_raw_features(test_df, dataset, feat_cfg)
            else:
                X_train, X_test = encode_features(train_df, test_df, dataset, family, feat_cfg)
            y_train, w_train, log_exp_train = get_targets(train_df, dataset, task)
            y_test, _, log_exp_test = get_targets(test_df, dataset, task)

            if model_name == "glm":
                model = make_glm(task)
            elif model_name == "xgboost":
                model = make_xgboost(task)
            else:
                model = make_tabpfn(
                    task,
                    max_train_size=tabpfn_max,
                    tabpfn_version=row_version,
                )

            fold_seed = cv_seed + fold
            t0 = time.perf_counter()
            fit_kwargs: dict = dict(
                sample_weight=w_train,
                log_exposure=log_exp_train,
            )
            if model_name == "tabpfn":
                fit_kwargs["fold_seed"] = fold_seed
            model.fit(X_train, y_train, **fit_kwargs)
            p_test = model.predict(X_test, log_exposure=log_exp_test)
            elapsed = time.perf_counter() - t0

            fold_auc = auc_roc(y_test, p_test)
            fold_aucs.append(fold_auc)
            all_y.append(y_test)
            all_p.append(p_test)

            logger.info(
                "  Fold %d: auc_roc=%.4f  time=%.1fs",
                fold, fold_auc, elapsed,
            )

            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "auc_roc", fold_auc, tabpfn_version=row_version,
            ))
            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "fit_predict_seconds", elapsed, tabpfn_version=row_version,
            ))

            del model, train_df, test_df, X_train, X_test, p_test
            del y_train, w_train, log_exp_train, log_exp_test
            _release_gpu()

        pooled = pooled_auc_roc(all_y, all_p)
        result_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "auc_roc", pooled, tabpfn_version=row_version,
        ))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs, ddof=1))
        logger.info(
            "  %s | %s | %s : mean=%.4f  std=%.4f  pooled=%.4f",
            dataset, model_name, task, mean_auc, std_auc, pooled,
        )

        append_results(result_rows, output_path=RESULTS_PATH)

    del df, splits, feat_cfg
    _release_gpu()


def main() -> None:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q4 fretelematic binary classification benchmark")
    logger.info("Config: %s", CFG_PATH)

    run_dataset(cfg["dataset"], cfg)

    logger.info("Q4 complete — AUC-ROC results appended to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
