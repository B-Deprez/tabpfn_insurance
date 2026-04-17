"""Q3 — Interpretability / SHAP.

Computes SHAP values (or GLM coefficients) on fold 0 for all four
dataset × task combinations:
    freMTPL2 × freq,  freMTPL2 × sev,
    beMTPL97 × freq,  beMTPL97 × sev

Raw arrays are saved to res/shap/ and per-fold deviances are appended to
res/results.csv.

SHAP strategy:
    TabPFN  — built-in SHAP (KernelExplainer fallback)
    XGBoost — shap.TreeExplainer
    GLM     — standardised coefficients (coef * std(X_train)) on test fold

Usage:
    python scripts/run_q3_shap.py
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
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
from src.utils.metrics import compute_deviance
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q3_shap.yaml"
SHAP_DIR = PROJECT_ROOT / "res" / "shap"
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SHAP / coefficient computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shap_xgboost(model, X_test: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for an XGBoost model via TreeExplainer."""
    import shap
    explainer = shap.TreeExplainer(model._model)
    vals = explainer.shap_values(X_test)
    # TreeExplainer may return a list for multi-output; take first element
    return vals[0] if isinstance(vals, list) else vals


def _shap_tabpfn(model, X_test: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a TabPFN model."""
    return model.get_shap_values(X_test)


def _glm_standardised_coefs(model, X_train: pd.DataFrame) -> pd.Series:
    """Return standardised GLM coefficients (coef × std(X_train)).

    Multiplying each raw coefficient by the standard deviation of its feature
    puts all coefficients on a comparable scale.  The intercept is excluded.
    """
    coefs = model.get_coefficients()
    # Exclude intercept
    feat_coefs = coefs.drop("const", errors="ignore")
    stds = X_train.std(ddof=1)
    # Align: some one-hot features may differ from stds index
    common = feat_coefs.index.intersection(stds.index)
    standardised = feat_coefs.loc[common] * stds.loc[common]
    return standardised


def _save_array(arr: np.ndarray, dataset: str, task: str, model_name: str) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    fname = SHAP_DIR / f"{dataset}_{task}_fold0_{model_name}.npy"
    np.save(fname, arr)
    logger.info("Saved SHAP array to %s  shape=%s", fname, arr.shape)


def _save_series(s: pd.Series, dataset: str, task: str, model_name: str) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    fname = SHAP_DIR / f"{dataset}_{task}_fold0_{model_name}.csv"
    s.to_csv(fname, header=True)
    logger.info("Saved GLM coefficients to %s", fname)


# ──────────────────────────────────────────────────────────────────────────────
# Per (dataset, task) runner
# ──────────────────────────────────────────────────────────────────────────────

def run_combo(dataset: str, task: str, cfg: dict, feat_cfg: dict) -> None:
    """Run SHAP / coefficient computation for one dataset × task pair."""
    experiment_id = cfg["experiment_id"]
    shap_fold: int = cfg["shap_fold"]
    tabpfn_max: int = cfg["tabpfn"]["max_train_size"]
    cv_seed: int = cfg["cv_seed"]
    fold_seed = cv_seed + shap_fold

    logger.info("-" * 60)
    logger.info("Dataset: %s  |  Task: %s  |  Fold: %d", dataset, task, shap_fold)

    # Load, validate, split
    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)
    splits = make_cv_splits(df, dataset, cfg)
    train_df, test_df = get_fold(df, splits, shap_fold)

    result_rows: list[dict] = []

    # ── GLM ───────────────────────────────────────────────────────────────────
    logger.info("  Fitting GLM")
    X_tr_glm, X_te_glm = encode_features(train_df, test_df, dataset, "glm", feat_cfg)
    y_tr, w_tr, log_exp_tr = get_targets(train_df, dataset, task)
    y_te, w_te, log_exp_te = get_targets(test_df, dataset, task)

    glm = make_glm(task)
    t0 = time.perf_counter()
    glm.fit(X_tr_glm, y_tr, sample_weight=w_tr, log_exposure=log_exp_tr)
    mu_glm = glm.predict(X_te_glm, log_exposure=log_exp_te)
    elapsed_glm = time.perf_counter() - t0

    dev_glm = compute_deviance(task, y_te, mu_glm, w_te)
    logger.info("  GLM deviance=%.6f  time=%.1fs", dev_glm, elapsed_glm)

    std_coefs = _glm_standardised_coefs(glm, X_tr_glm)
    _save_series(std_coefs, dataset, task, "glm")

    result_rows += [
        build_result_row(experiment_id, dataset, "glm", task, shap_fold,
                         _metric_name(task), dev_glm),
        build_result_row(experiment_id, dataset, "glm", task, shap_fold,
                         "fit_predict_seconds", elapsed_glm),
    ]

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("  Fitting XGBoost")
    X_tr_tree, X_te_tree = encode_features(train_df, test_df, dataset, "tree", feat_cfg)

    xgb_model = make_xgboost(task)
    t0 = time.perf_counter()
    xgb_model.fit(X_tr_tree, y_tr, sample_weight=w_tr, log_exposure=log_exp_tr)
    mu_xgb = xgb_model.predict(X_te_tree, log_exposure=log_exp_te)
    elapsed_xgb = time.perf_counter() - t0

    dev_xgb = compute_deviance(task, y_te, mu_xgb, w_te)
    logger.info("  XGBoost deviance=%.6f  time=%.1fs", dev_xgb, elapsed_xgb)

    shap_xgb = _shap_xgboost(xgb_model, X_te_tree)
    _save_array(shap_xgb, dataset, task, "xgboost")

    # Also save feature names for the plotting notebook
    feat_names_path = SHAP_DIR / f"{dataset}_{task}_fold0_feature_names.npy"
    np.save(feat_names_path, np.array(X_te_tree.columns.tolist()))

    result_rows += [
        build_result_row(experiment_id, dataset, "xgboost", task, shap_fold,
                         _metric_name(task), dev_xgb),
        build_result_row(experiment_id, dataset, "xgboost", task, shap_fold,
                         "fit_predict_seconds", elapsed_xgb),
    ]

    # ── TabPFN ────────────────────────────────────────────────────────────────
    # TabPFN-2.6 receives raw unencoded features directly (no tree encoding)
    logger.info("  Fitting TabPFN")
    X_tr_tabpfn = get_raw_features(train_df, dataset, feat_cfg)
    X_te_tabpfn = get_raw_features(test_df, dataset, feat_cfg)

    tabpfn_model = make_tabpfn(task, max_train_size=tabpfn_max, shap=True)

    t0 = time.perf_counter()
    tabpfn_model.fit(
        X_tr_tabpfn, y_tr,
        sample_weight=w_tr,
        log_exposure=log_exp_tr,
        fold_seed=fold_seed,
    )
    mu_tabpfn = tabpfn_model.predict(X_te_tabpfn, log_exposure=log_exp_te)
    elapsed_tabpfn = time.perf_counter() - t0

    dev_tabpfn = compute_deviance(task, y_te, mu_tabpfn, w_te)
    logger.info("  TabPFN deviance=%.6f  time=%.1fs", dev_tabpfn, elapsed_tabpfn)

    shap_tabpfn = _shap_tabpfn(tabpfn_model, X_te_tabpfn)
    _save_array(shap_tabpfn, dataset, task, "tabpfn")

    # Save raw feature names and test values for beeswarm colour coding
    np.save(SHAP_DIR / f"{dataset}_{task}_fold0_tabpfn_feature_names.npy",
            np.array(X_te_tabpfn.columns.tolist()))
    # Numeric representation for colour scale (object columns → codes)
    X_te_numeric = X_te_tabpfn.apply(
        lambda c: c.astype("category").cat.codes if c.dtype == object else c
    )
    np.save(SHAP_DIR / f"{dataset}_{task}_fold0_X_test_tabpfn.npy",
            X_te_numeric.to_numpy(dtype=float))

    result_rows += [
        build_result_row(experiment_id, dataset, "tabpfn", task, shap_fold,
                         _metric_name(task), dev_tabpfn),
        build_result_row(experiment_id, dataset, "tabpfn", task, shap_fold,
                         "fit_predict_seconds", elapsed_tabpfn),
    ]

    append_results(result_rows)
    logger.info(
        "  Summary  GLM=%.6f  XGBoost=%.6f  TabPFN=%.6f",
        dev_glm, dev_xgb, dev_tabpfn,
    )


def _metric_name(task: str) -> str:
    return "poisson_deviance" if task == "freq" else "gamma_deviance"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q3 SHAP / interpretability experiment")

    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    for dataset in cfg["datasets"]:
        for task in cfg["tasks"]:
            run_combo(dataset, task, cfg, feat_cfg)

    logger.info("Q3 complete — SHAP arrays in res/shap/, deviances in res/results.csv")


if __name__ == "__main__":
    main()
