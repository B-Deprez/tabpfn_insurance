"""Evaluation metrics for insurance pricing models.

Primary metrics:
    poisson_deviance  — frequency modelling
    gamma_deviance    — severity modelling
    auc_roc           — binary classification (fretelematic)

Secondary error-based metrics:
    rmse, mae                    — severity (unweighted by default)
    exposure_weighted_rmse_rate  — frequency (rate scale, exposure weights)

Calibration-style metrics (severity only):
    pearson_corr   — linear correlation between predictions and actuals
    spearman_corr  — rank correlation between predictions and actuals

Both deviance functions are computed at the observation level and then averaged
(optionally weighted).  Use ``pooled_*`` variants for the aggregate OOF score.
"""

from __future__ import annotations

import logging

import numpy as np

# sklearn is intentionally NOT imported at module level. Loading sklearn before
# torch (TabPFN) can cause an OpenMP runtime collision on Linux GPU nodes
# (libgomp from sklearn vs libiomp5 from torch), which manifests as a hard
# segfault on the VSC. AUC is the only sklearn metric still in use; it is
# lazy-imported inside ``auc_roc`` so the q1/q2 scripts never trigger it.

logger = logging.getLogger(__name__)

_EPS = 1e-10   # numerical floor to avoid log(0)


# ──────────────────────────────────────────────────────────────────────────────
# Poisson deviance
# ──────────────────────────────────────────────────────────────────────────────

def poisson_deviance(
    y_count: np.ndarray,
    mu_rate: np.ndarray,
    exposure: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute (weighted) mean Poisson deviance.

    Formula (per observation):
        d_i = 2 * [ y_i * log(y_i / (e_i * mu_i)) - (y_i - e_i * mu_i) ]
    where y_i = observed claim count, e_i = exposure, mu_i = predicted rate.
    The convention 0 * log(0) = 0 is applied for zero-count observations.

    Args:
        y_count: observed claim counts.
        mu_rate: predicted claim rate (per unit exposure).
        exposure: exposure (fraction of year at risk).
        sample_weight: optional observation weights for the mean.
            If None, uniform weights are used.

    Returns:
        Scalar weighted mean Poisson deviance.
    """
    y = np.asarray(y_count, dtype=float)
    mu = np.asarray(mu_rate, dtype=float)
    e = np.asarray(exposure, dtype=float)

    mu_count = e * mu  # expected counts
    mu_count = np.maximum(mu_count, _EPS)

    # 0 * log(0 / mu_count) = 0 by convention
    log_term = np.where(y > 0, y * np.log(y / mu_count), 0.0)
    d = 2.0 * (log_term - (y - mu_count))

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.average(d, weights=w))
    return float(np.mean(d))


def pooled_poisson_deviance(
    y_counts: list[np.ndarray],
    mu_rates: list[np.ndarray],
    exposures: list[np.ndarray],
    sample_weights: list[np.ndarray] | None = None,
) -> float:
    """Compute pooled OOF Poisson deviance across all folds.

    Concatenates all out-of-fold arrays and computes a single aggregate score.
    """
    y = np.concatenate(y_counts)
    mu = np.concatenate(mu_rates)
    e = np.concatenate(exposures)
    w = np.concatenate(sample_weights) if sample_weights is not None else None
    return poisson_deviance(y, mu, e, w)


# ──────────────────────────────────────────────────────────────────────────────
# Gamma deviance
# ──────────────────────────────────────────────────────────────────────────────

def gamma_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute (weighted) mean Gamma deviance.

    Formula (per observation):
        d_i = 2 * [ (y_i - mu_i) / mu_i - log(y_i / mu_i) ]

    Args:
        y: observed average severity.
        mu: predicted average severity.
        sample_weight: typically ClaimNb — used as observation weight.

    Returns:
        Scalar weighted mean Gamma deviance.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    mu = np.maximum(mu, _EPS)
    y = np.maximum(y, _EPS)

    d = 2.0 * ((y - mu) / mu - np.log(y / mu))

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.average(d, weights=w))
    return float(np.mean(d))


def pooled_gamma_deviance(
    ys: list[np.ndarray],
    mus: list[np.ndarray],
    sample_weights: list[np.ndarray] | None = None,
) -> float:
    """Compute pooled OOF Gamma deviance across all folds."""
    y = np.concatenate(ys)
    mu = np.concatenate(mus)
    w = np.concatenate(sample_weights) if sample_weights is not None else None
    return gamma_deviance(y, mu, w)


# ──────────────────────────────────────────────────────────────────────────────
# RMSE / MAE  (severity — unweighted by default)
# ──────────────────────────────────────────────────────────────────────────────

def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Root mean squared error (optionally weighted).

    Args:
        y_true: observed values.
        y_pred: predicted values.
        sample_weight: optional observation weights.  Defaults to uniform.

    Returns:
        Scalar RMSE.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    sq = (y - p) ** 2
    if sample_weight is None:
        return float(np.sqrt(np.mean(sq)))
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sqrt(np.average(sq, weights=w)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean absolute error (optionally weighted).

    Args:
        y_true: observed values.
        y_pred: predicted values.
        sample_weight: optional observation weights.  Defaults to uniform.

    Returns:
        Scalar MAE.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ae = np.abs(y - p)
    if sample_weight is None:
        return float(np.mean(ae))
    w = np.asarray(sample_weight, dtype=float)
    return float(np.average(ae, weights=w))


# ──────────────────────────────────────────────────────────────────────────────
# Correlation between predictions and actuals  (severity — calibration check)
# ──────────────────────────────────────────────────────────────────────────────

def pearson_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """(Weighted) Pearson correlation between predictions and actuals.

    A model that ranks risks well but is mis-scaled (a typical calibration
    failure) will still produce a high Pearson correlation, so this metric
    complements deviance/RMSE rather than replacing them.

    Args:
        y_true: observed values.
        y_pred: predicted values.
        sample_weight: optional observation weights (e.g. ClaimNb for sev).

    Returns:
        Scalar Pearson correlation in [-1, 1].  Returns ``nan`` if either
        input has zero variance under the supplied weights.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(sample_weight, dtype=float)

    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan")

    y_mean = np.sum(w * y) / w_sum
    p_mean = np.sum(w * p) / w_sum
    cov = np.sum(w * (y - y_mean) * (p - p_mean)) / w_sum
    var_y = np.sum(w * (y - y_mean) ** 2) / w_sum
    var_p = np.sum(w * (p - p_mean) ** 2) / w_sum
    denom = np.sqrt(var_y * var_p)
    if denom <= 0:
        return float("nan")
    return float(cov / denom)


def spearman_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Spearman rank correlation between predictions and actuals.

    Measures monotonic agreement — invariant to any monotone re-scaling of
    the predictions. Useful for diagnosing whether a regressor that is
    mis-calibrated on the original scale still ranks observations correctly.
    No weighting (weighted Spearman is non-standard); pass through any
    sample_weight argument explicitly via duplication if needed.

    Returns:
        Scalar Spearman correlation in [-1, 1].
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    from scipy.stats import spearmanr  # lazy: only used by severity script
    rho, _ = spearmanr(y, p)
    return float(rho)


# ──────────────────────────────────────────────────────────────────────────────
# Exposure-weighted RMSE (frequency — rate scale)
# ──────────────────────────────────────────────────────────────────────────────

def exposure_weighted_rmse_rate(
    y_rate_true: np.ndarray,
    y_rate_pred: np.ndarray,
    exposure: np.ndarray,
) -> float:
    """RMSE on the claim-rate scale, weighted by exposure.

    Both inputs must already be in claims-per-unit-exposure (i.e.
    ``y_count / exposure`` and the model's rate prediction).

    Args:
        y_rate_true: observed claim rate (counts / exposure).
        y_rate_pred: predicted claim rate.
        exposure: per-observation exposure used as the sample weight.

    Returns:
        Scalar exposure-weighted RMSE on the rate scale.
    """
    return rmse(y_rate_true, y_rate_pred, sample_weight=exposure)


# ──────────────────────────────────────────────────────────────────────────────
# AUC-ROC
# ──────────────────────────────────────────────────────────────────────────────

def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve.

    Args:
        y_true: binary ground-truth labels (0/1).
        y_score: predicted probabilities for the positive class.

    Returns:
        Scalar AUC-ROC value in [0, 1].
    """
    from sklearn.metrics import roc_auc_score  # lazy: only used by q4
    return float(roc_auc_score(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)))


def pooled_auc_roc(
    y_trues: list[np.ndarray],
    y_scores: list[np.ndarray],
) -> float:
    """Compute AUC-ROC on concatenated out-of-fold predictions."""
    return auc_roc(np.concatenate(y_trues), np.concatenate(y_scores))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def compute_deviance(
    task: str,
    y: np.ndarray,
    mu: np.ndarray,
    exposure_or_weight: np.ndarray,
) -> float:
    """Dispatch to the correct deviance function.

    Args:
        task: ``"freq"`` or ``"sev"``.
        y: for freq = claim counts; for sev = average severity.
        mu: predicted rate (freq) or predicted severity (sev).
        exposure_or_weight: for freq = exposure (also used as weight);
            for sev = claim count weights.

    Returns:
        Scalar deviance value.
    """
    if task == "freq":
        return poisson_deviance(y, mu, exposure_or_weight, sample_weight=exposure_or_weight)
    if task == "sev":
        return gamma_deviance(y, mu, sample_weight=exposure_or_weight)
    raise ValueError(f"Unknown task '{task}'")
