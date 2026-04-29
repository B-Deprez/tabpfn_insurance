"""Evaluation metrics for insurance pricing models.

Primary metrics:
    poisson_deviance  — frequency modelling
    gamma_deviance    — severity modelling
    auc_roc           — binary classification (fretelematic)

Both deviance functions are computed at the observation level and then averaged
(optionally weighted).  Use ``pooled_*`` variants for the aggregate OOF score.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

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
