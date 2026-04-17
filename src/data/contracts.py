"""Dataset contracts — structural and statistical assertions on loaded data.

Call the appropriate validate_* function immediately after loading a dataset.
All functions raise ``ValueError`` on violation; they never silently pass.

Post-split fold-level assertions are provided separately via
``validate_fold_claim_rates``, which must be called after CV splitting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tolerance on expected row counts (±1 %)
ROW_COUNT_TOL = 0.01
# Maximum allowed deviation of per-fold claim rate from overall rate (0.5 pp)
FOLD_RATE_TOL = 0.005


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_row_count(df: pd.DataFrame, expected: int, label: str) -> None:
    lo = int(expected * (1 - ROW_COUNT_TOL))
    hi = int(expected * (1 + ROW_COUNT_TOL))
    if not (lo <= len(df) <= hi):
        raise ValueError(
            f"[{label}] Row count {len(df):,} outside expected range "
            f"[{lo:,}, {hi:,}] (expected ≈ {expected:,})"
        )
    logger.info("[%s] Row count OK: %d (expected ≈ %d)", label, len(df), expected)


def _check_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}")
    logger.info("[%s] All required columns present", label)


def _check_no_nulls(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for col in cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(
                f"[{label}] Column '{col}' has {n_null} unexpected null values"
            )
    logger.info("[%s] No nulls in critical columns: %s", label, cols)


def _check_non_negative(df: pd.DataFrame, col: str, label: str) -> None:
    n_neg = (df[col] < 0).sum()
    if n_neg > 0:
        raise ValueError(
            f"[{label}] Column '{col}' has {n_neg} negative values"
        )


# ──────────────────────────────────────────────────────────────────────────────
# freMTPL2 contracts
# ──────────────────────────────────────────────────────────────────────────────

# Expected row count after cleaning (approx — contracts allow ±1%)
_FREMTPL2_FREQ_EXPECTED_ROWS = 677_991
_FREMTPL2_SEV_EXPECTED_ROWS = 24_885    # post-join/clean; observed on first run

_FREMTPL2_FREQ_REQUIRED_COLS = [
    "IDpol", "ClaimNb", "Exposure",
    "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "VehBrand", "VehGas", "Region", "Density",
]
_FREMTPL2_SEV_REQUIRED_COLS = [
    "IDpol", "ClaimNb", "Exposure", "ClaimAmount", "AvgSeverity",
    "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "VehBrand", "VehGas", "Region", "Density",
]


def validate_freMTPL2(df: pd.DataFrame, task: str) -> None:
    """Validate a loaded freMTPL2 DataFrame.

    Args:
        df: DataFrame returned by ``load_freMTPL2_freq`` or ``load_freMTPL2_sev``.
        task: ``"freq"`` or ``"sev"``.

    Raises:
        ValueError: on any contract violation.
    """
    label = f"freMTPL2_{task}"
    if task == "freq":
        _check_row_count(df, _FREMTPL2_FREQ_EXPECTED_ROWS, label)
        _check_columns(df, _FREMTPL2_FREQ_REQUIRED_COLS, label)
        _check_no_nulls(df, ["ClaimNb", "Exposure"], label)
        _check_non_negative(df, "ClaimNb", label)
        _check_non_negative(df, "Exposure", label)
        if (df["Exposure"] > 1).any():
            raise ValueError(f"[{label}] Exposure > 1 found after cleaning")
    elif task == "sev":
        _check_row_count(df, _FREMTPL2_SEV_EXPECTED_ROWS, label)
        _check_columns(df, _FREMTPL2_SEV_REQUIRED_COLS, label)
        _check_no_nulls(df, ["ClaimNb", "ClaimAmount", "AvgSeverity"], label)
        _check_non_negative(df, "ClaimNb", label)
        if (df["ClaimNb"] == 0).any():
            raise ValueError(f"[{label}] Severity dataset contains rows with ClaimNb=0")
        if (df["ClaimAmount"] <= 0).any():
            raise ValueError(f"[{label}] Non-positive ClaimAmount values found")
        if (df["AvgSeverity"] <= 0).any():
            raise ValueError(f"[{label}] Non-positive AvgSeverity values found")
    else:
        raise ValueError(f"Unknown task '{task}'; expected 'freq' or 'sev'")
    logger.info("[%s] All contracts passed", label)


# ──────────────────────────────────────────────────────────────────────────────
# beMTPL97 contracts
# ──────────────────────────────────────────────────────────────────────────────

_BEMTPL97_FREQ_EXPECTED_ROWS = 163_212
_BEMTPL97_SEV_EXPECTED_ROWS = 18_276   # post-filter; observed on first run

_BEMTPL97_FREQ_REQUIRED_COLS = [
    "nclaims", "expo",
    "coverage", "sex", "fuel", "use", "fleet",
    "bm", "power", "agec", "ageph",
]
_BEMTPL97_SEV_REQUIRED_COLS = _BEMTPL97_FREQ_REQUIRED_COLS + ["amount", "AvgSeverity"]


def validate_beMTPL97(df: pd.DataFrame, task: str) -> None:
    """Validate a loaded beMTPL97 DataFrame.

    Args:
        df: DataFrame returned by ``load_beMTPL97_freq`` or ``load_beMTPL97_sev``.
        task: ``"freq"`` or ``"sev"``.

    Raises:
        ValueError: on any contract violation.
    """
    label = f"beMTPL97_{task}"
    if task == "freq":
        _check_row_count(df, _BEMTPL97_FREQ_EXPECTED_ROWS, label)
        _check_columns(df, _BEMTPL97_FREQ_REQUIRED_COLS, label)
        _check_no_nulls(df, ["nclaims", "expo"], label)
        _check_non_negative(df, "nclaims", label)
        _check_non_negative(df, "expo", label)
        if (df["expo"] > 1).any():
            raise ValueError(f"[{label}] expo > 1 found after cleaning")
    elif task == "sev":
        _check_row_count(df, _BEMTPL97_SEV_EXPECTED_ROWS, label)
        _check_columns(df, _BEMTPL97_SEV_REQUIRED_COLS, label)
        _check_no_nulls(df, ["nclaims", "amount", "AvgSeverity"], label)
        if (df["nclaims"] == 0).any():
            raise ValueError(f"[{label}] Severity dataset contains rows with nclaims=0")
        if (df["amount"] <= 0).any():
            raise ValueError(f"[{label}] Non-positive amount values found")
        if (df["AvgSeverity"] <= 0).any():
            raise ValueError(f"[{label}] Non-positive AvgSeverity values found")
    else:
        raise ValueError(f"Unknown task '{task}'; expected 'freq' or 'sev'")
    logger.info("[%s] All contracts passed", label)


# ──────────────────────────────────────────────────────────────────────────────
# Post-split fold-level contract
# ──────────────────────────────────────────────────────────────────────────────

def validate_fold_claim_rates(
    df: pd.DataFrame,
    fold_indices: list[np.ndarray],
    claim_col: str,
    label: str,
) -> None:
    """Assert that per-fold claim rates stay within ±0.5 pp of the overall rate.

    Args:
        df: full (unsplit) DataFrame.
        fold_indices: list of test-fold index arrays (one per fold).
        claim_col: binary claim indicator column (1 if claim, 0 otherwise).
        label: dataset label for logging/error messages.

    Raises:
        ValueError: if any fold's claim rate deviates by more than 0.5 pp.
    """
    overall_rate = df[claim_col].mean()
    logger.info("[%s] Overall claim rate: %.4f", label, overall_rate)
    for i, test_idx in enumerate(fold_indices):
        fold_rate = df.loc[test_idx, claim_col].mean()
        deviation = abs(fold_rate - overall_rate)
        if deviation > FOLD_RATE_TOL:
            raise ValueError(
                f"[{label}] Fold {i} claim rate {fold_rate:.4f} deviates "
                f"{deviation:.4f} from overall {overall_rate:.4f} "
                f"(tolerance {FOLD_RATE_TOL})"
            )
        logger.info(
            "[%s] Fold %d claim rate: %.4f (deviation %.4f pp — OK)",
            label, i, fold_rate, deviation * 100,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame, dataset: str, task: str) -> None:
    """Dispatch to the appropriate validator.

    Args:
        df: loaded DataFrame.
        dataset: ``"freMTPL2"`` or ``"beMTPL97"``.
        task: ``"freq"`` or ``"sev"``.
    """
    if dataset == "freMTPL2":
        validate_freMTPL2(df, task)
    elif dataset == "beMTPL97":
        validate_beMTPL97(df, task)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
