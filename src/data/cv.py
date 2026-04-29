"""Cross-validation splitting utilities.

Provides a single shared 5-fold CV split, stratified by the claim indicator
(ClaimNb > 0 or nclaims > 0) using round-robin assignment.  The same split
object is reused by all models within an experiment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.contracts import validate_fold_claim_rates

logger = logging.getLogger(__name__)


def _claim_indicator_col(dataset: str) -> str:
    """Return the claim indicator column name for the given dataset."""
    if dataset == "freMTPL2":
        return "ClaimNb"
    if dataset == "beMTPL97":
        return "nclaims"
    if dataset == "fretelematic":
        # Claim_binary is 0/1; comparison > 0 in make_cv_splits works directly.
        return "Claim_binary"
    raise ValueError(f"Unknown dataset '{dataset}'")


def make_cv_splits(
    df: pd.DataFrame,
    dataset: str,
    cfg: dict,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified 5-fold CV splits using round-robin fold assignment.

    Policies are sorted by their claim indicator (1 = claimant, 0 = non-claimant)
    and then assigned to folds in round-robin order so that each fold receives
    approximately the same proportion of claimants.  This matches the approach
    described in the experiment outline.

    Args:
        df: full cleaned DataFrame (before any train/test split).
        dataset: ``"freMTPL2"`` or ``"beMTPL97"``.
        cfg: experiment config dict (must contain ``cv_seed`` and ``cv_folds``).

    Returns:
        List of ``(train_idx, test_idx)`` pairs — one per fold.  Indices are
        integer positions (iloc-compatible).
    """
    n_folds: int = cfg["cv_folds"]
    seed: int = cfg["cv_seed"]
    rng = np.random.default_rng(seed)

    claim_col = _claim_indicator_col(dataset)
    claim_indicator = (df[claim_col] > 0).astype(int).values

    n = len(df)
    positions = np.arange(n)

    # Shuffle within each stratum, then interleave via round-robin
    claimant_idx = positions[claim_indicator == 1]
    non_claimant_idx = positions[claim_indicator == 0]
    rng.shuffle(claimant_idx)
    rng.shuffle(non_claimant_idx)

    # Assign fold labels via round-robin within each stratum
    fold_labels = np.empty(n, dtype=int)
    for i, idx in enumerate(claimant_idx):
        fold_labels[idx] = i % n_folds
    for i, idx in enumerate(non_claimant_idx):
        fold_labels[idx] = i % n_folds

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        test_mask = fold_labels == fold
        train_idx = positions[~test_mask]
        test_idx = positions[test_mask]
        splits.append((train_idx, test_idx))

    # Log fold sizes
    for fold, (train_idx, test_idx) in enumerate(splits):
        logger.info(
            "Fold %d: %d train / %d test", fold, len(train_idx), len(test_idx)
        )

    # Validate claim rates per fold
    test_indices = [test_idx for _, test_idx in splits]
    validate_fold_claim_rates(
        df.reset_index(drop=True),
        test_indices,
        claim_col,
        label=f"{dataset}_cv",
    )

    return splits


def get_fold(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for a given fold index.

    Args:
        df: full DataFrame (must be iloc-indexable with the stored indices).
        splits: output of ``make_cv_splits``.
        fold: zero-based fold index.

    Returns:
        Tuple of (train_df, test_df) with reset indices.
    """
    train_idx, test_idx = splits[fold]
    df_reset = df.reset_index(drop=True)
    return (
        df_reset.iloc[train_idx].reset_index(drop=True),
        df_reset.iloc[test_idx].reset_index(drop=True),
    )
