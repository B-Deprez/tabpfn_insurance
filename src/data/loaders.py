"""Data loading utilities for freMTPL2 and beMTPL97 datasets.

Datasets are loaded via rpy2 from the CASdatasets R package and optionally
cached as parquet files in data/processed/ for faster subsequent access.
Cleaning steps follow Wüthrich & Buser (2021) for freMTPL2 and
Henckaerts et al. (2021) for beMTPL97.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CFG_PATH = PROJECT_ROOT / "config" / "data.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# R_HOME detection
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_r_home() -> None:
    """Set R_HOME if not already in the environment.

    rpy2 requires R_HOME to be set before it is imported.  When running inside
    a conda environment, R is installed at ``{sys.prefix}/lib/R`` but the
    shell variable is not always propagated automatically.  This function
    checks three locations in order:

    1. ``R_HOME`` already set — nothing to do.
    2. ``{sys.prefix}/lib/R`` exists (conda environment with r-base) — use it.
    3. Ask the ``R`` binary on PATH via ``R RHOME`` — use its output.
    """
    if os.environ.get("R_HOME"):
        logger.debug("R_HOME already set: %s", os.environ["R_HOME"])
        return

    # Conda environment path
    conda_r = Path(sys.prefix) / "lib" / "R"
    if conda_r.exists():
        os.environ["R_HOME"] = str(conda_r)
        logger.info("Set R_HOME from conda prefix: %s", conda_r)
        return

    # Fallback: ask the R binary
    import shutil
    r_exec = shutil.which("R")
    if r_exec:
        try:
            result = subprocess.run(
                [r_exec, "RHOME"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                r_home = result.stdout.strip()
                os.environ["R_HOME"] = r_home
                logger.info("Set R_HOME from `R RHOME`: %s", r_home)
                return
        except Exception as exc:
            logger.warning("Could not run `R RHOME`: %s", exc)

    raise RuntimeError(
        "Could not determine R_HOME automatically.\n"
        "Fix: activate the tabpfn-insurance conda environment and re-run, or "
        "set R_HOME manually:\n"
        "  export R_HOME=$(R RHOME)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_data_cfg() -> dict:
    """Load data.yaml configuration."""
    with open(DATA_CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_r_dataset(r_object_name: str) -> pd.DataFrame:
    """Load a named object from CASdatasets via rpy2 and return as DataFrame."""
    _ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    logger.info("Loading R object '%s' from CASdatasets", r_object_name)
    ro.r("library(CASdatasets)")
    ro.r(f"data({r_object_name})")
    r_obj = ro.r[r_object_name]
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.rpy2py(r_obj)
    logger.info("Loaded %d rows, %d columns from '%s'", len(df), df.shape[1], r_object_name)
    return df.reset_index(drop=True)


def _apply_cleaning(df: pd.DataFrame, rules: list[dict], label: str) -> pd.DataFrame:
    """Apply a sequence of row-filter rules, logging removed records."""
    n_before = len(df)
    for rule in rules:
        mask = df.eval(rule["filter"])
        n_removed = (~mask).sum()
        if n_removed:
            logger.info(
                "[%s] Removed %d rows — %s (filter: %s)",
                label, n_removed, rule["description"], rule["filter"],
            )
        df = df[mask].copy()
    n_after = len(df)
    logger.info(
        "[%s] Cleaning complete: %d → %d rows (%d removed, %.2f%%)",
        label, n_before, n_after, n_before - n_after,
        100.0 * (n_before - n_after) / n_before,
    )
    return df


def _cache_path(name: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"{name}.parquet"


def _try_cache(name: str) -> pd.DataFrame | None:
    p = _cache_path(name)
    if p.exists():
        logger.info("Cache hit — loading '%s' from %s", name, p)
        return pd.read_parquet(p)
    return None


def _write_cache(df: pd.DataFrame, name: str) -> None:
    p = _cache_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Cached '%s' → %s (%d rows)", name, p, len(df))


# ──────────────────────────────────────────────────────────────────────────────
# Public loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_freMTPL2_freq(cfg: dict | None = None) -> pd.DataFrame:
    """Load and clean freMTPL2freq for frequency modelling.

    Returns a DataFrame with one row per policy.  All cleaning steps from
    config/data.yaml (Wüthrich & Buser 2021) are applied before returning.
    """
    if cfg is None:
        cfg = _load_data_cfg()
    use_cache = cfg.get("cache_processed", True)

    if use_cache:
        cached = _try_cache("freMTPL2_freq")
        if cached is not None:
            return cached

    ds_cfg = cfg["datasets"]["freMTPL2"]["freq"]
    df = _load_r_dataset(ds_cfg["r_object"])
    df = _apply_cleaning(df, ds_cfg["cleaning"], "freMTPL2_freq")

    if use_cache:
        _write_cache(df, "freMTPL2_freq")
    return df


def load_freMTPL2_sev(cfg: dict | None = None) -> pd.DataFrame:
    """Load and clean freMTPL2 for severity modelling.

    freMTPL2sev (IDpol, ClaimAmount) is aggregated by IDpol and joined to
    freMTPL2freq (filtered to ClaimNb > 0) to obtain the full feature matrix.
    The column ``AvgSeverity`` (= ClaimAmount / ClaimNb) is added as the
    model target; the original ``ClaimNb`` column serves as the sample weight.
    """
    if cfg is None:
        cfg = _load_data_cfg()
    use_cache = cfg.get("cache_processed", True)

    if use_cache:
        cached = _try_cache("freMTPL2_sev")
        if cached is not None:
            return cached

    freq_cfg = cfg["datasets"]["freMTPL2"]["freq"]
    sev_cfg = cfg["datasets"]["freMTPL2"]["sev"]

    # ── Load and clean frequency base ─────────────────────────────────────────
    freq_df = _load_r_dataset(freq_cfg["r_object"])
    freq_df = _apply_cleaning(freq_df, freq_cfg["cleaning"], "freMTPL2_freq")
    # Keep only policies with at least one claim
    freq_with_claims = freq_df[freq_df["ClaimNb"] > 0].copy()
    logger.info("freMTPL2_freq: %d policies with ClaimNb > 0", len(freq_with_claims))

    # ── Load severity claims table ────────────────────────────────────────────
    sev_raw = _load_r_dataset(sev_cfg["r_object_sev"])
    logger.info("freMTPL2sev raw: %d claim rows", len(sev_raw))
    # Aggregate multiple claim rows per policy (sum amounts)
    sev_agg = (
        sev_raw.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    )

    # ── Join ──────────────────────────────────────────────────────────────────
    df = freq_with_claims.merge(sev_agg, on="IDpol", how="inner")
    logger.info("After join: %d rows", len(df))

    # ── Apply severity-level cleaning ─────────────────────────────────────────
    df = _apply_cleaning(df, sev_cfg["cleaning"], "freMTPL2_sev")

    # ── Derived target ────────────────────────────────────────────────────────
    df["AvgSeverity"] = df["ClaimAmount"] / df["ClaimNb"]

    if use_cache:
        _write_cache(df, "freMTPL2_sev")
    return df


def load_beMTPL97_freq(cfg: dict | None = None) -> pd.DataFrame:
    """Load and clean beMTPL97 for frequency modelling.

    Returns a DataFrame with one row per policy-year exposure.  Cleaning
    follows Henckaerts et al. (2021).
    """
    if cfg is None:
        cfg = _load_data_cfg()
    use_cache = cfg.get("cache_processed", True)

    if use_cache:
        cached = _try_cache("beMTPL97_freq")
        if cached is not None:
            return cached

    ds_cfg = cfg["datasets"]["beMTPL97"]
    df = _load_r_dataset(ds_cfg["r_object"])
    df = _apply_cleaning(df, ds_cfg["cleaning"], "beMTPL97_freq")

    # Cast fleet from string to int if needed
    if df["fleet"].dtype == object:
        df["fleet"] = df["fleet"].astype(int)

    if use_cache:
        _write_cache(df, "beMTPL97_freq")
    return df


def load_beMTPL97_sev(cfg: dict | None = None) -> pd.DataFrame:
    """Load and clean beMTPL97 for severity modelling.

    Filters to policies with nclaims > 0 and adds ``AvgSeverity``
    (= amount / nclaims) as the model target.
    """
    if cfg is None:
        cfg = _load_data_cfg()
    use_cache = cfg.get("cache_processed", True)

    if use_cache:
        cached = _try_cache("beMTPL97_sev")
        if cached is not None:
            return cached

    ds_cfg = cfg["datasets"]["beMTPL97"]
    df = _load_r_dataset(ds_cfg["r_object"])
    df = _apply_cleaning(df, ds_cfg["cleaning"], "beMTPL97_sev")

    # Filter to policies with at least one claim
    df = df[df["nclaims"] > 0].copy()
    logger.info("beMTPL97_sev: %d policies with nclaims > 0", len(df))

    # Cast fleet from string to int if needed
    if df["fleet"].dtype == object:
        df["fleet"] = df["fleet"].astype(int)

    df["AvgSeverity"] = df["amount"] / df["nclaims"]

    if use_cache:
        _write_cache(df, "beMTPL97_sev")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Convenience dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset: str, task: str, cfg: dict | None = None) -> pd.DataFrame:
    """Dispatch to the appropriate loader.

    Args:
        dataset: ``"freMTPL2"`` or ``"beMTPL97"``
        task: ``"freq"`` or ``"sev"``
        cfg: optional pre-loaded data config dict

    Returns:
        Cleaned DataFrame ready for CV splitting and preprocessing.
    """
    loaders = {
        ("freMTPL2", "freq"): load_freMTPL2_freq,
        ("freMTPL2", "sev"): load_freMTPL2_sev,
        ("beMTPL97", "freq"): load_beMTPL97_freq,
        ("beMTPL97", "sev"): load_beMTPL97_sev,
    }
    key = (dataset, task)
    if key not in loaders:
        raise ValueError(f"Unknown (dataset, task) combination: {key}")
    return loaders[key](cfg=cfg)
