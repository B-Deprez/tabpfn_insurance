"""TabPFN wrappers for insurance pricing (frequency, severity, and binary classification).

TabPFN handles mixed data types (strings, integers, floats) and missing
values natively — no feature encoding is required.  Raw feature columns from
``get_raw_features()`` are passed directly to the model.

The internal ``ModelVersion`` (``v2_5``, ``v2_6`` or ``v3``) is selected per
call via ``create_default_for_version``.  The default is ``v3``.

Exposure strategy B is used throughout:
    Frequency  : response = ClaimNb / Exposure (annualised rate)
    Severity   : response = log(AvgSeverity) (log-transformed),
                 predictions inverted with exp()

TabPFNRegressor.fit() accepts only (X, y) — no sample_weight argument.

Device auto-detection order: CUDA → MPS → CPU.

Training is capped at ``max_train_size`` rows.  When the training fold
exceeds this limit a random subsample is drawn (seed = fold_seed).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping of public string labels → tabpfn.constants.ModelVersion enum members.
# Resolved lazily inside helpers so importing this module does not require tabpfn.
_VERSION_LABELS = ("v2_5", "v2_6", "v3")

# Upstream pretraining-context ceilings enforced by TabPFN. Inputs larger than
# this raise ``TabPFNValidationError`` unless ``ignore_pretraining_limits=True``
# is passed.  Versions not listed here have no fixed ceiling.
_VERSION_MAX_TRAIN_SIZE = {"v2_5": 50_000, "v2_6": 100_000}


def _effective_max_train_size(version: str, requested: int) -> int:
    """Return ``min(requested, upstream_ceiling)`` for the given version."""
    cap = _VERSION_MAX_TRAIN_SIZE.get(version)
    return min(requested, cap) if cap is not None else requested


def _resolve_model_version(version: str):
    """Translate ``"v2_5"`` / ``"v2_6"`` / ``"v3"`` to the corresponding ``ModelVersion`` enum."""
    from tabpfn.constants import ModelVersion
    if version == "v2_5":
        return ModelVersion.V2_5
    if version == "v2_6":
        return ModelVersion.V2_6
    if version == "v3":
        return ModelVersion.V3
    raise ValueError(
        f"Unknown tabpfn_version {version!r}; expected one of {_VERSION_LABELS}"
    )


def _make_regressor(version: str, device: str):
    """Build a ``TabPFNRegressor`` for the requested internal model version."""
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version(
        _resolve_model_version(version), device=device,
    )


def _make_classifier(version: str, device: str):
    """Build a ``TabPFNClassifier`` for the requested internal model version."""
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier.create_default_for_version(
        _resolve_model_version(version), device=device,
    )


def _detect_device() -> str:
    """Return the best available device string for TabPFN."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X_sub, y_sub) subsampled to at most ``max_size`` rows."""
    n = len(X)
    if n <= max_size:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_size, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]


class TabPFNFreq:
    """TabPFN regressor for frequency modelling (Strategy B).

    Raw unencoded features are passed directly to TabPFN.
    Response = ClaimNb / Exposure (annualised rate).
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNFreq: device=%s, max_train_size=%d, version=%s",
            self._device, max_train_size, tabpfn_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNFreq":
        """Fit TabPFN on raw features with Strategy B response.

        Args:
            X_train: raw (unencoded) feature DataFrame.
            y_train: claim counts (NOT rates — conversion is done here).
            sample_weight: exposure array used to compute the rate response.
            log_exposure: unused; accepted for interface parity with GLM/XGBoost.
            fold_seed: RNG seed for the subsample draw.
        """
        self._feature_names = list(X_train.columns)
        exposure = sample_weight if sample_weight is not None else np.ones(len(y_train))

        # Strategy B: response = annualised rate
        y_rate = y_train / np.maximum(exposure, 1e-10)

        X_sub, y_sub = _subsample(X_train, y_rate, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_train):
            logger.info(
                "TabPFNFreq: subsampled %d → %d rows (seed=%d)",
                len(X_train), len(X_sub), fold_seed,
            )

        self._model = _make_regressor(self.tabpfn_version, self._device)
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNFreq fitted on %d rows", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted claim rates."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict(X_test)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNSev:
    """TabPFN regressor for severity modelling (Strategy B + log-transform).

    Raw unencoded features are passed directly to TabPFN.
    Response = log(AvgSeverity); predictions are inverted with exp().
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNSev: device=%s, max_train_size=%d, version=%s",
            self._device, max_train_size, tabpfn_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNSev":
        """Fit TabPFN on log-transformed average severity.

        Args:
            X_train: raw (unencoded) feature DataFrame.
            y_train: average severity (AvgSeverity = ClaimAmount / ClaimNb).
            sample_weight: accepted for interface parity; not used (TabPFN
                does not support sample_weight in fit()).
            log_exposure: unused for severity; accepted for interface parity.
            fold_seed: RNG seed for the subsample draw.
        """
        self._feature_names = list(X_train.columns)

        # Log-transform response (CLAUDE.md §Feature Encoding)
        log_y = np.log(np.maximum(y_train, 1e-10))

        X_sub, y_sub = _subsample(X_train, log_y, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_train):
            logger.info(
                "TabPFNSev: subsampled %d → %d rows (seed=%d)",
                len(X_train), len(X_sub), fold_seed,
            )

        self._model = _make_regressor(self.tabpfn_version, self._device)
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNSev fitted on %d rows", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted average severity (exp of log-space prediction)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return np.exp(self._model.predict(X_test))

    def get_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return SHAP values in log-space from TabPFN's built-in explainer."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        if hasattr(self._model, "get_shap_values"):
            return self._model.get_shap_values(X_test)
        logger.warning("TabPFN built-in SHAP not available; falling back to KernelExplainer")
        import shap
        bg = X_test.iloc[:min(100, len(X_test))]
        explainer = shap.KernelExplainer(self._model.predict, bg)
        return explainer.shap_values(X_test, nsamples=100)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNClf:
    """TabPFN classifier for binary classification tasks.

    Raw unencoded features are passed directly to TabPFN.
    ``predict()`` returns the probability of class 1 (claim).
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNClf: device=%s, max_train_size=%d, version=%s",
            self._device, max_train_size, tabpfn_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNClf":
        """Fit TabPFN classifier on raw features.

        Args:
            X_train: raw (unencoded) feature DataFrame.
            y_train: binary labels (0/1).
            sample_weight: accepted for interface parity; not used.
            log_exposure: accepted for interface parity; not used.
            fold_seed: RNG seed for the subsample draw.
        """
        self._feature_names = list(X_train.columns)
        y_int = y_train.astype(int)

        X_sub, y_sub = _subsample(X_train, y_int, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_train):
            logger.info(
                "TabPFNClf: subsampled %d → %d rows (seed=%d)",
                len(X_train), len(X_sub), fold_seed,
            )

        self._model = _make_classifier(self.tabpfn_version, self._device)
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNClf fitted on %d rows", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted claim probabilities (probability of class 1)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict_proba(X_test)[:, 1]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNFreqWithShap(TabPFNFreq):
    """TabPFNFreq extended with SHAP computation (used by run_q3_shap.py)."""

    def get_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return SHAP values from TabPFN's built-in explainer or KernelExplainer."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        if hasattr(self._model, "get_shap_values"):
            return self._model.get_shap_values(X_test)
        logger.warning("TabPFN built-in SHAP not available; falling back to KernelExplainer")
        import shap
        bg = X_test.iloc[:min(100, len(X_test))]
        explainer = shap.KernelExplainer(self._model.predict, bg)
        return explainer.shap_values(X_test, nsamples=100)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_tabpfn(
    task: str,
    max_train_size: int = 100_000,
    shap: bool = False,
    tabpfn_version: str = "v3",
):
    """Return the appropriate TabPFN wrapper for the given task.

    Args:
        task: ``"freq"``, ``"sev"``, or ``"clf"``.
        max_train_size: maximum training rows.
        shap: if True, return a SHAP-capable variant for Q3 (freq/sev only).
        tabpfn_version: internal ``ModelVersion`` to use — ``"v2_5"``, ``"v2_6"`` or ``"v3"``.
    """
    if task == "freq":
        cls = TabPFNFreqWithShap if shap else TabPFNFreq
        return cls(max_train_size, tabpfn_version=tabpfn_version)
    if task == "sev":
        return TabPFNSev(max_train_size, tabpfn_version=tabpfn_version)
    if task == "clf":
        return TabPFNClf(max_train_size, tabpfn_version=tabpfn_version)
    raise ValueError(f"Unknown task '{task}'")
