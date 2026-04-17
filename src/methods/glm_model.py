"""GLM wrapper for Poisson (frequency) and Gamma (severity) regression.

Uses statsmodels with a log link.  Frequency models include a log-exposure
offset; severity models use ClaimNb as observation weights.

Interface
---------
All method wrappers expose the same interface so experiment scripts can call
them interchangeably:

    model.fit(X_train, y_train, sample_weight, log_exposure)
    preds = model.predict(X_test, log_exposure)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class PoissonGLM:
    """Poisson GLM with log link and log-exposure offset (frequency task)."""

    def __init__(self) -> None:
        self._result = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
    ) -> "PoissonGLM":
        """Fit the Poisson GLM.

        Args:
            X_train: encoded feature matrix (no intercept column needed —
                statsmodels adds one automatically via ``add_constant``).
            y_train: claim counts.
            sample_weight: ignored for Poisson GLM (exposure enters via offset).
            log_exposure: log(Exposure) offset; required for frequency models.
        """
        self._feature_names = list(X_train.columns)
        X = sm.add_constant(X_train.to_numpy(dtype=float), has_constant="add")
        offset = log_exposure if log_exposure is not None else np.zeros(len(y_train))
        glm = sm.GLM(
            y_train,
            X,
            family=sm.families.Poisson(sm.families.links.Log()),
            offset=offset,
        )
        self._result = glm.fit(disp=False)
        logger.info(
            "PoissonGLM fitted: %d params, converged=%s",
            len(self._result.params),
            self._result.converged,
        )
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted claim rates (counts / exposure).

        The offset converts linear predictor to expected counts; dividing by
        exposure recovers the rate.
        """
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet")
        X = sm.add_constant(X_test.to_numpy(dtype=float), has_constant="add")
        offset = log_exposure if log_exposure is not None else np.zeros(len(X_test))
        mu_count = self._result.predict(X, offset=offset)
        exposure = np.exp(offset)
        return mu_count / exposure

    def get_coefficients(self) -> pd.Series:
        """Return fitted coefficients as a Series indexed by feature name."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet")
        names = ["const"] + self._feature_names
        return pd.Series(self._result.params, index=names, name="coefficient")


class GammaGLM:
    """Gamma GLM with log link (severity task).

    ClaimNb is passed as ``var_weights`` (frequency weights), which is
    statsmodels' term for the observation weights in a GLM.
    """

    def __init__(self) -> None:
        self._result = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
    ) -> "GammaGLM":
        """Fit the Gamma GLM.

        Args:
            X_train: encoded feature matrix.
            y_train: average severity (ClaimAmount / ClaimNb).
            sample_weight: ClaimNb — used as frequency weights.
            log_exposure: unused for severity; accepted for interface parity.
        """
        self._feature_names = list(X_train.columns)
        X = sm.add_constant(X_train.to_numpy(dtype=float), has_constant="add")
        glm = sm.GLM(
            y_train,
            X,
            family=sm.families.Gamma(sm.families.links.Log()),
            var_weights=sample_weight,
        )
        self._result = glm.fit(disp=False)
        logger.info(
            "GammaGLM fitted: %d params, converged=%s",
            len(self._result.params),
            self._result.converged,
        )
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted average severity."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet")
        X = sm.add_constant(X_test.to_numpy(dtype=float), has_constant="add")
        return self._result.predict(X)

    def get_coefficients(self) -> pd.Series:
        """Return fitted coefficients as a Series indexed by feature name."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet")
        names = ["const"] + self._feature_names
        return pd.Series(self._result.params, index=names, name="coefficient")


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_glm(task: str) -> PoissonGLM | GammaGLM:
    """Return the appropriate GLM for the given task.

    Args:
        task: ``"freq"`` or ``"sev"``.
    """
    if task == "freq":
        return PoissonGLM()
    if task == "sev":
        return GammaGLM()
    raise ValueError(f"Unknown task '{task}'")
