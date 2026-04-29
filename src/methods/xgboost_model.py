"""XGBoost wrappers with fixed Henckaerts et al. (2021) hyperparameters.

Frequency: objective = count:poisson, log-exposure enters via base_margin.
Severity:  objective = reg:gamma, ClaimNb as sample_weight.

Hyperparameters (fixed, not tuned):
    max_depth      = 3
    n_estimators   = 500
    learning_rate  = 0.01
    subsample      = 0.75
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# ── Fixed hyperparameters (Henckaerts et al. 2021) ────────────────────────────
_HENCKAERTS_PARAMS = dict(
    max_depth=3,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.75,
    tree_method="hist",   # fast histogram method
    random_state=42,
    verbosity=0,
)


class PoissonXGBoost:
    """XGBoost Poisson regressor for frequency modelling.

    The log-exposure offset is passed as ``base_margin`` in the DMatrix so
    that the model predicts expected counts relative to exposure.  Predictions
    are converted back to rates by dividing by exposure.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
    ) -> "PoissonXGBoost":
        """Fit the Poisson XGBoost model.

        Args:
            X_train: encoded feature matrix.
            y_train: claim counts.
            sample_weight: exposure — passed as sample_weight (secondary to
                base_margin but kept for interface parity; XGBoost Poisson
                with base_margin is the primary exposure-handling mechanism).
            log_exposure: log(Exposure) used as base_margin offset.
        """
        self._feature_names = list(X_train.columns)
        self._model = xgb.XGBRegressor(
            objective="count:poisson",
            **_HENCKAERTS_PARAMS,
        )
        fit_params: dict = {}
        if log_exposure is not None:
            fit_params["base_margin"] = log_exposure
        self._model.fit(
            X_train,
            y_train,
            **fit_params,
        )
        logger.info(
            "PoissonXGBoost fitted: %d estimators, %d features",
            self._model.n_estimators,
            len(self._feature_names),
        )
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted claim rates.

        XGBoost with ``count:poisson`` and base_margin returns expected counts;
        divide by exposure to recover rate.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        predict_params: dict = {}
        if log_exposure is not None:
            predict_params["base_margin"] = log_exposure
        mu_count = self._model.predict(X_test, **predict_params)
        exposure = np.exp(log_exposure) if log_exposure is not None else np.ones(len(X_test))
        return mu_count / exposure

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class GammaXGBoost:
    """XGBoost Gamma regressor for severity modelling."""

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
    ) -> "GammaXGBoost":
        """Fit the Gamma XGBoost model.

        Args:
            X_train: encoded feature matrix.
            y_train: average severity (ClaimAmount / ClaimNb).
            sample_weight: ClaimNb — frequency weights for Gamma loss.
            log_exposure: unused for severity; accepted for interface parity.
        """
        self._feature_names = list(X_train.columns)
        self._model = xgb.XGBRegressor(
            objective="reg:gamma",
            **_HENCKAERTS_PARAMS,
        )
        self._model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
        )
        logger.info(
            "GammaXGBoost fitted: %d estimators, %d features",
            self._model.n_estimators,
            len(self._feature_names),
        )
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted average severity."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict(X_test)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class BinaryXGBoost:
    """XGBoost binary classifier for binary classification tasks.

    Uses ``binary:logistic`` objective; ``predict()`` returns probabilities.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
    ) -> "BinaryXGBoost":
        """Fit the binary XGBoost classifier.

        Args:
            X_train: encoded feature matrix.
            y_train: binary labels (0/1).
            sample_weight: accepted for interface parity; not used.
            log_exposure: accepted for interface parity; not used.
        """
        self._feature_names = list(X_train.columns)
        self._model = xgb.XGBClassifier(
            objective="binary:logistic",
            **_HENCKAERTS_PARAMS,
        )
        self._model.fit(X_train, y_train.astype(int))
        logger.info(
            "BinaryXGBoost fitted: %d estimators, %d features",
            self._model.n_estimators,
            len(self._feature_names),
        )
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


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_xgboost(task: str) -> PoissonXGBoost | GammaXGBoost | BinaryXGBoost:
    """Return the appropriate XGBoost model for the given task.

    Args:
        task: ``"freq"``, ``"sev"``, or ``"clf"``.
    """
    if task == "freq":
        return PoissonXGBoost()
    if task == "sev":
        return GammaXGBoost()
    if task == "clf":
        return BinaryXGBoost()
    raise ValueError(f"Unknown task '{task}'")
