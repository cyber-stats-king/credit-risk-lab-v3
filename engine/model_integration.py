"""
CreditRisk Lab
Model Integration Engine

Connects the existing PD, EAD and LGD modelling components
to the central portfolio manager and model registry.

Responsibilities:
1. Register fitted models.
2. Load registered models.
3. Score portfolio accounts.
4. Merge predictions using loan_id.
5. Track model scoring status.
6. Provide a unified interface for the integrated credit risk tool.

Important:
The existing model development pages remain responsible for
model training and algorithm selection.

This module handles model registration and portfolio scoring.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from engine.portfolio_manager import (
    PortfolioManager,
    PortfolioError,
    PredictionMergeError,
)

from engine.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelNotFoundError,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelIntegrationError(Exception):
    """Base exception for model integration errors."""


class ModelScoringError(ModelIntegrationError):
    """Raised when model scoring fails."""


# ============================================================
# MODEL INTEGRATION ENGINE
# ============================================================

class ModelIntegration:

    def __init__(
        self,
        portfolio_manager=None,
        model_registry=None,
    ):

        self.portfolio_manager = (
            portfolio_manager
            if portfolio_manager is not None
            else PortfolioManager()
        )

        self.model_registry = (
            model_registry
            if model_registry is not None
            else ModelRegistry()
        )

    # ========================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================

    def load_portfolio(self, portfolio, source="in_memory"):

        return self.portfolio_manager.load_portfolio(
            portfolio,
            source=source
        )

    def get_original_portfolio(self):

        return self.portfolio_manager.get_original_portfolio()

    def get_prediction_portfolio(self):

        return self.portfolio_manager.get_prediction_portfolio()

    # ========================================================
    # REGISTER MODELS
    # ========================================================

    def register_pd_model(
        self,
        model,
        algorithm,
        features,
        metrics=None,
        target="default",
        metadata=None,
    ):

        return self._register_model(
            model_type="PD",
            model=model,
            algorithm=algorithm,
            features=features,
            metrics=metrics,
            target=target,
            metadata=metadata,
        )

    def register_ead_model(
        self,
        model,
        algorithm,
        features,
        metrics=None,
        target="ccf",
        metadata=None,
    ):

        return self._register_model(
            model_type="EAD",
            model=model,
            algorithm=algorithm,
            features=features,
            metrics=metrics,
            target=target,
            metadata=metadata,
        )

    def register_lgd_model(
        self,
        model,
        algorithm,
        features,
        metrics=None,
        target="adjusted_lgd",
        metadata=None,
    ):

        return self._register_model(
            model_type="LGD",
            model=model,
            algorithm=algorithm,
            features=features,
            metrics=metrics,
            target=target,
            metadata=metadata,
        )

    def _register_model(
        self,
        model_type,
        model,
        algorithm,
        features,
        metrics=None,
        target=None,
        metadata=None,
    ):

        try:

            return self.model_registry.register_model(
                model_type=model_type,
                model=model,
                algorithm=algorithm,
                features=features,
                metrics=metrics or {},
                target=target,
                metadata=metadata or {},
            )

        except ModelRegistryError as error:

            raise ModelIntegrationError(
                f"Could not register {model_type} model: {error}"
            ) from error

    # ========================================================
    # MODEL RETRIEVAL
    # ========================================================

    def load_model(self, model_type, model_id=None):

        try:

            return self.model_registry.load_model(
                model_type=model_type,
                model_id=model_id,
            )

        except ModelRegistryError as error:

            raise ModelIntegrationError(
                f"Could not load {model_type} model: {error}"
            ) from error

    # ========================================================
    # MODEL SCORING
    # ========================================================

    def score_pd(self, model=None, model_id=None):

        portfolio = self.get_original_portfolio()

        if model is None:
            model = self.load_model("PD", model_id)

        metadata = self.model_registry.get_model_metadata(
            "PD",
            model_id
        )

        features = metadata["features"]

        predictions = self._predict_probability(
            model,
            portfolio,
            features,
            model_type="PD",
        )

        prediction_data = pd.DataFrame({
            "loan_id": portfolio["loan_id"].values,
            "predicted_pd": predictions,
        })

        self._merge_predictions(
            prediction_data,
            "PD"
        )

        return prediction_data

    def score_ead(self, model=None, model_id=None):

        portfolio = self.get_original_portfolio()

        if model is None:
            model = self.load_model("EAD", model_id)

        metadata = self.model_registry.get_model_metadata(
            "EAD",
            model_id
        )

        features = metadata["features"]

        predicted_ccf = self._predict_numeric(
            model,
            portfolio,
            features,
            model_type="EAD",
        )

        if "current_balance" not in portfolio.columns:
            raise ModelScoringError(
                "Portfolio is missing current_balance."
            )

        if "undrawn_amount" not in portfolio.columns:
            raise ModelScoringError(
                "Portfolio is missing undrawn_amount."
            )

        current_balance = pd.to_numeric(
            portfolio["current_balance"],
            errors="coerce"
        )

        undrawn_amount = pd.to_numeric(
            portfolio["undrawn_amount"],
            errors="coerce"
        )

        if current_balance.isna().any() or undrawn_amount.isna().any():
            raise ModelScoringError(
                "EAD calculation fields contain missing or invalid values."
            )

        if (undrawn_amount < 0).any():
            raise ModelScoringError(
                "Undrawn exposure cannot be negative."
            )

        predicted_ead = (
            current_balance
            + predicted_ccf * undrawn_amount
        )

        prediction_data = pd.DataFrame({
            "loan_id": portfolio["loan_id"].values,
            "predicted_ccf": predicted_ccf,
            "predicted_ead": predicted_ead,
        })

        self._merge_predictions(
            prediction_data,
            "EAD"
        )

        return prediction_data

    def score_lgd(self, model=None, model_id=None):

        portfolio = self.get_original_portfolio()

        if model is None:
            model = self.load_model("LGD", model_id)

        metadata = self.model_registry.get_model_metadata(
            "LGD",
            model_id
        )

        features = metadata["features"]

        predicted_lgd = self._predict_numeric(
            model,
            portfolio,
            features,
            model_type="LGD",
        )

        prediction_data = pd.DataFrame({
            "loan_id": portfolio["loan_id"].values,
            "predicted_lgd": predicted_lgd,
        })

        self._merge_predictions(
            prediction_data,
            "LGD"
        )

        return prediction_data

    # ========================================================
    # PREDICTION HELPERS
    # ========================================================

    @staticmethod
    def _prepare_features(portfolio, features, model_type):

        if not features:
            raise ModelScoringError(
                f"{model_type} model has no registered features."
            )

        missing_features = [
            feature
            for feature in features
            if feature not in portfolio.columns
        ]

        if missing_features:
            raise ModelScoringError(
                f"{model_type} scoring is missing features: "
                f"{missing_features}"
            )

        X = portfolio[features].copy()

        if X.empty:
            raise ModelScoringError(
                f"No observations available for {model_type} scoring."
            )

        return X

    def _predict_probability(
        self,
        model,
        portfolio,
        features,
        model_type,
    ):

        X = self._prepare_features(
            portfolio,
            features,
            model_type,
        )

        if not hasattr(model, "predict_proba"):
            raise ModelScoringError(
                f"{model_type} model does not support predict_proba()."
            )

        try:

            probabilities = model.predict_proba(X)

        except Exception as error:

            raise ModelScoringError(
                f"{model_type} prediction failed: {error}"
            ) from error

        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 2 or probabilities.shape[0] != len(X):
            raise ModelScoringError(
                f"{model_type} returned an invalid probability array."
            )

        if probabilities.shape[1] != 2:
            raise ModelScoringError(
                f"{model_type} model must return probabilities "
                "for both non-default and default classes."
            )

        classes = getattr(model, "classes_", None)

        if classes is None or 1 not in classes:
            raise ModelScoringError(
                f"{model_type} model does not expose a default class "
                "label of 1."
            )

        default_index = list(classes).index(1)

        predictions = probabilities[:, default_index]

        self._validate_prediction_values(
            predictions,
            model_type
        )

        return predictions

    def _predict_numeric(
        self,
        model,
        portfolio,
        features,
        model_type,
    ):

        X = self._prepare_features(
            portfolio,
            features,
            model_type,
        )

        try:

            predictions = model.predict(X)

        except Exception as error:

            raise ModelScoringError(
                f"{model_type} prediction failed: {error}"
            ) from error

        predictions = np.asarray(
            predictions,
            dtype=float
        ).reshape(-1)

        self._validate_prediction_values(
            predictions,
            model_type
        )

        if model_type == "LGD":

            if ((predictions < 0) | (predictions > 1)).any():
                raise ModelScoringError(
                    "LGD predictions must be between 0 and 1."
                )

        if model_type == "EAD":

            if ((predictions < 0) | (predictions > 1)).any():
                raise ModelScoringError(
                    "CCF predictions must be between 0 and 1 "
                    "under the current model configuration."
                )

        return predictions

    @staticmethod
    def _validate_prediction_values(predictions, model_type):

        if len(predictions) == 0:
            raise ModelScoringError(
                f"{model_type} produced no predictions."
            )

        if not np.isfinite(predictions).all():
            raise ModelScoringError(
                f"{model_type} produced missing or infinite predictions."
            )

        if ((predictions < 0) | (predictions > 1)).any():
            raise ModelScoringError(
                f"{model_type} predictions must be between 0 and 1."
            )

    # ========================================================
    # PORTFOLIO MERGING
    # ========================================================

    def _merge_predictions(self, predictions, model_type):

        try:

            return self.portfolio_manager.merge_predictions(
                predictions=predictions,
                model_type=model_type,
                allow_partial=False,
            )

        except PredictionMergeError as error:

            raise ModelIntegrationError(
                f"Could not merge {model_type} predictions: {error}"
            ) from error

    # ========================================================
    # SCORING ALL MODELS
    # ========================================================

    def score_all_models(self):

        results = {}

        results["PD"] = self.score_pd()

        results["EAD"] = self.score_ead()

        results["LGD"] = self.score_lgd()

        return self.get_prediction_portfolio()

    # ========================================================
    # STATUS AND REPORTING
    # ========================================================

    def prediction_status(self):

        return self.portfolio_manager.prediction_status()

    def get_prediction_portfolio(self):

        return self.portfolio_manager.get_prediction_portfolio()

    def get_registry_summary(self):

        return pd.DataFrame(
            self.model_registry.registry_summary()
        )

    def save_portfolio(self):

        self.portfolio_manager.save_portfolio()

    def sync_to_session_state(self):

        self.portfolio_manager.sync_to_session_state()