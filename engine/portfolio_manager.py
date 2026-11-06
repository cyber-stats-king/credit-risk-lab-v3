"""
CreditRisk Lab
Portfolio Management Module

Responsibilities:
1. Store and retrieve the original portfolio.
2. Validate portfolio structure and loan identifiers.
3. Maintain a shared portfolio containing model predictions.
4. Merge PD, EAD and LGD predictions using loan_id.
5. Prevent duplicate or unexpected loan predictions.
6. Save and load portfolio datasets.
7. Integrate with Streamlit session state when available.

Important:
The original portfolio is never modified by prediction updates.
All prediction outputs are maintained separately.
"""

from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PORTFOLIO_DIR = DATA_DIR / "portfolios"

ORIGINAL_PORTFOLIO_PATH = (
    PORTFOLIO_DIR / "original_portfolio.parquet"
)

PREDICTION_PORTFOLIO_PATH = (
    PORTFOLIO_DIR / "prediction_portfolio.parquet"
)

PORTFOLIO_METADATA_PATH = (
    PORTFOLIO_DIR / "portfolio_metadata.json"
)


# ============================================================
# REQUIRED FIELDS AND PREDICTION DEFINITIONS
# ============================================================

PRIMARY_KEY = "loan_id"

PREDICTION_COLUMNS = {
    "PD": ["predicted_pd"],
    "EAD": ["predicted_ccf", "predicted_ead"],
    "LGD": ["predicted_lgd"],
}

ALL_PREDICTION_COLUMNS = [
    column
    for columns in PREDICTION_COLUMNS.values()
    for column in columns
]


# ============================================================
# EXCEPTIONS
# ============================================================

class PortfolioError(Exception):
    """Base exception for portfolio management errors."""


class PortfolioValidationError(PortfolioError):
    """Raised when portfolio data fails validation."""


class PredictionMergeError(PortfolioError):
    """Raised when model predictions cannot be merged safely."""


# ============================================================
# GENERAL HELPERS
# ============================================================

def _utc_timestamp():
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _deep_copy_metadata(metadata):
    """Return an independent copy of portfolio metadata."""

    return copy.deepcopy(metadata)


def _validate_numeric_predictions(
    predictions: pd.DataFrame,
    columns: list
) -> None:
    """
    Validate prediction columns for missing, non-numeric
    and infinite values.
    """

    for column in columns:

        values = pd.to_numeric(
            predictions[column],
            errors="coerce"
        )

        if values.isna().any():

            raise PredictionMergeError(
                f"Prediction column '{column}' contains "
                "missing or non-numeric values."
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():

            raise PredictionMergeError(
                f"Prediction column '{column}' contains "
                "infinite values."
            )


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

def validate_portfolio(
    portfolio: pd.DataFrame,
    require_loan_id: bool = True
) -> None:
    """
    Validate portfolio structure.

    Checks:
    - Input must be a pandas DataFrame.
    - Portfolio must not be empty.
    - loan_id must exist when required.
    - loan_id must not contain missing values.
    - loan_id must be unique.
    """

    if not isinstance(portfolio, pd.DataFrame):

        raise PortfolioValidationError(
            "Portfolio must be a pandas DataFrame."
        )

    if portfolio.empty:

        raise PortfolioValidationError(
            "The portfolio is empty."
        )

    if require_loan_id and PRIMARY_KEY not in portfolio.columns:

        raise PortfolioValidationError(
            f"Required identifier '{PRIMARY_KEY}' is missing."
        )

    if require_loan_id:

        if portfolio[PRIMARY_KEY].isna().any():

            raise PortfolioValidationError(
                "Portfolio contains missing loan_id values."
            )

        if portfolio[PRIMARY_KEY].duplicated().any():

            duplicate_ids = (
                portfolio.loc[
                    portfolio[PRIMARY_KEY].duplicated(
                        keep=False
                    ),
                    PRIMARY_KEY
                ]
                .drop_duplicates()
                .head(10)
                .tolist()
            )

            raise PortfolioValidationError(
                "Duplicate loan_id values detected. "
                f"Examples: {duplicate_ids}"
            )


def validate_prediction_data(
    predictions: pd.DataFrame,
    model_type: str
) -> None:
    """
    Validate predictions before merging them into the portfolio.
    """

    model_type = str(model_type).strip().upper()

    if model_type not in PREDICTION_COLUMNS:

        raise PredictionMergeError(
            f"Unsupported model type: {model_type}. "
            "Expected PD, EAD or LGD."
        )

    if not isinstance(predictions, pd.DataFrame):

        raise PredictionMergeError(
            "Predictions must be a pandas DataFrame."
        )

    if predictions.empty:

        raise PredictionMergeError(
            "The prediction dataset is empty."
        )

    if PRIMARY_KEY not in predictions.columns:

        raise PredictionMergeError(
            f"Predictions must contain '{PRIMARY_KEY}'."
        )

    if predictions[PRIMARY_KEY].isna().any():

        raise PredictionMergeError(
            "Predictions contain missing loan_id values."
        )

    if predictions[PRIMARY_KEY].duplicated().any():

        raise PredictionMergeError(
            "Predictions contain duplicate loan_id values."
        )

    required_columns = PREDICTION_COLUMNS[model_type]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:

        raise PredictionMergeError(
            f"{model_type} predictions are missing "
            f"required columns: {missing_columns}"
        )

    _validate_numeric_predictions(
        predictions,
        required_columns
    )


# ============================================================
# PORTFOLIO MANAGER
# ============================================================

class PortfolioManager:
    """
    Central manager for portfolio data and model predictions.

    The original portfolio is kept separately from the prediction
    portfolio.

    Predictions are merged using loan_id, never the DataFrame index.
    """

    def __init__(self):

        self._original_portfolio = None
        self._prediction_portfolio = None
        self._metadata = {}

    # --------------------------------------------------------
    # LOAD PORTFOLIO
    # --------------------------------------------------------

    def load_portfolio(
        self,
        portfolio: pd.DataFrame,
        source: str = "in_memory"
    ) -> pd.DataFrame:
        """
        Load a portfolio into the manager.

        The supplied DataFrame is copied to prevent accidental
        modification of the original data.
        """

        validate_portfolio(portfolio)

        original = portfolio.copy(deep=True)

        # Remove old prediction columns from the source portfolio.
        original = original.drop(
            columns=[
                column
                for column in ALL_PREDICTION_COLUMNS
                if column in original.columns
            ],
            errors="ignore"
        )

        self._original_portfolio = original.copy(deep=True)

        self._prediction_portfolio = original.copy(deep=True)

        self._metadata = {
            "source": str(source),
            "loaded_at": _utc_timestamp(),
            "record_count": int(len(original)),
            "column_count": int(len(original.columns)),
            "primary_key": PRIMARY_KEY,
            "prediction_models": [],
        }

        return self.get_original_portfolio()

    # --------------------------------------------------------
    # RETRIEVE PORTFOLIO DATA
    # --------------------------------------------------------

    def get_original_portfolio(self) -> pd.DataFrame:
        """Return a copy of the original portfolio."""

        self._ensure_loaded()

        return self._original_portfolio.copy(deep=True)

    def get_prediction_portfolio(self) -> pd.DataFrame:
        """Return a copy of the shared portfolio containing predictions."""

        self._ensure_loaded()

        return self._prediction_portfolio.copy(deep=True)

    def get_portfolio_metadata(self) -> dict:
        """Return an independent copy of portfolio metadata."""

        self._ensure_loaded()

        return _deep_copy_metadata(self._metadata)

    def get_loan_ids(self) -> list:
        """Return all loan identifiers."""

        self._ensure_loaded()

        return self._original_portfolio[PRIMARY_KEY].tolist()

    # --------------------------------------------------------
    # MERGE MODEL PREDICTIONS
    # --------------------------------------------------------

    def merge_predictions(
        self,
        predictions: pd.DataFrame,
        model_type: str,
        allow_partial: bool = False
    ) -> pd.DataFrame:
        """
        Merge model predictions into the shared portfolio.

        Parameters
        ----------
        predictions:
            DataFrame containing loan_id and prediction columns.

        model_type:
            PD, EAD or LGD.

        allow_partial:
            If False, every loan must have a prediction.

            If True, missing predictions are allowed. Existing
            predictions for accounts not included in the new
            scoring batch are retained.
        """

        self._ensure_loaded()

        model_type = str(model_type).strip().upper()

        validate_prediction_data(
            predictions,
            model_type
        )

        predictions_copy = predictions.copy(deep=True)

        portfolio_ids = set(
            self._original_portfolio[PRIMARY_KEY]
        )

        prediction_ids = set(
            predictions_copy[PRIMARY_KEY]
        )

        unexpected_ids = prediction_ids - portfolio_ids

        if unexpected_ids:

            raise PredictionMergeError(
                "Predictions contain loan_id values that do not "
                "exist in the original portfolio. "
                f"Examples: {list(unexpected_ids)[:10]}"
            )

        missing_ids = portfolio_ids - prediction_ids

        if missing_ids and not allow_partial:

            raise PredictionMergeError(
                f"{len(missing_ids)} portfolio accounts have no "
                f"{model_type} predictions. "
                "Set allow_partial=True only when partial scoring "
                "is intentional."
            )

        prediction_columns = PREDICTION_COLUMNS[model_type]

        # Build the new result separately.
        # The current shared portfolio is not modified until
        # the new result has passed validation.

        candidate = self._prediction_portfolio.copy(deep=True)

        existing_columns = [
            column
            for column in prediction_columns
            if column in candidate.columns
        ]

        existing_predictions = None

        if allow_partial and existing_columns:

            existing_predictions = candidate[
                [PRIMARY_KEY] + existing_columns
            ].copy(deep=True)

        candidate = candidate.drop(
            columns=prediction_columns,
            errors="ignore"
        )

        candidate = candidate.merge(
            predictions_copy[
                [PRIMARY_KEY] + prediction_columns
            ],
            on=PRIMARY_KEY,
            how="left",
            validate="one_to_one",
            sort=False
        )

        # Retain existing predictions for accounts that were not
        # included in an intentional partial scoring update.

        if existing_predictions is not None:

            candidate = candidate.merge(
                existing_predictions,
                on=PRIMARY_KEY,
                how="left",
                validate="one_to_one",
                suffixes=("", "_previous"),
                sort=False
            )

            for column in existing_columns:

                previous_column = f"{column}_previous"

                if previous_column in candidate.columns:

                    candidate[column] = candidate[column].combine_first(
                        candidate[previous_column]
                    )

                    candidate = candidate.drop(
                        columns=[previous_column]
                    )

        # Confirm record count and identifier integrity.
        if len(candidate) != len(self._original_portfolio):

            raise PredictionMergeError(
                "Prediction merge changed the portfolio record count."
            )

        validate_portfolio(candidate)

        if set(candidate[PRIMARY_KEY]) != portfolio_ids:

            raise PredictionMergeError(
                "Prediction merge changed the portfolio loan identifiers."
            )

        # Replace the shared portfolio only after all checks pass.
        self._prediction_portfolio = candidate.copy(deep=True)

        if model_type not in self._metadata["prediction_models"]:

            self._metadata["prediction_models"].append(model_type)

        self._metadata["last_prediction_update"] = _utc_timestamp()

        return self.get_prediction_portfolio()

    # --------------------------------------------------------
    # CHECK PREDICTION AVAILABILITY
    # --------------------------------------------------------

    def prediction_status(self) -> pd.DataFrame:
        """
        Return the number of available and missing predictions
        for each model type.
        """

        self._ensure_loaded()

        portfolio = self._prediction_portfolio

        status = []

        for model_type, columns in PREDICTION_COLUMNS.items():

            available = all(
                column in portfolio.columns
                for column in columns
            )

            if available:

                complete = portfolio[columns].notna().all(axis=1)

                predicted_count = int(complete.sum())
                missing_count = int((~complete).sum())

            else:

                predicted_count = 0
                missing_count = len(portfolio)

            status.append({
                "Model": model_type,
                "Portfolio Accounts": len(portfolio),
                "Predicted Accounts": predicted_count,
                "Missing Predictions": missing_count,
                "Completion Rate": (
                    predicted_count / len(portfolio)
                    if len(portfolio) > 0
                    else 0
                ),
            })

        return pd.DataFrame(status)

    # --------------------------------------------------------
    # SAVE PORTFOLIO
    # --------------------------------------------------------

    def save_portfolio(self) -> None:
        """
        Persist original portfolio, prediction portfolio and metadata.

        Temporary files are written first. Existing files are
        replaced only after the temporary files have been written.
        """

        self._ensure_loaded()

        PORTFOLIO_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        original_temp = PORTFOLIO_DIR / "original_portfolio.tmp.parquet"
        prediction_temp = PORTFOLIO_DIR / "prediction_portfolio.tmp.parquet"
        metadata_temp = PORTFOLIO_DIR / "portfolio_metadata.tmp.json"

        try:

            self._original_portfolio.to_parquet(
                original_temp,
                index=False
            )

            self._prediction_portfolio.to_parquet(
                prediction_temp,
                index=False
            )

            metadata_to_save = _deep_copy_metadata(self._metadata)

            metadata_to_save["saved_at"] = _utc_timestamp()

            with open(
                metadata_temp,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metadata_to_save,
                    file,
                    indent=4,
                    allow_nan=False
                )

            original_temp.replace(ORIGINAL_PORTFOLIO_PATH)

            prediction_temp.replace(PREDICTION_PORTFOLIO_PATH)

            metadata_temp.replace(PORTFOLIO_METADATA_PATH)

            self._metadata = metadata_to_save

        except ImportError as error:

            raise PortfolioError(
                "Parquet support is unavailable. Install pyarrow "
                "in the active creditrisklab environment using: "
                "pip install pyarrow"
            ) from error

        except Exception as error:

            raise PortfolioError(
                f"Could not save portfolio data: {error}"
            ) from error

        finally:

            # Remove temporary files left behind after a failure.
            for temporary_file in (
                original_temp,
                prediction_temp,
                metadata_temp
            ):

                if temporary_file.exists():

                    try:
                        temporary_file.unlink()
                    except OSError:
                        pass

    # --------------------------------------------------------
    # LOAD SAVED PORTFOLIO
    # --------------------------------------------------------

    def load_saved_portfolio(self) -> pd.DataFrame:
        """
        Load previously saved portfolio data from disk.
        """

        if not ORIGINAL_PORTFOLIO_PATH.exists():

            raise PortfolioError(
                "No saved original portfolio was found."
            )

        if not PREDICTION_PORTFOLIO_PATH.exists():

            raise PortfolioError(
                "No saved prediction portfolio was found."
            )

        try:

            original = pd.read_parquet(
                ORIGINAL_PORTFOLIO_PATH
            )

            prediction = pd.read_parquet(
                PREDICTION_PORTFOLIO_PATH
            )

        except ImportError as error:

            raise PortfolioError(
                "Parquet support is unavailable. Install pyarrow "
                "in the active creditrisklab environment using: "
                "pip install pyarrow"
            ) from error

        except Exception as error:

            raise PortfolioError(
                f"Could not read saved portfolio data: {error}"
            ) from error

        validate_portfolio(original)
        validate_portfolio(prediction)

        if len(original) != len(prediction):

            raise PortfolioError(
                "Saved original and prediction portfolios "
                "have different record counts."
            )

        original_ids = set(original[PRIMARY_KEY])
        prediction_ids = set(prediction[PRIMARY_KEY])

        if original_ids != prediction_ids:

            raise PortfolioError(
                "Saved original and prediction portfolios "
                "contain different loan identifiers."
            )

        # Ensure the prediction portfolio has not lost any
        # original portfolio columns.
        missing_original_columns = [
            column
            for column in original.columns
            if column not in prediction.columns
        ]

        if missing_original_columns:

            raise PortfolioError(
                "Saved prediction portfolio is missing original "
                f"columns: {missing_original_columns}"
            )

        self._original_portfolio = original.copy(deep=True)

        self._prediction_portfolio = prediction.copy(deep=True)

        if PORTFOLIO_METADATA_PATH.exists():

            try:

                with open(
                    PORTFOLIO_METADATA_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self._metadata = json.load(file)

            except (json.JSONDecodeError, OSError) as error:

                raise PortfolioError(
                    f"Could not read portfolio metadata: {error}"
                ) from error

        else:

            self._metadata = {}

        # Restore essential metadata fields if older metadata
        # does not contain them.
        self._metadata.setdefault(
            "source",
            "saved_portfolio"
        )

        self._metadata.setdefault(
            "record_count",
            int(len(original))
        )

        self._metadata.setdefault(
            "column_count",
            int(len(original.columns))
        )

        self._metadata.setdefault(
            "primary_key",
            PRIMARY_KEY
        )

        self._metadata.setdefault(
            "prediction_models",
            []
        )

        return self.get_prediction_portfolio()

    # --------------------------------------------------------
    # STREAMLIT SESSION STATE INTEGRATION
    # --------------------------------------------------------

    def sync_to_session_state(self) -> None:
        """
        Synchronise managed portfolios with Streamlit session state.

        Streamlit is imported only when this method is called.
        """

        self._ensure_loaded()

        try:

            import streamlit as st

        except ImportError as error:

            raise PortfolioError(
                "Streamlit is not installed in the active environment."
            ) from error

        st.session_state["original_portfolio"] = (
            self.get_original_portfolio()
        )

        st.session_state["prediction_portfolio"] = (
            self.get_prediction_portfolio()
        )

        # Compatibility with existing modelling pages.
        st.session_state["credit_data"] = (
            self.get_original_portfolio()
        )

        st.session_state["portfolio_manager_metadata"] = (
            self.get_portfolio_metadata()
        )

    def load_from_session_state(self) -> pd.DataFrame:
        """
        Load the original portfolio currently stored in Streamlit.
        """

        try:

            import streamlit as st

        except ImportError as error:

            raise PortfolioError(
                "Streamlit is not installed in the active environment."
            ) from error

        if "original_portfolio" in st.session_state:

            portfolio = st.session_state["original_portfolio"]

        elif "credit_data" in st.session_state:

            portfolio = st.session_state["credit_data"]

        else:

            raise PortfolioError(
                "No portfolio was found in Streamlit session state."
            )

        return self.load_portfolio(
            portfolio,
            source="streamlit_session_state"
        )

    # --------------------------------------------------------
    # INTERNAL CHECKS
    # --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Ensure that a portfolio has been loaded."""

        if self._original_portfolio is None:

            raise PortfolioError(
                "No portfolio has been loaded. "
                "Call load_portfolio() first."
            )

        if self._prediction_portfolio is None:

            raise PortfolioError(
                "Prediction portfolio has not been initialised."
            )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_portfolio_manager(
    portfolio: pd.DataFrame,
    source: str = "in_memory"
) -> PortfolioManager:
    """Create and initialise a PortfolioManager."""

    manager = PortfolioManager()

    manager.load_portfolio(
        portfolio,
        source=source
    )

    return manager


def get_prediction_columns() -> list:
    """Return the list of supported prediction columns."""

    return ALL_PREDICTION_COLUMNS.copy()