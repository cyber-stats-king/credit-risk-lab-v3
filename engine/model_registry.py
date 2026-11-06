"""
CreditRisk Lab
Model Registry

Responsibilities:
1. Register fitted PD, EAD and LGD models.
2. Persist fitted model artifacts using joblib.
3. Maintain model versions.
4. Store model metadata and performance metrics.
5. Retrieve registered models.
6. List available models.
7. Track the currently active model for each model type.

Security:
Only load model artifacts created by this application or from
other trusted sources. Joblib uses Python serialization and
loading untrusted artifacts can execute malicious code.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numbers
import re
import uuid
import copy

import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_TYPES = ("PD", "EAD", "LGD")


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelRegistryError(Exception):
    """Base exception for model registry errors."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model cannot be found."""


class ModelValidationError(ModelRegistryError):
    """Raised when model registration details are invalid."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _utc_timestamp():
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_name(value):
    """Convert a name into a filesystem-safe identifier."""

    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value
    )

    value = value.strip("_")

    if not value:
        raise ModelValidationError(
            "Model name cannot be empty."
        )

    return value


def _json_safe(value):
    """
    Convert common Python and NumPy values into JSON-compatible data.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:
    """
    Central registry for fitted credit risk models.

    Each registered model receives a version number.

    Example:
        PD model:  pd_model_v001
        PD model:  pd_model_v002
        EAD model: ead_model_v001
        LGD model: lgd_model_v001
    """

    def __init__(self, artifact_dir=None):

        self.artifact_dir = (
            Path(artifact_dir)
            if artifact_dir is not None
            else ARTIFACT_DIR
        )

        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for model_type in MODEL_TYPES:

            (
                self.artifact_dir / model_type.lower()
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    # ========================================================
    # PATH MANAGEMENT
    # ========================================================

    def _model_directory(self, model_type):

        model_type = self._validate_model_type(model_type)

        directory = (
            self.artifact_dir / model_type.lower()
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return directory

    def _metadata_path(self, model_type):

        return (
            self._model_directory(model_type)
            / "registry.json"
        )

    def _validate_model_type(self, model_type):

        model_type = str(model_type).strip().upper()

        if model_type not in MODEL_TYPES:
            raise ModelValidationError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {MODEL_TYPES}"
            )

        return model_type

    # ========================================================
    # METADATA MANAGEMENT
    # ========================================================

    def _read_metadata(self, model_type):

        path = self._metadata_path(model_type)

        if not path.exists():
            return {
                "model_type": model_type,
                "models": [],
                "active_model": None,
            }

        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

        except (json.JSONDecodeError, OSError) as error:

            raise ModelRegistryError(
                f"Could not read registry metadata: {error}"
            ) from error

        if not isinstance(metadata, dict):
            raise ModelRegistryError(
                "Registry metadata must be a dictionary."
            )

        metadata.setdefault("model_type", model_type)
        metadata.setdefault("models", [])
        metadata.setdefault("active_model", None)

        if not isinstance(metadata["models"], list):
            raise ModelRegistryError(
                "Registry models must be stored as a list."
            )

        return metadata

    def _write_metadata(self, model_type, metadata):

        path = self._metadata_path(model_type)

        temporary_path = path.with_suffix(".tmp")

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                _json_safe(metadata),
                file,
                indent=4,
                allow_nan=False
            )

        temporary_path.replace(path)

    # ========================================================
    # VERSION MANAGEMENT
    # ========================================================

    def _next_version(self, metadata):

        versions = [
            int(record["version"])
            for record in metadata["models"]
            if "version" in record
        ]

        return max(versions, default=0) + 1

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    def register_model(
        self,
        model_type,
        model,
        model_name=None,
        algorithm=None,
        target=None,
        features=None,
        metrics=None,
        training_date=None,
        training_observations=None,
        metadata=None,
        activate=True,
    ):
        """
        Register and save a fitted model.
        """

        model_type = self._validate_model_type(model_type)

        if model is None:
            raise ModelValidationError(
                "Cannot register an empty model."
            )

        if model_name is None:
            model_name = f"{model_type.lower()}_model"

        model_name = _safe_name(model_name)

        features = [] if features is None else features
        metrics = {} if metrics is None else metrics
        metadata = {} if metadata is None else metadata

        if not isinstance(features, (list, tuple)):
            raise ModelValidationError(
                "Features must be supplied as a list or tuple."
            )

        if not isinstance(metrics, dict):
            raise ModelValidationError(
                "Metrics must be supplied as a dictionary."
            )

        if not isinstance(metadata, dict):
            raise ModelValidationError(
                "Metadata must be supplied as a dictionary."
            )

        # Validate training observation count.
        if training_observations is not None:

            if isinstance(training_observations, bool):
                raise ModelValidationError(
                    "Training observations must be a non-negative integer."
                )

            try:
                observation_value = float(training_observations)

            except (TypeError, ValueError, OverflowError):
                raise ModelValidationError(
                    "Training observations must be a non-negative integer."
                )

            if (
                not math.isfinite(observation_value)
                or observation_value < 0
                or not observation_value.is_integer()
            ):
                raise ModelValidationError(
                    "Training observations must be a non-negative integer."
                )

            training_observations = int(observation_value)

        # Read registry and determine model version.
        registry = self._read_metadata(model_type)

        version = self._next_version(registry)

        model_id = (
            f"{model_name}_v{version:03d}_"
            f"{uuid.uuid4().hex[:8]}"
        )

        model_directory = self._model_directory(model_type)

        model_path = (
            model_directory
            / f"{model_id}.joblib"
        )

        temporary_model_path = (
            model_directory
            / f"{model_id}.tmp"
        )

        # Save the fitted model to a temporary artifact first.
        try:
            joblib.dump(
                model,
                temporary_model_path,
                compress=3
            )

            temporary_model_path.replace(model_path)

        except Exception as error:

            if temporary_model_path.exists():
                temporary_model_path.unlink()

            raise ModelRegistryError(
                f"Could not save model artifact: {error}"
            ) from error

        model_record = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "version": version,
            "algorithm": algorithm,
            "target": target,
            "features": list(features),
            "metrics": _json_safe(metrics),
            "training_date": (
                training_date or _utc_timestamp()
            ),
            "training_observations": training_observations,
            "registered_at": _utc_timestamp(),
            "artifact_path": str(model_path.resolve().relative_to(
                self.artifact_dir.resolve()
            )),
            "status": "REGISTERED",
            "additional_metadata": _json_safe(metadata),
        }

        # Add the new record to the registry.
        registry["models"].append(model_record)

        if activate:
            registry["active_model"] = model_id

        registry["updated_at"] = _utc_timestamp()

        try:
            self._write_metadata(
                model_type,
                registry
            )

        except Exception as error:

            if model_path.exists():
                model_path.unlink()

            raise ModelRegistryError(
                f"Could not update model registry: {error}"
            ) from error

        return copy.deepcopy(model_record)

    # ========================================================
    # RETRIEVE MODEL METADATA
    # ========================================================

    def get_model_metadata(
        self,
        model_type,
        model_id=None
    ):

        model_type = self._validate_model_type(model_type)

        registry = self._read_metadata(model_type)

        if model_id is None:
            model_id = registry.get("active_model")

        if model_id is None:
            raise ModelNotFoundError(
                f"No active {model_type} model is registered."
            )

        for record in registry["models"]:

            if record["model_id"] == model_id:
                return copy.deepcopy(record)

        raise ModelNotFoundError(
            f"Model '{model_id}' was not found."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        model_type,
        model_id=None
    ):

        model_type = self._validate_model_type(model_type)

        record = self.get_model_metadata(
            model_type,
            model_id
        )

        relative_path = Path(record["artifact_path"])

        if relative_path.is_absolute():
            raise ModelRegistryError(
                "Absolute artifact paths are not permitted."
            )

        model_path = (
            self.artifact_dir / relative_path
        ).resolve()

        expected_directory = (
            self._model_directory(model_type).resolve()
        )

        try:
            model_path.relative_to(expected_directory)

        except ValueError as error:

            raise ModelRegistryError(
                "Model artifact path is outside the "
                "registered model directory."
            ) from error

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model artifact does not exist: {model_path}"
            )

        try:
            model = joblib.load(model_path)

        except Exception as error:

            raise ModelRegistryError(
                f"Could not load model '{record['model_id']}': "
                f"{error}"
            ) from error

        return model

    # ========================================================
    # ACTIVATE MODEL
    # ========================================================

    def activate_model(
        self,
        model_type,
        model_id
    ):

        model_type = self._validate_model_type(model_type)

        registry = self._read_metadata(model_type)

        matching_models = [
            record
            for record in registry["models"]
            if record["model_id"] == model_id
        ]

        if not matching_models:
            raise ModelNotFoundError(
                f"Model '{model_id}' was not found."
            )

        registry["active_model"] = model_id
        registry["updated_at"] = _utc_timestamp()

        self._write_metadata(
            model_type,
            registry
        )

        return model_id

    # ========================================================
    # LIST MODELS
    # ========================================================

    def list_models(self, model_type=None):

        if model_type is not None:

            model_type = self._validate_model_type(model_type)

            registry = self._read_metadata(model_type)

            active_model = registry.get("active_model")

            records = []

            for record in registry["models"]:

                item = copy.deepcopy(record)

                item["is_active"] = (
                    item["model_id"] == active_model
                )

                records.append(item)

            return records

        all_models = []

        for current_type in MODEL_TYPES:

            all_models.extend(
                self.list_models(current_type)
            )

        return all_models

    # ========================================================
    # GET ACTIVE MODEL
    # ========================================================

    def get_active_model_id(self, model_type):

        model_type = self._validate_model_type(model_type)

        registry = self._read_metadata(model_type)

        return registry.get("active_model")

    # ========================================================
    # MODEL REGISTRY SUMMARY
    # ========================================================

    def registry_summary(self):

        summary = []

        for model_type in MODEL_TYPES:

            registry = self._read_metadata(model_type)

            models = registry["models"]

            summary.append({
                "Model Type": model_type,
                "Registered Versions": len(models),
                "Active Model": registry.get("active_model"),
            })

        return summary

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def compare_models(self, model_type):

        records = self.list_models(model_type)

        comparison = []

        for record in records:

            comparison.append({
                "Model ID": record["model_id"],
                "Algorithm": record["algorithm"],
                "Version": record["version"],
                "Target": record["target"],
                "Training Observations": (
                    record["training_observations"]
                ),
                "Metrics": record["metrics"],
                "Active": record["is_active"],
                "Registered At": record["registered_at"],
            })

        return comparison