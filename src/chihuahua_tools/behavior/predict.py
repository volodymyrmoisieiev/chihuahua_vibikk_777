from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping, Sequence, Optional, Protocol, cast

import joblib
import pandas as pd


# Fallback feature list (used if the feature list file is missing).
FEATURE_COLUMNS: list[str] = [
    "Не гавкає 24/7",
    "Не гризе взуття",
    "Шарить в таро та астрології",
    "Цінує СМП, ML та DL",
    "Кайфує від Taylor Swift",
    "Їсть суші",
    "Шанує IQOS культуру",
]


class SklearnLikeClassifier(Protocol):
    """Minimal sklearn-like classifier interface used by this module."""

    def predict(self, X: pd.DataFrame) -> Sequence[int]: ...
    def predict_proba(self, X: pd.DataFrame) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class ChiVibeResult:
    """Prediction output for the Chihuahua vibe model."""

    features: dict[str, int]
    prediction: int
    probability: float
    text: str


def _assets_models_dir():
    """Return the package path to the bundled `assets/models` directory."""
    return files("chihuahua_tools").joinpath("assets/models")


@lru_cache(maxsize=1)
def load_vibe_model() -> tuple[SklearnLikeClassifier, list[str]]:
    """
    Load the pre-trained vibe model and the ordered feature list from package assets.

    Expected files:
    - assets/models/chihuahua_vibe.pkl
    - assets/models/chihuahua_vibe_features.pkl (optional; falls back to FEATURE_COLUMNS)
    """
    base = _assets_models_dir()
    model_path = base / "chihuahua_vibe.pkl"
    features_path = base / "chihuahua_vibe_features.pkl"

    if not model_path.is_file():
        raise FileNotFoundError("Model file not found: assets/models/chihuahua_vibe.pkl")

    model = cast(SklearnLikeClassifier, joblib.load(str(model_path)))

    if features_path.is_file():
        feature_cols = joblib.load(str(features_path))
        feature_cols = list(feature_cols)
    else:
        feature_cols = FEATURE_COLUMNS

    return model, feature_cols


def predict_chi_vibe(
    features_dict: Mapping[str, Any],
    model: Optional[SklearnLikeClassifier] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> ChiVibeResult:
    """
    Predict the "chihuahua vibe" label and probability for the given feature flags.

    Args:
        features_dict: Mapping of feature name -> 0/1 (missing features default to 0).
        model: Optional pre-loaded sklearn-like classifier.
        feature_cols: Optional ordered feature list used to build the input row.

    Returns:
        ChiVibeResult with:
          - `prediction` in {0, 1}
          - `probability` = P(class=1)
          - `text` = human-friendly message
    """
    if model is None or feature_cols is None:
        model, loaded_cols = load_vibe_model()
        feature_cols = loaded_cols

    cols_list = list(feature_cols)

    row: dict[str, int] = {}
    for col in cols_list:
        row[col] = int(features_dict.get(col, 0))

    df_row = pd.DataFrame([row], columns=cols_list)

    if all(v == 0 for v in row.values()):
        raise ValueError("features_dict порожній: передай хоча б одну ознаку = 1.")

    pred = int(model.predict(df_row)[0])
    prob = float(model.predict_proba(df_row)[0][1])

    text = "😊 має позитивний вайбік" if pred == 1 else "😈 шось не те з вайбом"

    return ChiVibeResult(
        features=row,
        prediction=pred,
        probability=prob,
        text=text,
    )
