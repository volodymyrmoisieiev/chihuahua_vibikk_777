"""
Chihuahua vs Muffin classification module.

Install extras:
    pip install -e ".[torch]"
"""

from .predict import (
    load_model,
    predict_image,
    predict_path,
    predict_path_with_plot,
    predict_many,
    ClassificationResult,
    CLASS_NAMES,
)

__all__ = [
    "load_model",
    "predict_image",
    "predict_path",
    "predict_path_with_plot",
    "predict_many",
    "ClassificationResult",
    "CLASS_NAMES",
]
