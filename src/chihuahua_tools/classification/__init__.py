"""
Chihuahua vs Muffin classification module.

Install extras:
    pip install -e ".[torch]"
"""

from .predict import load_model, predict_path

__all__ = ["load_model", "predict_path"]
