"""
Emoji/webcam demo module (OpenCV + MediaPipe).

Install extra dependencies:
    pip install -e ".[cv]"
"""

try:
    from .main import main  # noqa: F401
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        'Emoji module dependencies are missing. Install with: pip install -e ".[cv]"'
    ) from e

__all__ = ["main"]
