try:
    from .generator import generate_chihuahua_horoscope  # noqa: F401
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        'Horoscope dependencies are missing. Install with: pip install -e ".[horoscope]"'
    ) from e

__all__ = ["generate_chihuahua_horoscope"]