from __future__ import annotations

import argparse

from .generator import generate_chihuahua_horoscope


def main() -> None:
    """CLI entrypoint for manual testing."""
    parser = argparse.ArgumentParser(description="Chihuahua horoscope generator")
    parser.add_argument("name", type=str)
    parser.add_argument("--details", type=str, default=None)
    args = parser.parse_args()

    print(generate_chihuahua_horoscope(args.name, details=args.details))


if __name__ == "__main__":
    main()
