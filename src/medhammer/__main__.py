"""Entry point for ``python -m medhammer``."""

from __future__ import annotations

from medhammer.cli import app


def main() -> None:
    """Run the Typer CLI application.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
