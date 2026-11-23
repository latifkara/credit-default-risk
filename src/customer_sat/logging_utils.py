from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
