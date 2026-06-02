"""
SMARTZI - Logging Configuration
Structured JSON logging for production observability.
"""

import logging
import sys
from app.core.config import settings


def setup_logging():
    """Configure application-wide logging."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger("smartzi")
    logger.info("SMARTZI API logging initialized (level=%s)", logging.getLevelName(log_level))
