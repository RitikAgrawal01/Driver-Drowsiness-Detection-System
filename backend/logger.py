"""
backend/logger.py
──────────────────
Structured logging configuration using structlog.
Outputs JSON in production, coloured console in development.
All modules import get_logger() from here.
"""

import logging
import os
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", environment: str = "development"):
    """
    Configure structlog + stdlib logging.
    Call once at application startup from main.py lifespan.
    """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # Ensure log directory exists
    Path("logs/backend").mkdir(parents=True, exist_ok=True)

    # ── Handlers ──────────────────────────────────────────────────────────────
    handlers = [logging.StreamHandler(sys.stdout)]

    # Rotating file handler
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            "logs/backend/backend.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    except Exception:
        pass  # file logging optional

    logging.basicConfig(
        level=log_level_int,
        format="%(message)s",
        handlers=handlers,
    )

    # ── Structlog processors ──────────────────────────────────────────────────
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment == "production":
        # JSON output for production (machine-parseable)
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Coloured console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a structlog logger bound to the given module name."""
    return structlog.get_logger(name)
