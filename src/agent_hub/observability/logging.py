"""Structured JSON logging, matching the Pino-shaped output of the TS hubs.

One line per event, machine-parseable, with a `correlation_id` slot so agent
runs join against Kong's X-Correlation-Id and (from M4) the Langfuse trace.
Uniformity across languages is a feature (TDD §5.3 principal's note).
"""

import logging
import sys

import structlog


def configure_logging(level: str = "info") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    # Quiet noisy third-party loggers even at LOG_LEVEL=debug — aiokafka in
    # particular dumps full ApiVersion/Metadata responses per poll, which buries
    # our structured logs. The app's own loggers stay at the configured level.
    for noisy in ("aiokafka", "kafka", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "agent-hub") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
