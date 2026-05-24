import logging


def get_logger(name: str) -> logging.Logger:
    """Return a package logger with a simple default handler."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI entrypoints."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
