import functools
import logging
from typing import Any, Callable

from .constants import LOG_DIR


def setup_logger(
    name: str,
    level: int = logging.INFO,
    file_logger: bool = False,
    stream_logger: bool = True,
) -> logging.Logger:

    if not (file_logger or stream_logger):
        raise ValueError("At least one logger must be enabled!")

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    if file_logger:
        logger = create_file_logger(logger, name, level)

    if stream_logger:
        logger = create_stream_logger(logger, level)

    return logger


def create_file_logger(
    logger: logging.Logger, name: str, level: int = logging.INFO
) -> logging.Logger:
    # create a file handler
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log")
    file_handler.setLevel(level)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    return logger


def create_stream_logger(
    logger: logging.Logger, level: int = logging.INFO
) -> logging.Logger:
    # create a stream handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # create a logging format
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(console_handler)
    return logger


def with_logging(logger: logging.Logger):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(f"Calling {func.__name__}")
            value = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}")
            return value

        return wrapper

    return decorator
