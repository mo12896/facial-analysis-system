import logging

import pytest

from src.emotion.utils.logger import setup_logger


# Test the setup_logger function with pytest
def test_setup_logger():
    logger = setup_logger(name="test_logger", level=logging.DEBUG)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2


def test_setup_logger_no_loggers():
    with pytest.raises(ValueError):
        setup_logger(
            name="test_logger",
            level=logging.DEBUG,
            file_logger=False,
            stream_logger=False,
        )
