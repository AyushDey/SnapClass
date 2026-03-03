"""Tests for utils.py – setup_logger and intercept_uvicorn_logs."""

import logging
import sys
from unittest.mock import patch, MagicMock
import pytest


def test_setup_logger_creates_logger_with_handlers():
    """setup_logger returns a Logger with console + file handlers."""
    from utils import setup_logger

    with patch("utils.RotatingFileHandler") as mock_rfh:
        mock_rfh.return_value = MagicMock(spec=logging.Handler)
        logger = setup_logger("test.logger.unique1")
        assert logger.name == "test.logger.unique1"
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logger_does_not_add_duplicate_handlers():
    """Calling setup_logger twice does not add extra handlers."""
    from utils import setup_logger

    with patch("utils.RotatingFileHandler") as mock_rfh:
        mock_rfh.return_value = MagicMock(spec=logging.Handler)
        logger1 = setup_logger("test.logger.unique2")
        count_after_first = len(logger1.handlers)
        logger2 = setup_logger("test.logger.unique2")
        assert len(logger2.handlers) == count_after_first


def test_intercept_uvicorn_logs_reconfigures_loggers():
    """intercept_uvicorn_logs clears and replaces handlers on uvicorn loggers."""
    from utils import intercept_uvicorn_logs

    with patch("utils.RotatingFileHandler") as mock_rfh:
        mock_rfh.return_value = MagicMock(spec=logging.Handler)
        intercept_uvicorn_logs()

    for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(name)
        assert not logger.propagate
        # Should have exactly 2 handlers: StreamHandler + RotatingFileHandler
        assert len(logger.handlers) == 2
