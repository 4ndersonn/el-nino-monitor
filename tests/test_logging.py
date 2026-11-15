"""Testes da configuração de logging."""

import logging

import pytest

from el_nino_monitor.utils.logging import configure_logging


def test_configure_logging_does_not_duplicate_handlers() -> None:
    logger = configure_logging("INFO")
    original_count = len(logger.handlers)
    assert configure_logging("DEBUG") is logger
    assert len(logger.handlers) == original_count
    assert logger.level == logging.DEBUG


def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Nível de logging inválido"):
        configure_logging("INVALID")
