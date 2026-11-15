"""Configuração de logs com saída uniforme e legível."""

import logging
import sys


LOGGER_NAME = "el_nino_monitor"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configura o logger da aplicação sem duplicar mensagens."""
    normalized_level = level.upper()
    if normalized_level not in logging.getLevelNamesMapping():
        raise ValueError(f"Nível de logging inválido: {level}")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(normalized_level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
