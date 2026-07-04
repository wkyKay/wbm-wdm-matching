# -*- coding: utf-8 -*-
"""Small logging helpers."""

import logging
from pathlib import Path


def get_logger(path):
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

