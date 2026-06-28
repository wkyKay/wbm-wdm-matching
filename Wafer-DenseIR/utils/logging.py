# -*- coding: utf-8 -*-

import logging
import os


def get_logger(logfile: str):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_epoch_description(history: dict, current: int, total: int, best: int):
    desc = f'Epoch [{current}/{total}] best={best}'
    for metric_name, metric_dict in history.items():
        for split, value in metric_dict.items():
            desc += f' | {split}_{metric_name}: {value:.4f}'
    return desc
