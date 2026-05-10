# -*- coding: utf-8 -*-

import os
import logging
import tqdm


def get_tqdm_config(total: int, leave: bool = True, color: str = 'white') -> dict:
    return dict(
        total=total,
        leave=leave,
        dynamic_ncols=True,
        colour=color,
    )


def make_epoch_description(history: dict, current: int, total: int, best: int) -> str:
    desc = f" Epoch [{current:>4d}/{total:>4d}] (best: {best:>4d}): "
    for metric_name, metric_dict in history.items():
        if not isinstance(metric_dict, dict):
            continue
        for split, val in metric_dict.items():
            if val is not None:
                desc += f" {split}_{metric_name}: {val:.4f} |"
    return desc


def get_logger(name: str, log_dir: str, filename: str = 'train.log') -> logging.Logger:
    """
    创建同时输出到文件和控制台的 logger。
    日志文件保存在 log_dir/filename，追加模式（不覆盖已有日志）。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（多次调用时）
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    # 文件 handler
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
