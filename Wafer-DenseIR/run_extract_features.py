# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch

from configs.task_configs import DenseRetrievalConfig
from datasets.wm38k import WM38K
from models.proposed_encoder import build_dense_encoder, load_encoder_checkpoint
from tasks.dense_retrieval import DenseRetrieval


def main():
    config = DenseRetrievalConfig.parse_arguments()
    config.save()
    device = _get_device(config.device)
    data_file = _resolve_path(config.data_file)

    dataset = WM38K(
        npz_file=data_file,
        input_size=config.input_size,
        split=config.split,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        max_samples=config.max_samples,
        split_manifest=_resolve_optional_path(config.split_manifest),
        query_manifest=_resolve_optional_path(config.query_manifest),
    )
    backbone = build_dense_encoder(config.encoder, config.embedding_dim, config.encoder_width)
    if config.pretrained_model_file is not None:
        load_encoder_checkpoint(backbone, config.pretrained_model_file, key=config.pretrained_model_key)

    task = DenseRetrieval(backbone=backbone, device=device, output_dir=config.output_dir)
    records = task.extract_features(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        token_mode=config.token_mode,
        token_dilation=config.token_dilation,
        max_tokens=config.max_tokens,
    )
    task.save_features(records)
    print(f'Feature file: {os.path.join(config.output_dir, "dense_features.npz")}')


def _get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        torch.cuda.set_device(device)
        print(f'[Device] GPU: {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})')
        torch.backends.cudnn.benchmark = True
    else:
        print('[Device] GPU not available, falling back to CPU')
    return device


def _resolve_path(path):
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, path)),
        os.path.abspath(os.path.join(here, '..', '..', path)),
        os.path.abspath(os.path.join(here, '..', '..', '..', path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_optional_path(path):
    return None if path is None else _resolve_path(path)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
