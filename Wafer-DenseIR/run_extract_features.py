# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch

from configs.network_configs import RESNET_BACKBONE_CONFIGS
from configs.task_configs import DenseRetrievalConfig
from datasets.wm38k import WM38K
from models.resnet import ResNetBackbone
from tasks.dense_retrieval import DenseRetrieval


def main():
    config = DenseRetrievalConfig.parse_arguments()
    config.save()
    device = torch.device('cuda' if config.device == 'auto' and torch.cuda.is_available() else 'cpu')
    if config.device != 'auto':
        device = torch.device(config.device)
    data_file = _resolve_path(config.data_file)

    dataset = WM38K(
        npz_file=data_file,
        input_size=config.input_size,
        split=config.split,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        max_samples=config.max_samples,
        decouple_input=config.decouple_input,
    )
    backbone = ResNetBackbone(RESNET_BACKBONE_CONFIGS[config.backbone_config], in_channels=2)
    if config.pretrained_model_file is not None:
        backbone.load_weights_from_checkpoint(config.pretrained_model_file, key=config.pretrained_model_key)

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


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
