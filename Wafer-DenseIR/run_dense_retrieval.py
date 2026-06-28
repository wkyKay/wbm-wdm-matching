# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch

from configs.network_configs import RESNET_BACKBONE_CONFIGS, VIT_BACKBONE_CONFIGS
from configs.task_configs import DenseRetrievalConfig
from datasets.wm38k import WM38K
from models.resnet import ResNetBackbone
from models.vit import ViTTinyBackbone
from tasks.dense_retrieval import DenseRetrieval


def main():
    config = DenseRetrievalConfig.parse_arguments()
    _normalize_backbone_config(config)
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
        decouple_input=config.decouple_input,
    )

    in_channels = 2 if config.decouple_input else 1
    backbone = _build_backbone(config, in_channels)
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
    if config.save_features:
        task.save_features(records)
    rankings, scores, metrics, match_cache = task.run_retrieval(
        records,
        topk_tokens=config.topk_tokens,
        sigma_pos=config.sigma_pos,
        ks=tuple(config.topk_retrieval),
    )
    task.save_explanations(records, rankings, scores, match_cache, num_queries=config.explain_top_queries)

    print(f'Output directory: {config.output_dir}')
    print(metrics)


def _get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _build_backbone(config, in_channels):
    if config.backbone_type == 'resnet':
        return ResNetBackbone(RESNET_BACKBONE_CONFIGS[config.backbone_config], in_channels=in_channels)
    if config.backbone_type == 'vit':
        return ViTTinyBackbone(
            VIT_BACKBONE_CONFIGS[config.backbone_config],
            in_channels=in_channels,
            img_size=config.input_size,
        )
    raise ValueError(f'Unknown backbone_type: {config.backbone_type}')


def _normalize_backbone_config(config):
    if config.backbone_type == 'vit' and config.backbone_config == '18':
        config.backbone_config = 'tiny'


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
