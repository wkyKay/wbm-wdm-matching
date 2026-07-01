# -*- coding: utf-8 -*-

import os
import random
import sys

import numpy as np
import torch

from configs.network_configs import RESNET_BACKBONE_CONFIGS, VIT_BACKBONE_CONFIGS
from configs.task_configs import WaPIRLPretrainConfig
from datasets.wapirl_wm38k import WM38KForWaPIRL
from models.head import LinearHead, MLPHead
from models.resnet import ResNetBackbone
from models.vit import ViTTinyBackbone
from tasks.wapirl_pretrain import MemoryBank, WaPIRLPretrain
from utils.logging import get_logger
from utils.loss import WaPIRLLoss
from utils.optimization import get_optimizer, get_scheduler


PROJECTOR_TYPES = {
    'linear': LinearHead,
    'mlp': MLPHead,
}


def main():
    config = WaPIRLPretrainConfig.parse_arguments()
    _normalize_backbone_config(config)
    _seed_everything(config.seed)
    config.save()

    device = _get_device(config.device)
    _configure_cuda(device)
    data_file = _resolve_path(config.data_file)
    in_channels = 2 if config.decouple_input else 1

    train_set = _build_dataset(config, data_file, split='train')
    valid_set = _build_dataset(config, data_file, split='valid')
    if len(valid_set) < 2:
        valid_set = _build_dataset(config, data_file, split='test')
    if len(valid_set) < 2 and config.quality_filter:
        setattr(config, 'quality_filter', False)
        valid_set = _build_dataset(config, data_file, split='valid')
        setattr(config, 'quality_filter', True)

    backbone = _build_backbone(config, in_channels)
    projector = PROJECTOR_TYPES[config.projector_type](backbone.out_channels, config.projector_size)
    if config.pretrained_model_file is not None:
        backbone.load_weights_from_checkpoint(config.pretrained_model_file, key=config.pretrained_model_key)

    params = [{'params': backbone.parameters()}, {'params': projector.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
    )

    memory = MemoryBank(
        size=(len(train_set), config.projector_size),
        device=device,
        weight=config.memory_momentum,
    )
    task = WaPIRLPretrain(
        backbone=backbone,
        projector=projector,
        memory=memory,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=WaPIRLLoss(temperature=config.temperature),
        device=device,
        output_dir=config.output_dir,
        loss_weight=config.loss_weight,
        num_negatives=config.num_negatives,
        write_summary=config.write_summary,
    )

    logger = get_logger(os.path.join(config.output_dir, 'main.log'))
    logger.info(f'Data file: {data_file}')
    logger.info(f'Train samples after filtering: {len(train_set)}')
    logger.info(f'Valid samples after filtering: {len(valid_set)}')
    logger.info(f'Backbone: {config.model_name}')
    logger.info(f'Projector: {config.projector_type}.{config.projector_size}')
    logger.info(f'Device: {device}')
    logger.info(f'Output directory: {config.output_dir}')
    print(f'[Device] {device}')
    if device.type == 'cuda':
        print(f'[Device] GPU: {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})')

    task.run(
        train_set=train_set,
        valid_set=valid_set,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        logger=logger,
        save_every=config.save_every,
    )
    print(f'Checkpoint directory: {config.output_dir}')
    print(f'Best checkpoint: {os.path.join(config.output_dir, "best_model.pt")}')


def _build_dataset(config, data_file, split):
    return WM38KForWaPIRL(
        npz_file=data_file,
        input_size=config.input_size,
        split=split,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        max_samples=config.max_samples if split == 'train' else None,
        decouple_input=config.decouple_input,
        augmentation=config.augmentation,
        crop_min_scale=config.crop_min_scale,
        noise_prob=config.noise_prob,
        rotate_prob=config.rotate_prob,
        quality_filter=config.quality_filter,
        min_defect_pixels=config.min_defect_pixels,
        min_defect_ratio=config.min_defect_ratio,
        max_defect_ratio=config.max_defect_ratio,
        min_valid_ratio=config.min_valid_ratio,
        max_valid_ratio=config.max_valid_ratio,
        deduplicate=config.deduplicate,
        split_manifest=_resolve_optional_path(config.split_manifest),
        query_manifest=_resolve_optional_path(config.query_manifest),
    )


def _get_device(name):
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        raise RuntimeError('CUDA is not available. Use --device cpu only for local debugging.')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, but --device cuda was requested.')
    return device


def _configure_cuda(device):
    if device.type != 'cuda':
        return
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(device.index if device.index is not None else 0)


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


def _resolve_optional_path(path):
    return None if path is None else _resolve_path(path)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
