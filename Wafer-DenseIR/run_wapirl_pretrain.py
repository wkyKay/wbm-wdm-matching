# -*- coding: utf-8 -*-

import os
import random
import sys

import numpy as np
import torch

from configs.task_configs import WaPIRLPretrainConfig
from datasets.wapirl_wm38k import WM38KForWaPIRL
from models.proposed_encoder import build_dense_encoder
from utils.logging import get_logger

from proposed.models.head import MLPHead
from proposed.tasks.cluster_pretrain import ClusterPretrainTask, MemoryBank
from proposed.utils.loss import ClusterNCELoss
from proposed.utils.optimization import get_optimizer, get_scheduler


def main():
    config = WaPIRLPretrainConfig.parse_arguments()
    _seed_everything(config.seed)
    config.save()

    device = _get_device(config.device)
    _configure_cuda(device)
    data_file = _resolve_path(config.data_file)

    train_set = _build_dataset(config, data_file, split='train')
    valid_set = _build_dataset(config, data_file, split='valid')
    if len(valid_set) < 2:
        valid_set = _build_dataset(config, data_file, split='test')
    if len(valid_set) < 2 and config.quality_filter:
        setattr(config, 'quality_filter', False)
        valid_set = _build_dataset(config, data_file, split='valid')
        setattr(config, 'quality_filter', True)

    backbone = build_dense_encoder(config.encoder, config.embedding_dim, config.encoder_width)
    projector = MLPHead(config.embedding_dim, config.projector_size)
    if config.pretrained_model_file is not None:
        from models.proposed_encoder import load_encoder_checkpoint
        load_encoder_checkpoint(backbone, config.pretrained_model_file, key=config.pretrained_model_key)

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
    task = ClusterPretrainTask(
        encoder=backbone,
        projector=projector,
        memory=memory,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=ClusterNCELoss(temperature=config.temperature),
        device=device,
        output_dir=config.output_dir,
        loss_weight=config.loss_weight,
        num_negatives=config.num_negatives,
    )

    logger = get_logger(os.path.join(config.output_dir, 'main.log'))
    logger.info(f'Data file: {data_file}')
    logger.info(f'Train samples after filtering: {len(train_set)}')
    logger.info(f'Valid samples after filtering: {len(valid_set)}')
    logger.info(f'Encoder: {config.model_name}.{config.embedding_dim}')
    logger.info(f'Projector: mlp.{config.projector_size}')
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
