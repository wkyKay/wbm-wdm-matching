# -*- coding: utf-8 -*-
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.task_configs import WaPIRLConfig
from datasets.wm811k import WM811KForWaPIRL
# from datasets.transforms import WM811KTransform
from wbm_transform import WBM10x10Transform
from models.head import LinearHead, MLPHead
from tasks.wapirl import WaPIRL, MemoryBank
from utils.loss import WaPIRLLoss
from utils.metrics import TopKAccuracy
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler
from model import WBM_Encoder

def main():
    local_rank = 0
    config = WaPIRLConfig.parse_arguments()
    setattr(config, 'distributed', False)
    config.save()
    
    in_channels = int(config.decouple_input) + 1
    encoder = WBM_Encoder(in_channels=in_channels, output_dim=512)
    head = LinearHead(in_channels=512, num_features=128)

    params = [{'params': encoder.parameters()}, {'params': head.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps
    )

    # Data
    data_kwargs = {
        # 'transform': WM811KTransform(size=config.input_size, mode='test'),
        'transform': WBM10x10Transform(mode='test'),
        'positive_transform': WBM10x10Transform(mode='flip'),
        # 'positive_transform': WM811KTransform(size=config.input_size, mode=config.augmentation),
        'decouple_input': config.decouple_input,
    }

    train_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/wbm(10*10)/unlabeled/train/-/', **data_kwargs),
        # WM811KForWaPIRL('./data/wm811k/wbm(10*10)/labeled/train/', **data_kwargs),
    ])
    valid_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/wbm(10*10)/unlabeled/valid/-/', **data_kwargs),
        # WM811KForWaPIRL('./data/wm811k/wbm(10*10)/labeled/valid/', **data_kwargs),
    ])
    test_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/wbm(10*10)/unlabeled/test/-/', **data_kwargs),
        # WM811KForWaPIRL('./data/wm811k/wbm(10*10)/labeled/test/', **data_kwargs),
    ])
    print(f'训练集大小：{len(train_set)}')
    print(f'验证集大小：{len(valid_set)}')
    
    experiment_kwargs = {
        'backbone': encoder,
        'projector': head,
        'memory': MemoryBank(
            size=(len(train_set), config.projector_size),
            device=local_rank
            ),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': WaPIRLLoss(temperature=config.temperature),
        'loss_weight': config.loss_weight,
        'num_negatives': config.num_negatives,
        'distributed': config.distributed,
        'local_rank': local_rank,
        'metrics': {
            'top@1': TopKAccuracy(num_classes=1 + config.num_negatives, k=1),
            'top@5': TopKAccuracy(num_classes=1 + config.num_negatives, k=5)
            },
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = WaPIRL(**experiment_kwargs)


    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)
    logger.info(f"Data: {config.data}")
    logger.info(f"Augmentation: {config.augmentation}")
    logger.info(f"Observations: {len(train_set):,}")
    logger.info(f"Trainable parameters ({encoder.__class__.__name__}): {encoder.num_parameters:,}")
    logger.info(f"Trainable parameters ({head.__class__.__name__}): {head.num_parameters:,}")
    logger.info(f"Projection head: {config.projector_type} ({config.projector_size})")
    logger.info(f"Checkpoint directory: {config.checkpoint_dir}")

        # Train (WaPIRL)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'logger': logger,
        'save_every': config.save_every,
    }
    experiment.run(**run_kwargs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
