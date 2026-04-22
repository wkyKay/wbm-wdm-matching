from pdb import run
import sched
import select
from tokenize import triple_quoted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from triplet_loss.CrossModal import CrossModalRetrievalModel, CrossModalTrainer
from triplet_loss.TripletLoss import TripletLoss, OnlineTripletLoss, InfoNCELoss, MultiSimilarityLoss
from triplet_loss.triplet_config import TripletConfig
from datasets.transforms import WM811KTransform
from configs.network_configs import ALEXNET_BACKBONE_CONFIGS
from configs.network_configs import VGGNET_BACKBONE_CONFIGS
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from triplet_loss.triplet_wm811k import WM811KForTriplet
from models.head import LinearHead, MLPHead
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler
from utils.metrics import TopKAccuracy
from utils.loss import WaPIRLLoss
from triplet_loss.my_metrics import TopKRecall, TopKF1Score

AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': LinearHead,
    'mlp': MLPHead,
}

LOSS_TYPES = {
    'triplet': TripletLoss,
    'online_tripplet': OnlineTripletLoss,
    'info_nce': InfoNCELoss,
    'multi_similarity': MultiSimilarityLoss,
    'wapirl': WaPIRLLoss
}

def train_cross_modal():
    """
    跨模态训练示例
    """
    config = TripletConfig.parse_arguments()
    in_channels = int(config.decouple_input) + 1

    BACKBONE_CONFIGS_WBM, Backbone_WBM = AVAILABLE_MODELS[config.backbone_wbm]
    encoder_wbm = Backbone_WBM(BACKBONE_CONFIGS_WBM[config.wbm_config], in_channels=in_channels)

    BACKBONE_CONFIGS_WDM, Backbone_WDM = AVAILABLE_MODELS[config.backbone_wdm]
    encoder_wdm = Backbone_WDM(BACKBONE_CONFIGS_WDM[config.wdm_config], in_channels=in_channels)

    config.save()

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    logger = get_logger(stream=False, logfile=logfile)
    logger.info(f"Backbone_wbm: {config.backbone_wbm}")
    logger.info(f"Backbone_wdm: {config.backbone_wdm}")
    logger.info(f"Wbm_config: {config.wbm_config}")
    logger.info(f"Wdm_config: {config.wdm_config}")
    

    if config.pretrained_wbm_file is not None:
        try:
            encoder_wbm.load_weights_from_checkpoint(path=config.pretrained_wbm_file, key='backbone')
        except KeyError:
            encoder_wbm.load_weights_from_checkpoint(path=config.pretrained_wbm_file, key='encoder')
        finally:
            if logger is not None:
                logger.info(f"Loaded pre-trained model from: {config.pretrained_wbm_file}")

    if config.pretrained_wdm_file is not None:
        try:
            encoder_wdm.load_weights_from_checkpoint(path=config.pretrained_wdm_file, key='backbone')
        except KeyError:
            encoder_wdm.load_weights_from_checkpoint(path=config.pretrained_wdm_file, key='encoder')
        finally:
            if logger is not None:
                logger.info(f"Loaded pre-trained model from: {config.pretrained_wdm_file}")

    # Data
    data_kwargs = {
        'wbm_transform': WM811KTransform(size=config.wbm_input_size, mode='rotate'),
        'wdm_transform': WM811KTransform(size=config.wdm_input_size, mode='noise'),
        'decouple_input': config.decouple_input,
    }

    train_set = torch.utils.data.ConcatDataset([
        WM811KForTriplet(
            wbm_root ='./data/wm811k/paired_data(20&96)/train/wbm/', 
            wdm_root = './data/wm811k/paired_data(20&96)/train/wdm/', 
            proportion=config.proportion,
            **data_kwargs),
    ])
    valid_set = torch.utils.data.ConcatDataset([
        WM811KForTriplet(
            wbm_root ='./data/wm811k/paired_data(20&96)/val/wbm', 
            wdm_root="./data/wm811k/paired_data(20&96)/val/wdm/", 
            proportion=config.proportion,
            **data_kwargs),
    ])
    test_set = torch.utils.data.ConcatDataset([
        WM811KForTriplet(
            wbm_root='./data/wm811k/paired_data(20&96)/test/wbm/',
            wdm_root='./data/wm811k/paired_data(20&96)/test/wdm/',
            proportion=config.proportion,
            **data_kwargs),
    ])
    print(f'train_set size: {len(train_set)}')
    print(f'valid_set size: {len(valid_set)}')
    print(f'test_set size: {len(test_set)}')
    
   # 使用示例
    # inspect_dataloader(train_loader, num_samples=10)
            
    # 创建模型
    model = CrossModalRetrievalModel(
        encoder_wbm=encoder_wbm,
        encoder_wdm=encoder_wdm,
        embedding_dim=128,
        freeze_encoder=False,  # 冻结encoder，只训练投影头
    )
    
    # 损失函数
    triplet_loss = LOSS_TYPES[config.loss_type]()
    
    # 优化器（只优化投影头）
    optimizer = torch.optim.Adam(
        list(model.proj_wbm.parameters()) +
        list(model.proj_wdm.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )
    
    experiment_kwargs = {
        'model': model,
        'triplet_loss': triplet_loss,
        'optimizer': optimizer,
        'scheduler':scheduler,
        'logger':logger,
        'metrics':{
            'topaccuracy@1': TopKAccuracy(num_classes=1 + config.num_negatives, k=1),
            'topaccuracy@3': TopKAccuracy(num_classes=1 + config.num_negatives, k=3),
            # 'toprecall@1': TopKRecall(num_classes=1 + config.num_negatives, k=1),
            # 'toprecall@3': TopKRecall(num_classes=1 + config.num_negatives, k=3),
            'topf1@1': TopKF1Score(num_classes=1 + config.num_negatives, k=1),
            'topf1@3': TopKF1Score(num_classes=1 + config.num_negatives, k=3),
        },
        'num_negatives': config.num_negatives,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'local_rank': 0
    }
    # 创建训练器
    trainer = CrossModalTrainer(**experiment_kwargs)
    # 训练
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'logger': logger,
        'save_every': config.save_every,
        'k_list': [1, 3, 5],
    }
    trainer.run(
        **run_kwargs
    )
    
    # 检索评估
    test_history = trainer.evaluate(test_set, batch_size=config.batch_size, num_workers=config.num_workers, save_images=True)
    logger.info(test_history)
    


# ============================================================
# 6. 预测/检索
# ============================================================

def retrieval_example(model, query_wbm, gallery_wdm):
    """
    使用训练好的模型进行检索
    
    Args:
        model: 训练好的 CrossModalRetrievalModel
        query_wbm: 查询的WBM batch
        gallery_wdm: WDM图库
    """
    # 检索
    similarities, indices = model.retrieval(query_wbm, gallery_wdm)
    
    print("查询结果：")
    print(f"最相似的WDM索引: {indices[:, 0]}")
    print(f"相似度: {similarities[:, 0]}")
    
    return indices, similarities




if __name__ == '__main__':
    train_cross_modal()

