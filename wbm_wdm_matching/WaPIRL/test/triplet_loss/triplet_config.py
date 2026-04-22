import os
import copy
import json
import argparse
import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from configs.task_configs import PretrainConfigBase

class TripletConfig(PretrainConfigBase):
    """Configurations for triplet loss."""
    def __init__(self, args=None, **kwargs):
        super(TripletConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser('Triplet', add_help=False)
        # parser.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
        # parser.add_argument('--projector_size', type=int, default=128, help='Dimension of projector head.')
        
        parser.add_argument('--temperature',  type=float, default=0.07, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative examples.')
        parser.add_argument('--loss_weight', type=float, default=0.5, help='Weighting factor of two loss terms, [0, 1].')
        parser.add_argument('--backbone_wbm', type=str, default='resnet',help='Backbone of wbm encoder.')
        parser.add_argument('--wbm_config', type=str, default='18',help='Configuration of wbm encoder.')
        parser.add_argument('--pretrained_wbm_file', type=str, default=None ,help='Pretrained wbm encoder.')
        parser.add_argument('--backbone_wdm', type=str, default='resnet',help='Backbone of wdm encoder.')
        parser.add_argument('--wdm_config', type=str, default='18',help='Configuration of wdm encoder.')
        parser.add_argument('--pretrained_wdm_file', type=str, default=None ,help='Pretrained wdm encoder.')
        parser.add_argument('--wbm_input_size', type=int, default=40, help='Input size of wbm encoder.')
        parser.add_argument('--wdm_input_size', type=int, default=96, help='Input size of wdm encoder.')
        parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of training data.')
        parser.add_argument('--loss_type', type=str, default='triplet', help='Type of loss.')
        return parser

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,          # 'wm811k'
            self.task,          # 'wapirl'
            f'{self.wbm_encoder_name}_{self.wdm_encoder_name}',    # {'alexnet.bn', 'vgg.16.bn', 'resnet.18', 'resnet.50'}
            self.augmentation,  # {'crop', 'cutout', 'noise', 'rotate', 'shift'}
            self.hash
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def wbm_encoder_name(self):
        return f'{self.backbone_wbm}.{self.wbm_config}'

    @property
    def wdm_encoder_name(self):
        return f'{self.backbone_wdm}.{self.wdm_config}'

    @property
    def task(self):
        return 'triplet'
