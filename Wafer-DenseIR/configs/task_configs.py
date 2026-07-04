# -*- coding: utf-8 -*-

import argparse
import copy
import datetime
import json
import os


class ConfigBase(object):
    def __init__(self, args=None, **kwargs):
        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    @classmethod
    def parse_arguments(cls):
        parents = [
            cls.data_parser(),
            cls.model_parser(),
            cls.retrieval_parser(),
            cls.logging_parser(),
        ]
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args
        config = cls()
        parser.parse_args(namespace=config)
        return config

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            return cls(args=json.load(f))

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(self.output_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['model_name'] = self.model_name
        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'

    @property
    def output_dir(self):
        out = os.path.join(
            self.output_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(out, exist_ok=True)
        return out

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if arg.strip():
                yield arg

    @staticmethod
    def data_parser():
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data', type=str, default='wm38k')
        parser.add_argument('--data_file', type=str, default='data/wm38k/Wafer_Map_Datasets.npz')
        parser.add_argument('--input_size', type=int, default=96)
        parser.add_argument('--split', type=str, default='test', choices=('train', 'valid', 'test', 'all'))
        parser.add_argument('--train_ratio', type=float, default=0.7)
        parser.add_argument('--valid_ratio', type=float, default=0.1)
        parser.add_argument('--seed', type=int, default=1993)
        parser.add_argument('--max_samples', type=int, default=None)
        parser.add_argument('--split_manifest', type=str, default=None)
        parser.add_argument('--query_manifest', type=str, default=None)
        parser.add_argument('--candidate_manifest', type=str, default=None)
        return parser

    @staticmethod
    def model_parser():
        parser = argparse.ArgumentParser("Backbone", add_help=False)
        parser.add_argument('--backbone_type', type=str, default='resnet', choices=('resnet', 'vit'))
        parser.add_argument('--backbone_config', type=str, default='18')
        parser.add_argument('--decouple_input', dest='decouple_input', action='store_true')
        parser.add_argument('--no_decouple_input', dest='decouple_input', action='store_false')
        parser.set_defaults(decouple_input=True)
        parser.add_argument('--pretrained_model_file', type=str, default=None)
        parser.add_argument('--pretrained_model_key', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        return parser

    @staticmethod
    def retrieval_parser():
        parser = argparse.ArgumentParser("Dense retrieval", add_help=False)
        parser.add_argument('--features_file', type=str, default=None)
        parser.add_argument('--token_mode', type=str, default='defect_band',
                            choices=('defect_band', 'defect', 'valid', 'all'))
        parser.add_argument('--token_dilation', type=int, default=1)
        parser.add_argument('--topk_tokens', type=int, default=5)
        parser.add_argument('--sigma_pos', type=float, default=0.35)
        parser.add_argument('--max_tokens', type=int, default=256)
        parser.add_argument('--topk_retrieval', type=int, nargs='+', default=[1, 5, 10])
        parser.add_argument('--explain_top_queries', type=int, default=8)
        parser.add_argument('--explain_top_matches', type=int, default=20)
        return parser

    @staticmethod
    def logging_parser():
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--output_root', type=str, default='artifacts/dense_retrieval')
        parser.add_argument('--save_features', action='store_true')
        return parser


class DenseRetrievalConfig(ConfigBase):
    @property
    def task(self):
        return 'denseir'


class WaPIRLPretrainConfig(ConfigBase):
    @classmethod
    def parse_arguments(cls):
        parents = [
            cls.data_parser(),
            cls.model_parser(),
            cls.augmentation_parser(),
            cls.optimization_parser(),
            cls.logging_parser(),
        ]
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args
        config = cls()
        parser.parse_args(namespace=config)
        return config

    @property
    def task(self):
        return 'wapirl_pretrain'

    @property
    def output_dir(self):
        out = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(out, exist_ok=True)
        return out

    @staticmethod
    def augmentation_parser():
        parser = argparse.ArgumentParser("WaPIRL augmentation", add_help=False)
        parser.add_argument('--augmentation', type=str, default='crop_noise_rotate',
                            choices=('none', 'crop', 'crop_noise', 'crop_rotate', 'crop_noise_rotate'))
        parser.add_argument('--crop_min_scale', type=float, default=0.85)
        parser.add_argument('--noise_prob', type=float, default=0.002)
        parser.add_argument('--rotate_prob', type=float, default=0.5)
        return parser

    @staticmethod
    def optimization_parser():
        parser = argparse.ArgumentParser("WaPIRL optimization", add_help=False)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw'))
        parser.add_argument('--scheduler', type=str, default='cosine', choices=('none', 'cosine'))
        parser.add_argument('--warmup_epochs', type=int, default=5)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
        parser.add_argument('--projector_size', type=int, default=128)
        parser.add_argument('--temperature', type=float, default=0.07)
        parser.add_argument('--loss_weight', type=float, default=0.5)
        parser.add_argument('--num_negatives', type=int, default=1024)
        parser.add_argument('--memory_momentum', type=float, default=0.5)
        parser.add_argument('--save_every', type=int, default=25)
        return parser

    @staticmethod
    def logging_parser():
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='checkpoints')
        parser.add_argument('--write_summary', action='store_true')
        parser.add_argument('--no_quality_filter', dest='quality_filter', action='store_false')
        parser.set_defaults(quality_filter=True)
        parser.add_argument('--min_defect_pixels', type=int, default=3)
        parser.add_argument('--min_defect_ratio', type=float, default=1e-4)
        parser.add_argument('--max_defect_ratio', type=float, default=0.8)
        parser.add_argument('--min_valid_ratio', type=float, default=0.05)
        parser.add_argument('--max_valid_ratio', type=float, default=0.98)
        parser.add_argument('--deduplicate', dest='deduplicate', action='store_true')
        parser.add_argument('--no_deduplicate', dest='deduplicate', action='store_false')
        parser.set_defaults(deduplicate=True)
        return parser
