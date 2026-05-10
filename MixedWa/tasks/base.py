# -*- coding: utf-8 -*-

import os
import json


class Task:
    def __init__(self):
        self.checkpoint_dir = None

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self):
        raise NotImplementedError

    def save_config(self, config: dict):
        """将训练参数保存到 checkpoint_dir/config.json。"""
        path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @property
    def best_ckpt(self):
        return os.path.join(self.checkpoint_dir, 'best_model.pt')

    @property
    def last_ckpt(self):
        return os.path.join(self.checkpoint_dir, 'last_model.pt')
