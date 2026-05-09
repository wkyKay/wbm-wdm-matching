# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_layers(self, layer_names: list):
        """冻结指定名称的层，如 ['layer1', 'layer2']。"""
        for name, param in self.named_parameters():
            if any(name.startswith(ln) for ln in layer_names):
                param.requires_grad = False

    def load_weights_from_checkpoint(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('backbone', ckpt)
        self.load_state_dict(state, strict=strict)


class HeadBase(nn.Module):
    def load_weights_from_checkpoint(self, path: str, key: str = 'classifier', strict: bool = True):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get(key, ckpt)
        self.load_state_dict(state, strict=strict)
