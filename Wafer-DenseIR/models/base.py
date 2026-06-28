# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    def __init__(self, layer_config: dict, in_channels: int):
        super(BackboneBase, self).__init__()
        assert isinstance(layer_config, dict)
        assert in_channels in [1, 2, 3]

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def load_weights_from_checkpoint(self, path: str, key: str = None):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if key is not None:
            state = ckpt[key]
        elif 'backbone' in ckpt:
            state = ckpt['backbone']
        elif 'encoder' in ckpt:
            state = ckpt['encoder']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        self.load_state_dict(self._strip_module_prefix(state), strict=False)

    @staticmethod
    def _strip_module_prefix(state):
        return {k.replace('module.', '', 1): v for k, v in state.items()}

