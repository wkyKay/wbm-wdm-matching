# -*- coding: utf-8 -*-

RESNET_BACKBONE_CONFIGS = RESNET_ENCODER_CONFIGS = {
    '18': {
        'block_type': 'basic',
        'channels': [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2,
        'strides': [1, 1] + [2, 1] + [2, 1] + [2, 1],
        'first_conv': 3,
    },
    '50': {
        'block_type': 'bottleneck',
        'channels': [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3,
        'strides': [1, 1, 1] + [2, 1, 1, 1] + [2, 1, 1, 1, 1, 1] + [2, 1, 1],
        'first_conv': 3,
    },
}


VIT_BACKBONE_CONFIGS = VIT_ENCODER_CONFIGS = {
    'tiny': {
        'patch_size': 16,
        'embed_dim': 192,
        'num_heads': 3,
        'depth': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    'small': {
        'patch_size': 12,
        'embed_dim': 96,
        'num_heads': 4,
        'depth': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    'micro': {
        'patch_size': 16,
        'embed_dim': 96,
        'num_heads': 3,
        'depth': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
}
