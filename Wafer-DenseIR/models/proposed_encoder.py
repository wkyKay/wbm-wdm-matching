"""DenseIR adapter for the shared proposed-method encoders."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proposed.models.encoder import build_encoder


def build_dense_encoder(name: str, embedding_dim: int = 256, width: int = 32):
    """Build the exact proposed encoder with DenseIR's 3-channel whole-map input."""
    return build_encoder(name, in_channels=3, embedding_dim=embedding_dim, width=width)


def extract_dense_features(encoder, x):
    if not hasattr(encoder, 'forward_features'):
        raise TypeError('DenseIR requires an encoder with forward_features(x).')
    return encoder.forward_features(x)


def load_encoder_checkpoint(encoder, path, key: str | None = None):
    checkpoint = torch.load(path, map_location='cpu')
    if key and isinstance(checkpoint, dict) and key in checkpoint:
        state = checkpoint[key]
    elif isinstance(checkpoint, dict) and 'encoder' in checkpoint:
        state = checkpoint['encoder']
    elif isinstance(checkpoint, dict) and 'backbone' in checkpoint:
        state = checkpoint['backbone']
    else:
        state = checkpoint
    state = {name.replace('module.', '', 1): value for name, value in state.items()}
    encoder.load_state_dict(state)
