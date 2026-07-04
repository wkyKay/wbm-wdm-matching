# -*- coding: utf-8 -*-
"""Run cluster-level contrastive pretraining for the proposed method."""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from proposed.core.cluster_patches import PatchBuilder, PatchConfig
from proposed.core.proposal import PartialMatchProposalProvider, ProposalConfig, save_proposal_config
from proposed.datasets.cluster_contrastive import ClusterContrastiveDataset, write_patch_manifest
from proposed.datasets.wm38k_maps import load_split_records
from proposed.models.encoder import ClusterEncoder
from proposed.models.head import MLPHead
from proposed.tasks.cluster_pretrain import ClusterPretrainTask, MemoryBank
from proposed.utils.logging import get_logger
from proposed.utils.loss import ClusterNCELoss
from proposed.utils.optimization import get_optimizer, get_scheduler


def main():
    args = parse_args()
    _seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_config(out_dir / 'configs.json', vars(args))

    proposal_config = ProposalConfig(
        method=args.proposal_method,
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        enable_ring_aware=not args.disable_ring_aware,
        max_defect_ratio_for_ring=args.max_defect_ratio_for_ring,
        min_edge_defect_fraction_for_ring=args.min_edge_defect_fraction_for_ring,
    )
    patch_config = PatchConfig(patch_size=args.patch_size, window_size=args.patch_window_size)
    save_proposal_config(out_dir / 'proposal_config.json', proposal_config)
    _save_config(out_dir / 'patch_config.json', asdict(patch_config))

    provider = PartialMatchProposalProvider(proposal_config)
    patch_builder = PatchBuilder(patch_config)
    train_records = load_split_records(args.data_file, args.split_manifest, split=args.split)
    valid_records = load_split_records(args.data_file, args.split_manifest, split=args.valid_split)
    train_set = ClusterContrastiveDataset(
        train_records,
        proposal_provider=provider,
        patch_builder=patch_builder,
        max_clusters=args.max_train_clusters,
        seed=args.seed,
        tokens_csv=str(out_dir / 'train_proposal_tokens.csv'),
    )
    valid_set = ClusterContrastiveDataset(
        valid_records,
        proposal_provider=provider,
        patch_builder=patch_builder,
        max_clusters=args.max_valid_clusters,
        seed=args.seed + 1,
        tokens_csv=str(out_dir / 'valid_proposal_tokens.csv'),
    )
    write_patch_manifest(out_dir / 'train_cluster_patches_manifest.csv', train_set.tokens, patch_config)
    write_patch_manifest(out_dir / 'valid_cluster_patches_manifest.csv', valid_set.tokens, patch_config)

    device = _get_device(args.device)
    encoder = ClusterEncoder(in_channels=3, embedding_dim=args.embedding_dim, width=args.encoder_width)
    projector = MLPHead(args.embedding_dim, args.projector_size)
    optimizer = get_optimizer(
        [{'params': encoder.parameters()}, {'params': projector.parameters()}],
        name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = get_scheduler(optimizer, name=args.scheduler, epochs=args.epochs, warmup_epochs=args.warmup_epochs)
    memory = MemoryBank(size=(len(train_set), args.projector_size), device=device, weight=args.memory_momentum)
    task = ClusterPretrainTask(
        encoder=encoder,
        projector=projector,
        memory=memory,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=ClusterNCELoss(args.temperature),
        device=device,
        output_dir=str(out_dir),
        loss_weight=args.loss_weight,
        num_negatives=args.num_negatives,
    )
    logger = get_logger(out_dir / 'main.log')
    logger.info(f'Train cluster tokens: {len(train_set)}')
    logger.info(f'Valid cluster tokens: {len(valid_set)}')
    task.run(
        train_set=train_set,
        valid_set=valid_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logger=logger,
        save_every=args.save_every,
    )
    print(f'Checkpoint directory: {out_dir}')
    print(f'Best checkpoint: {out_dir / "best_model.pt"}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--split-manifest', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--valid-split', type=str, default='valid')
    parser.add_argument('--proposal-method', type=str, default='retrieval_compact')
    parser.add_argument('--min-area', type=int, default=5)
    parser.add_argument('--top-k-proposals', type=int, default=6)
    parser.add_argument('--disable-ring-aware', action='store_true')
    parser.add_argument('--max-defect-ratio-for-ring', type=float, default=0.45)
    parser.add_argument('--min-edge-defect-fraction-for-ring', type=float, default=0.45)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--patch-window-size', type=int, default=52)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--encoder-width', type=int, default=32)
    parser.add_argument('--projector-size', type=int, default=256)
    parser.add_argument('--max-train-clusters', type=int, default=None)
    parser.add_argument('--max-valid-clusters', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--num-negatives', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--loss-weight', type=float, default=0.5)
    parser.add_argument('--memory-momentum', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out-dir', type=str, default='artifacts/proposed/cluster_pretrain')
    return parser.parse_args()


def _get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu for local debugging.')
    return device


def _save_config(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()

