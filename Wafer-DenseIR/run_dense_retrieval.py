# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.task_configs import DenseRetrievalConfig
from datasets.wm38k import WM38K
from evaluation.experiment_a.evaluate_rankings import evaluate_rankings_from_files, write_flat_metrics
from models.proposed_encoder import build_dense_encoder, load_encoder_checkpoint
from shared.wm38k.candidates import load_candidate_manifest
from tasks.dense_retrieval import DenseRetrieval


def main():
    config = DenseRetrievalConfig.parse_arguments()
    config.save()
    device = _get_device(config.device)
    data_file = _resolve_path(config.data_file)

    dataset = WM38K(
        npz_file=data_file,
        input_size=config.input_size,
        split=config.split,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        max_samples=config.max_samples,
        split_manifest=_resolve_optional_path(config.split_manifest),
        query_manifest=None,
    )
    query_ids = _load_query_ids(_resolve_optional_path(config.query_manifest))
    candidate_ids_by_query = _load_candidate_manifest(_resolve_optional_path(config.candidate_manifest))

    backbone = build_dense_encoder(config.encoder, config.embedding_dim, config.encoder_width)
    if config.pretrained_model_file is not None:
        load_encoder_checkpoint(backbone, config.pretrained_model_file, key=config.pretrained_model_key)

    task = DenseRetrieval(backbone=backbone, device=device, output_dir=config.output_dir)
    records = task.extract_features(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        token_mode=config.token_mode,
        token_dilation=config.token_dilation,
        max_tokens=config.max_tokens,
    )
    if config.save_features:
        task.save_features(records)
    rankings, scores, metrics, match_cache = task.run_retrieval(
        records,
        topk_tokens=config.topk_tokens,
        sigma_pos=config.sigma_pos,
        ks=tuple(config.topk_retrieval),
        query_ids=query_ids,
        candidate_ids_by_query=candidate_ids_by_query,
    )
    task.save_explanations(
        records,
        rankings,
        scores,
        match_cache,
        num_queries=config.explain_top_queries,
        query_ids=query_ids,
    )
    split_manifest = _resolve_optional_path(config.split_manifest)
    query_manifest = _resolve_optional_path(config.query_manifest)
    candidate_manifest = _resolve_optional_path(config.candidate_manifest)
    if split_manifest and query_manifest:
        label_metrics = evaluate_rankings_from_files(
            rankings_path=os.path.join(config.output_dir, 'rankings.csv'),
            split_manifest=split_manifest,
            query_manifest=query_manifest,
            candidate_manifest=candidate_manifest,
            split=config.split,
            ks=tuple(config.topk_retrieval),
            relevance_mode='jaccard',
            gain_mode='identity',
            strict=False,
        )
        label_metrics_path = os.path.join(config.output_dir, 'label_metrics.json')
        with open(label_metrics_path, 'w') as f:
            import json
            json.dump(label_metrics, f, indent=2)
        write_flat_metrics(label_metrics, os.path.join(config.output_dir, 'label_metrics_flat.csv'))
        print(f'Official label metrics: {label_metrics_path}')

    print(f'Output directory: {config.output_dir}')
    print(metrics)


def _get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        torch.cuda.set_device(device.index if device.index is not None else 0)
        print(f'[Device] GPU: {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})')
        torch.backends.cudnn.benchmark = True
    else:
        print('[Device] GPU not available, falling back to CPU')
    return device


def _resolve_path(path):
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, path)),
        os.path.abspath(os.path.join(here, '..', '..', path)),
        os.path.abspath(os.path.join(here, '..', '..', '..', path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_optional_path(path):
    return None if path is None else _resolve_path(path)


def _load_query_ids(path):
    if path is None:
        return None
    import csv
    with open(path, 'r', newline='') as f:
        return [int(row['sample_id']) for row in csv.DictReader(f)]


def _load_candidate_manifest(path):
    return None if path is None else load_candidate_manifest(path)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
