# Project Structure

This repository contains several related wafer-map matching experiments, but not every top-level directory belongs to the current retrieval plan. Use this file to decide which code to inspect first.

## Current Plan

The current plan is described in:

- `/Users/kayw/Documents/trae_projects/match-test/tasks/plan.md`

The active experimental goal is wafer-map retrieval on MixedWM38K and related WBM/WDM study cases, with shared train / validation / test splits and comparable retrieval outputs.

## Directly Relevant Directories

### `partial_match/`

Traditional local-retrieval baseline for the current plan. It implements cluster proposal, handcrafted cluster descriptors, local token matching, proposal-based retrieval pipelines, evaluation helpers, and visualization utilities. Inspect this directory when working on the traditional baseline, proposal behavior, handcrafted descriptors, local matching, or proposal-based review figures.

Important entry points:

- `partial_match/run_proposal_retrieval_pipeline.py`
- `partial_match/scripts/run_proposal_local_retrieval.py`
- `partial_match/scripts/evaluate_proposal_retrieval.py`
- `partial_match/core/`
- `partial_match/docs/`

### `Wafer-DenseIR/`

Machine-learning retrieval baseline for the current plan. It implements WaPIRL-style self-supervised training, dense feature extraction, proposal-free dense token matching, retrieval rankings, and local heatmap explanations. Inspect this directory when working on the DenseIR baseline, dense feature tokens, WaPIRL-style pretraining adapted for retrieval, or dense retrieval outputs.

Important entry points:

- `Wafer-DenseIR/run_wapirl_pretrain.py`
- `Wafer-DenseIR/run_extract_features.py`
- `Wafer-DenseIR/run_dense_retrieval.py`
- `Wafer-DenseIR/datasets/`
- `Wafer-DenseIR/tasks/`
- `Wafer-DenseIR/utils/`

### `proposed/`

Main proposed method for the current plan. It should use the same proposal and local token matching protocol as `partial_match`, while replacing handcrafted cluster descriptors with WaPIRL-style self-supervised cluster embeddings implemented inside this directory.

Important entry points:

- `proposed/README_INTERNAL.md`
- `proposed/run_cluster_pretrain.py`
- `proposed/run_learned_retrieval_pipeline.py`
- `proposed/core/`
- `proposed/datasets/`
- `proposed/tasks/`

### `match/`

Production-data study-case code. This directory is for WBM/WDM matching experiments outside the public MixedWM38K-only benchmark. Inspect this directory when working on production WDM-to-WBM conversion, classnumber-based matching, production study-case scripts, or production visualizations.

Important entry points:

- `match/core/`
- `match/data/`
- `match/scripts/`
- `match/viz/`
- `match/docs/`

### `shared/`

Shared utilities used by all current experimental methods. At present it owns the MixedWM38K split and query manifest logic, so methods should consume the frozen manifests generated here instead of creating separate random splits.

Important entry points:

- `shared/README.md`
- `shared/wm38k/io.py`
- `shared/wm38k/split.py`
- `shared/wm38k/query.py`
- `shared/wm38k/manifest.py`
- `shared/wm38k/cli_build_split.py`

### `evaluation/`

Method-independent evaluation code for the current plan. It consumes ranking files from `partial_match`, `Wafer-DenseIR`, or later methods, reads the shared split/query manifests, and computes official label-derived retrieval metrics. Inspect this directory when working on Label NDCG, MeanJaccard, ExactRate, HitRate, mAP, ranking schema validation, or cross-method metric comparability.

Important entry points:

- `evaluation/README.md`
- `evaluation/evaluate_rankings.py`
- `evaluation/relevance.py`
- `evaluation/metrics.py`
- `evaluation/schemas.py`

## Background Or Currently Out Of Scope

### `WaPIRL/`

Reference implementation from another paper. Some current methods are inspired by or adapted from WaPIRL, but this directory is not the primary implementation target for the current plan unless the task explicitly asks to inspect the original WaPIRL code.

### `MixedWa/`

Earlier WaPIRL-based adaptation work. It is related historically, but it is not part of the current plan unless a task explicitly asks about MixedWa or legacy comparisons.

## Shared Data Protocol

All comparable methods should use the same split and query manifests:

- `artifacts/splits/wm38k_seed2026_sig_70_10_20.csv`
- `artifacts/splits/wm38k_seed2026_test_queries_2000.csv`
- `artifacts/splits/wm38k_seed2026_test_candidates_1000.csv`

The intended protocol is:

- `train`: self-supervised or unsupervised representation learning only.
- `valid`: hyperparameter and checkpoint selection.
- `test`: final retrieval evaluation only.
- final query set: fixed query IDs from the query manifest.
- default full-pool candidate set: test split from the split manifest, excluding the query itself.
- controlled candidate set: fixed per-query candidates from the candidate manifest.

All retrieval methods should output compatible ranking files with:

- `query_id`
- `candidate_id`
- `similarity_score`

## Where To Look First

For data split or query-set questions, inspect `shared/` first. For official retrieval metrics or ranking-file evaluation, inspect `evaluation/` first. For the main proposed local learned cluster retrieval method, inspect `proposed/` first. For traditional local handcrafted retrieval, inspect `partial_match/` first. For dense learned retrieval, inspect `Wafer-DenseIR/` first. For production WBM/WDM study cases, inspect `match/` first. Avoid spending time in `WaPIRL/` or `MixedWa/` unless the task explicitly asks for those legacy or reference implementations.
