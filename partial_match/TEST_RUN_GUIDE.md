# Proposal-based Local Retrieval Run Guide

This guide shows how to run the current proposal-based local retrieval baseline.

## A. Dataset

Place the MixedWM38K dataset at:

```text
../data/wm38k/Wafer_Map_Datasets.npz
```

Run commands from:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
```

## B. One-command Pipeline

Run proposal extraction, descriptor embedding, local retrieval, metrics, and Top3 review figures:

```bash
python3 partial_match/run_proposal_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --out-dir ../artifacts/proposal_based/system_test_512_stratified \
  --max-samples 512 \
  --sample-strategy stratified \
  --seed 42 \
  --review-max-queries 64 \
  --review-top-k 3 \
  --metric-k 1 3 5 10
```

Output files:

```text
../artifacts/proposal_based/system_test_512_stratified/
├── rankings.csv
├── tokens.csv
├── descriptors.npz
├── metrics_summary.json
├── metrics_summary_flat.csv
├── top3_review/
└── top3_review/
```

Run help:

```bash
python3 partial_match/run_proposal_retrieval_pipeline.py --help
```

## C. Step-by-step Run

### C-1. Retrieval

Generate proposal tokens, descriptors, and retrieval rankings:

```bash
python3 partial_match/scripts/run_proposal_local_retrieval.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --out ../artifacts/proposal_based/system_test_512_stratified/rankings.csv \
  --max-samples 512 \
  --sample-strategy stratified \
  --seed 42 \
  --save-token-details
```

### C-2. Evaluation

Compute retrieval metrics and proposal statistics:

```bash
python3 partial_match/scripts/evaluate_proposal_retrieval.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --rankings ../artifacts/proposal_based/system_test_512_stratified/rankings.csv \
  --tokens ../artifacts/proposal_based/system_test_512_stratified/tokens.csv \
  --out ../artifacts/proposal_based/system_test_512_stratified/metrics_summary.json \
  --k 1 3 5 10
```

### C-3. TopK Review Figures

Generate query + Top3 retrieval comparison figures:

```bash
python3 partial_match/scripts/visualize_topk_retrieval.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --rankings ../artifacts/proposal_based/system_test_512_stratified/rankings.csv \
  --out-dir ../artifacts/proposal_based/system_test_512_stratified/top3_review \
  --max-queries 64 \
  --top-k 3
```

## D. Common Options

Change the retrieval pool size:

```bash
--max-samples 1024
```

Change the number of TopK review figures:

```bash
--review-max-queries 128
```

Show query + Top5 instead of query + Top3:

```bash
--review-top-k 5
```
