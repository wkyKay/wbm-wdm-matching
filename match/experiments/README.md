# Match Experiments

## Mixed38K shifted-target retrieval

`mixed38k_shift_retrieval.py` evaluates whether the current local matching
pipeline can retrieve a translated same-pattern map from distractors.

For each single-class pattern in Mixed38K:

- sample one single-class query map;
- create one positive target by translating the query defect mask;
- sample the remaining gallery entries as evenly as possible from other
  single-class patterns;
- rank all gallery maps with `explain_count_partial_match`;
- report top10, top5, top3, top1, and rank-based target metrics;
- save a query + top10 figure, with the shifted target marked in red if it
  appears in the top10.

Run the default experiment:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_shift_retrieval
```

The default dataset path is:

```text
data/wm38k/Wafer_Map_Datasets.npz
```

The default output directory is:

```text
wbm-wdm-matching/match/experiments/artifacts/mixed38k_shift_retrieval
```

Useful options:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_shift_retrieval \
  --trials-per-class 20 \
  --gallery-size 100 \
  --proposal-mode compact \
  --shift-row 3 \
  --shift-col -3
```

Outputs:

- `metrics.json`: overall and per-class top10/top5/top3/top1 accuracy,
  mean target rank, median target rank, MRR, and mean rank percentile.
- `trials.csv`: one row per query trial, including target rank, reciprocal
  rank, rank percentile, hit flags, and distractor class counts.
- `rankings.csv`: full candidate ranking for every trial.
- `figures/*.png`: query plus top10 retrieval visualization for each trial.

## Mixed38K class retrieval

`mixed38k_class_retrieval.py` evaluates class-level retrieval. For each
single-class query, the gallery contains multiple same-label positives and
balanced other-label negatives.

Default setup:

- 20 query trials per class;
- 10 same-label positives per trial;
- 90 other-label negatives per trial;
- negatives sampled as evenly as possible from the other 7 classes;
- query itself is excluded from positives.

Run:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_class_retrieval
```

Useful options:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_class_retrieval \
  --trials-per-class 20 \
  --positives-per-trial 10 \
  --negatives-per-trial 90 \
  --proposal-mode compact
```

When compact misses broken ring/arc evidence, compare against the arc mode:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_class_retrieval \
  --trials-per-class 20 \
  --positives-per-trial 10 \
  --negatives-per-trial 90 \
  --proposal-mode arc
```

Outputs:

- `metrics.json`: overall and per-class precision@k, recall@k, hit@k,
  nDCG@k, AP, nDCG, MRR, and positive-rank summaries.
- `trials.csv`: one row per query trial, including class counts and all
  retrieval metrics.
- `rankings.csv`: full candidate ranking for every trial.
- `figures/*.png`: query, top10 retrieval results, and any same-label
  positives outside top10. Each column shows the raw map on top and its
  proposal/token visualization below. Same-label positives are marked in red.

## Mixed38K single-to-multi retrieval

`mixed38k_single_to_multi_retrieval.py` evaluates whether a single-label query
can retrieve real Mixed38K multi-label maps that contain the query pattern.
For example, a `center` query has positives whose label sets include `center`;
all negatives have no `center` label.

Default setup:

- 50 query trials per class;
- 5 positive candidates per trial;
- 95 negative candidates per trial;
- positives are real multi-label maps containing the query class;
- negatives are real maps whose label set does not contain the query class;
- classes with no multi-label positives are skipped by default.

Run:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_single_to_multi_retrieval
```

Useful options:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_single_to_multi_retrieval \
  --class-name center \
  --trials-per-class 50 \
  --positives-per-trial 5 \
  --negatives-per-trial 95
```

Outputs:

- `metrics.json`: overall and per-class top5 positive hit/recall, precision@k,
  recall@k, hit@k, nDCG@k, AP, MRR, and positive-rank summaries.
- `trials.csv`: one row per query trial, including positive and negative label
  counts.
- `rankings.csv`: full candidate ranking with candidate label sets.
- `figures/*.png`: query, top retrieval results, and positives outside the
  displayed top-k.

## Mixed38K proposal check

`mixed38k_proposal_check.py` samples one wafer map from every Mixed38K label
combination, including both single-pattern and mixed-pattern maps, runs
proposal extraction, and renders raw/proposal views for quick visual
inspection.

Run:

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.experiments.mixed38k_proposal_check \
  --proposal-mode arc
```

Outputs:

- `mixed38k_all_pattern_proposals_page*.png`: paged overview figures covering
  all detected single and mixed pattern types.
- `figures/*.png`: one raw/proposal figure per pattern type.
- `summary.csv`: one row per generated token/proposal.
- `summary.json`: sampled map ids, token summaries, and proposal debug data.
