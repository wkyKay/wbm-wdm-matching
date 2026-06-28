# -*- coding: utf-8 -*-

import numpy as np


def relevance_matrix(labels):
    labels = labels.astype(np.int32)
    inter = labels @ labels.T
    row_sum = labels.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - inter
    hit = inter > 0
    jaccard = inter / np.maximum(union, 1)
    np.fill_diagonal(hit, False)
    np.fill_diagonal(jaccard, 0.0)
    return hit, jaccard


def retrieval_metrics(rankings, labels, ks=(1, 5, 10)):
    hit_rel, jac_rel = relevance_matrix(labels)
    out = {}
    aps = []
    ndcgs = {k: [] for k in ks}
    precisions = {k: [] for k in ks}
    recalls = {k: [] for k in ks}

    for q, ranked in enumerate(rankings):
        rel = hit_rel[q, ranked].astype(np.float32)
        jac = jac_rel[q, ranked].astype(np.float32)
        total = max(hit_rel[q].sum(), 1)
        cum = np.cumsum(rel)
        pos = np.where(rel > 0)[0]
        aps.append(float((cum[pos] / (pos + 1)).sum() / total) if len(pos) else 0.0)
        for k in ks:
            kk = min(k, len(ranked))
            top_rel = rel[:kk]
            precisions[k].append(float(top_rel.mean()) if kk else 0.0)
            recalls[k].append(float(top_rel.sum() / total))
            gains = jac[:kk]
            discount = 1.0 / np.log2(np.arange(2, kk + 2))
            dcg = float((gains * discount).sum())
            ideal = np.sort(jac_rel[q])[::-1][:kk]
            idcg = float((ideal * discount).sum())
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    out['mAP'] = float(np.mean(aps))
    for k in ks:
        out[f'Precision@{k}'] = float(np.mean(precisions[k]))
        out[f'Recall@{k}'] = float(np.mean(recalls[k]))
        out[f'NDCG@{k}'] = float(np.mean(ndcgs[k]))
    return out
