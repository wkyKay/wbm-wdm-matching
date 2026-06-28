# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F


def dense_tokenize(feature_map, defect_mask, valid_mask, token_mode='defect_band', token_dilation=1, max_tokens=256):
    c, h, w = feature_map.shape
    feat = F.normalize(feature_map, dim=0)
    defect = _resize_mask(defect_mask, (h, w))
    valid = _resize_mask(valid_mask, (h, w))

    if token_mode == 'defect':
        token_mask = defect > 0
    elif token_mode == 'defect_band':
        token_mask = _dilate_mask(defect > 0, token_dilation)
    elif token_mode == 'valid':
        token_mask = valid > 0
    elif token_mode == 'all':
        token_mask = torch.ones((h, w), dtype=torch.bool, device=feature_map.device)
    else:
        raise ValueError(f'Unknown token_mode: {token_mode}')

    if not token_mask.any():
        token_mask = valid > 0
    if not token_mask.any():
        token_mask = torch.ones((h, w), dtype=torch.bool, device=feature_map.device)

    ys, xs = torch.where(token_mask)
    scores = defect[ys, xs] + 0.1 * valid[ys, xs]
    if len(ys) > max_tokens:
        keep = torch.topk(scores, k=max_tokens, largest=True).indices
        ys, xs = ys[keep], xs[keep]
        scores = scores[keep]
    tokens = feat[:, ys, xs].transpose(0, 1).contiguous()
    pos = torch.stack([
        ys.float() / max(h - 1, 1),
        xs.float() / max(w - 1, 1),
    ], dim=1)
    weights = torch.clamp(scores, min=0.1)
    return {
        'tokens': tokens.cpu().numpy().astype(np.float32),
        'pos': pos.cpu().numpy().astype(np.float32),
        'weights': weights.cpu().numpy().astype(np.float32),
        'grid_size': np.array([h, w], dtype=np.int32),
    }


def dense_match(query, candidate, topk_tokens=5, sigma_pos=0.35):
    q = query['tokens']
    c = candidate['tokens']
    if len(q) == 0 or len(c) == 0:
        return 0.0, np.zeros(len(q), dtype=np.float32), np.zeros(len(c), dtype=np.float32), []

    sim = q @ c.T
    dist2 = ((query['pos'][:, None, :] - candidate['pos'][None, :, :]) ** 2).sum(axis=2)
    sim = sim * np.exp(-dist2 / max(sigma_pos ** 2, 1e-6))
    k = min(topk_tokens, sim.shape[1])
    top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(sim, top_idx, axis=1)
    q_scores = top_scores.mean(axis=1)
    q_weights = query.get('weights', np.ones(len(q), dtype=np.float32))
    score = float((q_scores * q_weights).sum() / max(q_weights.sum(), 1e-6))

    best_j = sim.argmax(axis=1)
    c_scores = np.zeros(len(c), dtype=np.float32)
    np.maximum.at(c_scores, best_j, sim[np.arange(len(q)), best_j])
    matches = [(int(i), int(best_j[i]), float(sim[i, best_j[i]])) for i in np.argsort(-q_scores)]
    return score, q_scores.astype(np.float32), c_scores, matches


def _pair_score_gpu(q_tokens, c_tokens, q_pos, c_pos, q_weights, topk_tokens, sigma_pos):
    """GPU-accelerated pair score computation. All inputs are torch tensors on GPU."""
    if len(q_tokens) == 0 or len(c_tokens) == 0:
        return 0.0
    
    sim = q_tokens @ c_tokens.T  # (Mq, Mc)
    dist2 = ((q_pos[:, None, :] - c_pos[None, :, :]) ** 2).sum(dim=2)
    sim = sim * torch.exp(-dist2 / max(sigma_pos ** 2, 1e-6))
    
    k = min(topk_tokens, sim.shape[1])
    top_scores, _ = torch.topk(sim, k, dim=1)  # (Mq, k)
    q_scores = top_scores.mean(dim=1)  # (Mq,)
    
    weights = q_weights if q_weights is not None else torch.ones(len(q_tokens), device=q_tokens.device)
    score = float((q_scores * weights).sum() / max(weights.sum(), 1e-6))
    return score


def make_heatmap(token_scores, positions, grid_size):
    h, w = int(grid_size[0]), int(grid_size[1])
    heatmap = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for score, (py, px) in zip(token_scores, positions):
        y = min(max(int(round(py * max(h - 1, 1))), 0), h - 1)
        x = min(max(int(round(px * max(w - 1, 1))), 0), w - 1)
        heatmap[y, x] += score
        count[y, x] += 1
    heatmap = heatmap / np.maximum(count, 1)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def _resize_mask(mask, size):
    x = mask.float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(x, size=size, mode='nearest')
    return out.squeeze(0).squeeze(0)


def _dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(0).unsqueeze(0)
    out = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out.squeeze(0).squeeze(0).bool()
