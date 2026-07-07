from __future__ import annotations

from typing import List

import numpy as np


def _connected_components(mask: np.ndarray, connectivity: int = 8) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    elif connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError(f"Unsupported connectivity: {connectivity}")

    for row in range(h):
        for col in range(w):
            if not mask[row, col] or visited[row, col]:
                continue
            queue = [(row, col)]
            visited[row, col] = True
            comp = []
            while queue:
                r, c = queue.pop(0)
                comp.append((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(comp, dtype=np.int64))
    return components


def _perimeter(rows: np.ndarray, cols: np.ndarray) -> int:
    pixel_set = set((int(r), int(c)) for r, c in zip(rows, cols))
    perimeter = 0
    for r, c in pixel_set:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) not in pixel_set:
                perimeter += 1
    return perimeter


def _binary_closing_square(mask: np.ndarray) -> np.ndarray:
    dilated = _binary_dilation_square(mask)
    return _binary_erosion_square(dilated)


def _binary_dilation_square(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )


def _binary_erosion_square(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )