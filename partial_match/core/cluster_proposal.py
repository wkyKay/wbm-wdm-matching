# -*- coding: utf-8 -*-
"""
Cluster Proposal Module
Implement three types of cluster proposals:
- Raw connected components
- Closed connected components (morphological closing)
- Filtered connected components (small cluster removal)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

try:
    from scipy.ndimage import binary_closing, generate_binary_structure
except ImportError:
    print("scipy not found. Using custom implementation of binary_closing.")

    # Custom morphological closing implementation
    def generate_binary_structure(rank: int, connectivity: int):
        if rank == 2 and connectivity == 2:
            return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        elif rank == 2 and connectivity == 1:
            return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:
            raise NotImplementedError("Only 2D binary structures are implemented.")

    def custom_binary_closing(image: np.ndarray, structure: Optional[np.ndarray] = None):
        if structure is None:
            structure = generate_binary_structure(2, 2)
        pad = structure.shape[0] // 2
        padded = np.pad(image, pad, mode='constant')
        H_img, W_img = image.shape
        # Erosion: pixel is True only if ALL structure=True positions in region are True
        eroded = np.zeros_like(image, dtype=bool)
        for i in range(H_img):
            for j in range(W_img):
                region = padded[i:i+structure.shape[0], j:j+structure.shape[1]]
                if np.all(region[structure]):
                    eroded[i, j] = True
        # Dilation: pixel is True if ANY structure=True position in region is True
        padded_e = np.pad(eroded, pad, mode='constant')
        dilated = np.zeros_like(image, dtype=bool)
        for i in range(H_img):
            for j in range(W_img):
                region = padded_e[i:i+structure.shape[0], j:j+structure.shape[1]]
                if np.any(region & structure):
                    dilated[i, j] = True
        return dilated


def connected_components_with_stats(mask: np.ndarray, connectivity: int = 2) -> List[Dict]:
    """
    Find connected components in a binary mask and compute statistics for each.
    
    Args:
        mask: Binary mask (defect regions)
        connectivity: Connectivity (1 for 4-connectivity, 2 for 8-connectivity)
        
    Returns:
        List of dictionaries with component stats
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    
    # 8-connectivity neighbors
    if connectivity == 2:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]
    else:  # 4-connectivity
        neighbors = [          (-1, 0),
                     (0, -1),          (0, 1),
                               (1, 0)]
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                # BFS to find component
                queue = [(i, j)]
                visited[i, j] = True
                comp_pixels = []
                
                while queue:
                    x, y = queue.pop(0)
                    comp_pixels.append((x, y))
                    
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                # Compute stats
                if comp_pixels:
                    comp = _compute_component_stats(comp_pixels, H, W)
                    components.append(comp)
    
    return components


def _compute_component_stats(pixels: List[Tuple[int, int]], H: int, W: int) -> Dict:
    """
    Compute statistics for a connected component.
    
    Args:
        pixels: List of (row, col) pixel coordinates in the component
        H: Height of original image
        W: Width of original image
        
    Returns:
        Dictionary with component statistics
    """
    pixels_arr = np.array(pixels)
    rows = pixels_arr[:, 0]
    cols = pixels_arr[:, 1]
    
    # Area
    area = len(pixels)
    
    # Centroid
    centroid_row = rows.mean()
    centroid_col = cols.mean()
    centroid_row_norm = centroid_row / H
    centroid_col_norm = centroid_col / W
    
    # Bounding box
    bbox_row_min = rows.min()
    bbox_col_min = cols.min()
    bbox_row_max = rows.max()
    bbox_col_max = cols.max()
    bbox_height = bbox_row_max - bbox_row_min + 1
    bbox_width = bbox_col_max - bbox_col_min + 1
    bbox_area = bbox_height * bbox_width
    
    # Aspect ratio
    aspect_ratio = bbox_width / max(bbox_height, 1)
    
    # Compactness (perimeter / area ratio)
    perimeter = _compute_perimeter(pixels, H, W)
    compactness = perimeter / max(area, 1) if area > 0 else 0.0
    
    # Radial distance (distance from center of image)
    center_row, center_col = H / 2, W / 2
    radial_distance = np.sqrt((centroid_row - center_row) ** 2 + (centroid_col - center_col) ** 2)
    radial_distance_norm = radial_distance / np.sqrt((H / 2) ** 2 + (W / 2) ** 2)
    
    # Covariance and orientation
    if len(pixels) > 1:
        mean_row = centroid_row
        mean_col = centroid_col
        cov_rr = np.mean((rows - mean_row) ** 2)
        cov_cc = np.mean((cols - mean_col) ** 2)
        cov_rc = np.mean((rows - mean_row) * (cols - mean_col))
        
        cov_matrix = np.array([[cov_rr, cov_rc], [cov_rc, cov_cc]])
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        pca_lambda1 = eigenvalues[1]
        pca_lambda2 = eigenvalues[0]
        
        if eigenvalues[1] > 0:
            angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            orientation = np.degrees(angle)
        else:
            orientation = 0.0
    else:
        cov_rr = cov_cc = cov_rc = 0.0
        pca_lambda1 = pca_lambda2 = 0.0
        orientation = 0.0
    
    return {
        'area': area,
        'area_ratio': area / (H * W),
        'centroid_row': centroid_row,
        'centroid_col': centroid_col,
        'centroid_row_norm': centroid_row_norm,
        'centroid_col_norm': centroid_col_norm,
        'bbox_row_min': bbox_row_min,
        'bbox_col_min': bbox_col_min,
        'bbox_row_max': bbox_row_max,
        'bbox_col_max': bbox_col_max,
        'bbox_height': bbox_height,
        'bbox_width': bbox_width,
        'bbox_area': bbox_area,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'perimeter': perimeter,
        'radial_distance_norm': radial_distance_norm,
        'mean_row': mean_row if len(pixels) > 1 else centroid_row,
        'mean_col': mean_col if len(pixels) > 1 else centroid_col,
        'cov_rr': cov_rr,
        'cov_cc': cov_cc,
        'cov_rc': cov_rc,
        'pca_lambda1': pca_lambda1,
        'pca_lambda2': pca_lambda2,
        'orientation': orientation,
        'pixels': pixels,  # Store pixel coordinates for later use
        'pixel_coords': [(int(r), int(c)) for r, c in pixels],  # 转换为整数列表用于可视化
    }


def _compute_perimeter(pixels: List[Tuple[int, int]], H: int, W: int) -> int:
    """
    Compute perimeter of a component using 8-connectivity.
    """
    pixel_set = set(pixels)
    perimeter = 0
    
    for x, y in pixels:
        # Check 4 neighbors
        if (x - 1, y) not in pixel_set or x - 1 < 0:
            perimeter += 1
        if (x + 1, y) not in pixel_set or x + 1 >= H:
            perimeter += 1
        if (x, y - 1) not in pixel_set or y - 1 < 0:
            perimeter += 1
        if (x, y + 1) not in pixel_set or y + 1 >= W:
            perimeter += 1
    
    return perimeter


def spatial_filter(mask: np.ndarray) -> np.ndarray:
    """
    5x5 Spatial Filter from the paper:
    - '1' for strong defects
    - '0.5' for weak defects
    - '0' for noise
    
    Filter weights:
    100 100 100 100 100
    100  10  10  10 100
    100  10   1  10 100
    100  10  10  10 100
    100 100 100 100 100
    """
    H, W = mask.shape
    output = np.zeros_like(mask, dtype=np.float32)
    
    padded = np.pad(mask, 2, mode='constant')
    
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
                
            # Extract 5x5 region
            region = padded[i:i+5, j:j+5]
            
            # Count neighbor support
            center = region[2, 2]
            near_neighbors = region[1:4, 1:4].sum()  # includes center
            far_neighbors = region.sum()  # total in 5x5
            
            # Apply filtering rules
            if near_neighbors >= 3:
                # Strong defect
                output[i, j] = 1.0
            elif far_neighbors >= 5:
                # Weak/uncertain defect
                output[i, j] = 0.5
            else:
                # Isolated noise
                output[i, j] = 0.0
    
    return output


def diamond_structuring_element(size: int = 5) -> np.ndarray:
    """Generate diamond structuring element for closing operation."""
    selem = np.zeros((size, size), dtype=bool)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                selem[i, j] = True
    
    return selem


def cluster_proposal(defect_mask: np.ndarray, 
                     valid_mask: Optional[np.ndarray] = None,
                     proposal_type: str = 'raw',
                     min_area: int = 5,
                     top_k: int = 5,
                     topk_base_method: str = 'geometry_merge',
                     dilation_radius: int = 1,
                     use_closing: bool = False,
                     suspicious_area: int = 40,
                     min_suspicious_cues: int = 2,
                     max_split_count: int = 6,
                     min_split_coverage: float = 0.75,
                     skip_ring_like: bool = True) -> List[Dict]:
    """
    Generate cluster proposals based on defect mask.
    
    Args:
        defect_mask: Binary mask for defect regions
        valid_mask: Optional mask for valid wafer region
        proposal_type: 'raw', 'filtered', 'adhesion', 'dilated_group',
            'dilated_adhesion', 'topk', 'closing', 'simi_paper'
        min_area: Minimum area for filtered clusters
        top_k: Number of regions to keep for topk proposal
        topk_base_method: Candidate generator used by topk proposal
        dilation_radius: Radius for dilated grouping
        use_closing: Whether dilated grouping uses closing instead of dilation
        suspicious_area: Minimum area before adhesion suspicious checks
        min_suspicious_cues: Number of shape cues required before adhesion split
        max_split_count: Reject adhesion split results with too many fragments
        min_split_coverage: Reject split results that keep too little original area
        skip_ring_like: Avoid adhesion split for ring-like dilated groups
        
    Returns:
        List of cluster components with statistics
    """
    H, W = defect_mask.shape
    
    # Apply valid mask
    if valid_mask is not None:
        mask = defect_mask & valid_mask
    else:
        mask = defect_mask.copy()
    
    if proposal_type == 'raw':
        return connected_components_with_stats(mask)
    
    elif proposal_type == 'closing':
        # Full morphological closing with 3x3 cross (better for 52x52 grid)
        cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=bool)
        try:
            from scipy.ndimage import binary_closing
            closed_mask = binary_closing(mask, cross)
        except ImportError:
            closed_mask = custom_binary_closing(mask, cross)
        return connected_components_with_stats(closed_mask)
    
    elif proposal_type == 'filtered':
        # First get raw components
        components = connected_components_with_stats(mask)
        # Filter by area
        filtered_components = [comp for comp in components if comp['area'] >= min_area]
        return filtered_components

    elif proposal_type == 'adhesion':
        from partial_match.core.clustering import cluster
        return cluster(
            defect_mask,
            valid_mask,
            method='adhesion',
            min_area=min_area,
            suspicious_area=suspicious_area,
            min_suspicious_cues=min_suspicious_cues,
            max_split_count=max_split_count,
            min_split_coverage=min_split_coverage,
        )

    elif proposal_type in ('dilated_group', 'dilated', 'dilated_adhesion', 'dilated_group_adhesion',
                           'group_then_adhesion', 'dilated_group_then_adhesion', 'gta',
                           'geometry_merge', 'geom_merge', 'fragment_merge'):
        from partial_match.core.clustering import cluster
        if proposal_type in ('geometry_merge', 'geom_merge', 'fragment_merge'):
            method = 'geometry_merge'
        elif proposal_type in ('group_then_adhesion', 'dilated_group_then_adhesion', 'gta'):
            method = 'group_then_adhesion'
        else:
            method = 'dilated_adhesion' if 'adhesion' in proposal_type else 'dilated_group'
        return cluster(
            defect_mask,
            valid_mask,
            method=method,
            min_area=min_area,
            dilation_radius=dilation_radius,
            use_closing=use_closing,
            suspicious_area=suspicious_area,
            min_suspicious_cues=min_suspicious_cues,
            max_split_count=max_split_count,
            min_split_coverage=min_split_coverage,
            skip_ring_like=skip_ring_like,
        )

    elif proposal_type in ('topk', 'compact', 'adhesion_topk', 'topk_dilated',
                           'topk_geometry_merge', 'topk_geom_merge',
                           'topk_group_then_adhesion', 'topk_gta'):
        from partial_match.core.clustering import cluster
        if proposal_type == 'topk_dilated':
            method = 'topk_dilated'
            base_method = 'dilated_adhesion'
        elif proposal_type in ('topk_geometry_merge', 'topk_geom_merge'):
            method = 'topk'
            base_method = 'geometry_merge'
        elif proposal_type in ('topk_group_then_adhesion', 'topk_gta'):
            method = 'topk'
            base_method = 'group_then_adhesion'
        else:
            method = 'topk'
            base_method = topk_base_method
        return cluster(
            defect_mask,
            valid_mask,
            method=method,
            min_area=min_area,
            top_k=top_k,
            base_method=base_method,
            dilation_radius=dilation_radius,
            use_closing=use_closing,
            suspicious_area=suspicious_area,
            min_suspicious_cues=min_suspicious_cues,
            max_split_count=max_split_count,
            min_split_coverage=min_split_coverage,
            skip_ring_like=skip_ring_like,
        )
    
    elif proposal_type == 'simi_paper':
        # Method from the paper: Closing + Spatial Filter
        # Use 3x3 cross for 52x52 grid (original paper uses 5x5 diamond on larger maps)
        cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=bool)
        try:
            from scipy.ndimage import binary_closing
            closed_mask = binary_closing(mask, cross)
        except ImportError:
            closed_mask = custom_binary_closing(mask, cross)
        
        # Step 2: Spatial Filter to get 0/0.5/1 map
        soft_map = spatial_filter(closed_mask)
        
        # Step 3: Only use strong defects (1.0) for clustering
        strong_defect = soft_map >= 0.9
        return connected_components_with_stats(strong_defect)
    
    else:
        raise ValueError(f"Unknown proposal type: {proposal_type}")


def generate_cluster_tokens(maps: np.ndarray, 
                            original_indices: np.ndarray,
                            proposal_types: List[str] = ('topk',),
                            min_area: int = 5,
                            top_k: int = 5,
                            topk_base_method: str = 'geometry_merge',
                            dilation_radius: int = 1,
                            use_closing: bool = False,
                            suspicious_area: int = 40,
                            min_suspicious_cues: int = 2,
                            max_split_count: int = 6,
                            min_split_coverage: float = 0.75,
                            skip_ring_like: bool = True) -> List[Dict]:
    """
    Generate cluster tokens for all wafer maps.
    
    Args:
        maps: Raw wafer maps
        original_indices: Original indices from dataset
        proposal_types: List of proposal types to generate
        min_area: Minimum area for filtered clusters
        top_k: Number of regions to keep for topk proposal
        topk_base_method: Candidate generator used by topk proposal
        dilation_radius: Radius for dilated grouping
        use_closing: Whether dilated grouping uses closing instead of dilation
        suspicious_area: Minimum area before adhesion suspicious checks
        min_suspicious_cues: Number of shape cues required before adhesion split
        max_split_count: Reject adhesion split results with too many fragments
        min_split_coverage: Reject split results that keep too little original area
        skip_ring_like: Avoid adhesion split for ring-like dilated groups
        
    Returns:
        List of cluster token dictionaries
    """
    tokens = []
    N = len(maps)
    
    for sample_idx in range(N):
        raw_map = maps[sample_idx]
        defect_mask = raw_map == 2
        valid_mask = (raw_map == 1) | (raw_map == 2)
        
        for proposal_type in proposal_types:
            components = cluster_proposal(
                defect_mask,
                valid_mask,
                proposal_type,
                min_area,
                top_k=top_k,
                topk_base_method=topk_base_method,
                dilation_radius=dilation_radius,
                use_closing=use_closing,
                suspicious_area=suspicious_area,
                min_suspicious_cues=min_suspicious_cues,
                max_split_count=max_split_count,
                min_split_coverage=min_split_coverage,
                skip_ring_like=skip_ring_like,
            )
            
            for token_idx, comp in enumerate(components):
                token = {
                    'sample_id': sample_idx,
                    'original_index': int(original_indices[sample_idx]),
                    'proposal_type': proposal_type,
                    'token_id': token_idx,
                }
                # Add component stats except the pixel list (too large for JSON)
                for key, value in comp.items():
                    if key == 'proposal_type':
                        token['component_proposal_type'] = value
                        continue
                    if key != 'pixels':
                        # Convert numpy types to native types for JSON serialization
                        if isinstance(value, np.integer):
                            token[key] = int(value)
                        elif isinstance(value, np.floating):
                            token[key] = float(value)
                        else:
                            token[key] = value
                # Store pixel coordinates in compressed form
                if 'pixels' in comp and comp['pixels']:
                    token['pixel_coords'] = [
                        {'row': int(p[0]), 'col': int(p[1])} 
                        for p in comp['pixels']
                    ]
                
                tokens.append(token)
    
    return tokens


def save_cluster_tokens(tokens: List[Dict], json_path: str, npz_path: Optional[str] = None):
    """
    Save cluster tokens to JSON and NPZ files.
    """
    with open(json_path, 'w') as f:
        # Save line by line for large files (JSONL format)
        for token in tokens:
            f.write(json.dumps(token, ensure_ascii=False) + '\n')
    
    if npz_path is not None:
        # Save structured array to NPZ for fast loading
        sample_ids = np.array([t['sample_id'] for t in tokens])
        proposal_types = np.array([t['proposal_type'] for t in tokens])
        areas = np.array([t['area'] for t in tokens])
        
        np.savez_compressed(
            npz_path,
            sample_ids=sample_ids,
            proposal_types=proposal_types,
            areas=areas,
            tokens=tokens
        )


def compute_cluster_statistics(tokens: List[Dict]) -> Dict:
    """
    Compute statistics about clusters.
    """
    stats = defaultdict(lambda: {
        'count': 0,
        'areas': [],
        'component_counts': defaultdict(int)
    })
    
    for token in tokens:
        pt = token['proposal_type']
        stats[pt]['count'] += 1
        stats[pt]['areas'].append(token['area'])
        stats[pt]['component_counts'][token['sample_id']] += 1
    
    result = {}
    for pt, pt_stats in stats.items():
        areas = np.array(pt_stats['areas'])
        comp_counts = list(pt_stats['component_counts'].values())
        
        result[pt] = {
            'total_clusters': int(pt_stats['count']),
            'area_mean': float(areas.mean()) if len(areas) > 0 else 0,
            'area_std': float(areas.std()) if len(areas) > 0 else 0,
            'area_min': int(areas.min()) if len(areas) > 0 else 0,
            'area_max': int(areas.max()) if len(areas) > 0 else 0,
            'component_count_mean': np.mean(comp_counts) if comp_counts else 0,
            'component_count_std': np.std(comp_counts) if comp_counts else 0,
        }
    
    return result
