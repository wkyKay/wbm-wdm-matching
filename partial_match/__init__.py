# -*- coding: utf-8 -*-
"""
MixedWM38K Retrieval Protocol and Cluster Proposal

统一聚类接口:
    from partial_match.core.clustering import cluster
    clusters = cluster(defect_mask, valid_mask, method='dbscan')
"""

from . import core
from . import data
from . import evaluation
from . import utils

from .core.clustering import cluster

__all__ = [
    'core',
    'data',
    'evaluation',
    'utils',
    'cluster',
]
