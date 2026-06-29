"""Core matching algorithms and data models."""

from .models import DefectTable, GridMaps, BACKGROUND, VALID_NO_DEFECT, VALID_HAS_DEFECT, UNINSPECTED
from .mappers import GridMapper, DieIndexGridMapper, RelativeCoordinateGridMapper, PhysicalCoordinateGridMapper, MAPPERS
from .representations import (
    RepresentationBuilder,
    BinaryMapBuilder,
    CountMapBuilder,
    DensityMapBuilder,
    SoftMapBuilder,
    ThreeValueMapBuilder,
    MountainMapBuilder,
    REPRESENTATIONS,
)
from .pipeline import map_klarf_to_grid
from .similarity import (
    SimilarityMethod,
    SimilarityResult,
    DiceSimilarity,
    IoUSimilarity,
    NccSimilarity,
    CosineSimilarity,
    ChamferSimilarity,
    CoverageSimilarity,
    LeakagePenalty,
    CoverageLeakageScore,
    SIMILARITIES,
    compute_similarity,
    top_k_retrieval,
)
from .local_matching import LocalMatchResult, compute_count_partial_match, explain_count_partial_match
from .classnumber_matching import ClassNumberMatchResult, ClassSplitMatch, compute_classnumber_matches
