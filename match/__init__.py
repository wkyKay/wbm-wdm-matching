"""WBM-WDM matching utilities.

Modules:
  core     – models, mappers, representations, matching algorithms
  data     – KLARF/WBM file I/O
  viz      – comparison and count-partial visualizations
  scripts  – CLI entry points
"""

from .core.models import DefectTable, GridMaps, BACKGROUND, VALID_NO_DEFECT, VALID_HAS_DEFECT, UNINSPECTED
from .data.fileio import read_wbm_shape, read_wbm_png, load_defect_tables, load_die_pitch, save_grid_maps
from .core.mappers import (
    GridMapper,
    DieIndexGridMapper,
    RelativeCoordinateGridMapper,
    PhysicalCoordinateGridMapper,
    MAPPERS,
)
from .core.representations import (
    RepresentationBuilder,
    BinaryMapBuilder,
    CountMapBuilder,
    DensityMapBuilder,
    SoftMapBuilder,
    ThreeValueMapBuilder,
    MountainMapBuilder,
    REPRESENTATIONS,
)
from .core.pipeline import map_klarf_to_grid
from .core.similarity import (
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
from .core.local_matching import LocalMatchResult, compute_count_partial_match, explain_count_partial_match
