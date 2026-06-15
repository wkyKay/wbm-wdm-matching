"""WBM-WDM matching utilities.

Modules:
  models          – DefectTable, GridMaps, status constants
  io              – KLARF/WBM file I/O
  mappers         – coordinate mapping strategies (MAPPERS)
  representations – grid map representation builders (REPRESENTATIONS)
  pipeline        – map_klarf_to_grid end-to-end entry point
  main            – CLI entry point
"""

from .models import DefectTable, GridMaps, BACKGROUND, VALID_NO_DEFECT, VALID_HAS_DEFECT, UNINSPECTED
from .io import read_wbm_shape, load_defect_tables, load_die_pitch, save_grid_maps
from .mappers import (
    GridMapper,
    DieIndexGridMapper,
    RelativeCoordinateGridMapper,
    PhysicalCoordinateGridMapper,
    MAPPERS,
)
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
