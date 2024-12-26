import sys
from pathlib import Path

# Add Src to sys.path for external imports (like `from CRM.data import ...`)
module_root = Path(__file__).parent / "Src"
if str(module_root) not in sys.path:
    sys.path.append(str(module_root))

# Internal imports for relative usage
from Src.scripts.data.tabular_dataset_handler import TabularDatasetHandler
from Src.scripts.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler

from Src.scripts.approaches.filters.popularity_rankings import PopularityRanking
from Src.scripts.approaches.filters.content_based_filtering import ContentBasedFiltering
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface
from Src.scripts.approaches.filters.hybrid_filtering import HybridFiltering
from Src.scripts.approaches.collaborative_filters.GNN_based_CF import GNN_Based_CollaborativeFilter
from Src.scripts.approaches.collaborative_filters.SVD_based_CF import SVD_Based_CollaborativeFilter

from Src.scripts.approaches.models import (
    GNNEncoderInterface,
    GCNEncoder,
    GraphSAGEEncoder,
    GATEncoder,
    EdgeDecoder,
    Model
)

# External alias (e.g., `CRM`)
from Src import scripts, trained_models

# Expose imports at the package level for convenience
__all__ = [
    "TabularDatasetHandler",
    "HeterogeneousGraphDatasetHandler"

    "PopularityRanking",
    "ContentBasedFiltering",
    "CollaborativeFilteringInterface",
    "HybridFiltering",
    "GNN_Based_CollaborativeFilter",
    "SVD_Based_CollaborativeFilter",
    
    "GNNEncoderInterface",
    "GCNEncoder",
    "GraphSAGEEncoder",
    "GATEncoder",
    "EdgeDecoder",
    "Model"
]