import sys
from pathlib import Path

# Internal imports for relative usage
from Src.scripts.data.tabular_dataset_handler import TabularDatasetHandler
from Src.scripts.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler

from Src.scripts.approaches.filters.popularity_rankings import PopularityRanking
from Src.scripts.approaches.filters.content_based_filtering import ContentBasedFiltering
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface
from Src.scripts.approaches.filters.hybrid_filtering import HybridFiltering
from Src.scripts.approaches.collaborative_filters.GNN_based_CF import GNN_Based_CollaborativeFilter
from Src.scripts.approaches.collaborative_filters.SVD_based_CF import SVD_Based_CollaborativeFilter

from Src.scripts.approaches.models.GNN_regression_model import (
    GNNEncoderInterface,
    GCNEncoder,
    GraphSAGEEncoder,
    GATEncoder,
    EdgeDecoder,
    Model
)

# Expose imports at the package level for convenience
__all__ = [
    "TabularDatasetHandler",
    "HeterogeneousGraphDatasetHandler",

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
    "Model",
]