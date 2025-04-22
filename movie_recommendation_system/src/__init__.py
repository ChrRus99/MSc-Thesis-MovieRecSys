# Import Dataset Handlers
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.data.expandable_graph_dataset_handler import ExpandableHeterogeneousGraphDatasetHandler

# Import Recommendation Strategies
from movie_recommender.recommenders.popularity_ranking import PopularityRanking
from movie_recommender.recommenders.content_based_filtering import ContentBasedFiltering
from movie_recommender.recommenders.collaborative_filtering import CollaborativeFiltering
from movie_recommender.recommenders.hybrid_filtering import HybridFiltering

# Import GNN Models and Components
from movie_recommender.models.gnn_model import (
    GNNEncoderInterface,
    GCNEncoder,
    GraphSAGEEncoder,
    GATEncoder,
    EdgeRegressionDecoder,
    GNNModel
)
from movie_recommender.models.svd_train_eval_pred import SVDModelHandler
from movie_recommender.models.gnn_train_eval_pred import GNNModelHandler
from movie_recommender.models.gnn_retrain_strategies import GNNRetrainModelHandler


# Define what gets exposed when `import *` is used on the package
__all__ = [
    # Dataset Handlers
    "TabularDatasetHandler",
    "HeterogeneousGraphDatasetHandler",
    "ExpandableHeterogeneousGraphDatasetHandler",

    # Recommendation Strategies
    "PopularityRanking",
    "ContentBasedFiltering",
    "CollaborativeFiltering",
    "HybridFiltering",

    # SVD Models and Components
    "SVDModelHandler",

    # GNN Models and Components
    "GNNEncoderInterface",
    "GCNEncoder",
    "GraphSAGEEncoder",
    "GATEncoder",
    "EdgeRegressionDecoder",
    "GNNModel",
    "GNNModelHandler",
    "GNNRetrainModelHandler",
]
