# movie_recommender/models/__init__.py

from .gnn_model import (
    GNNEncoderInterface,
    GCNEncoder,
    GraphSAGEEncoder,
    GATEncoder,
    EdgeRegressionDecoder,
    GNNModel
)
from .abstract_model_train_eval_pred import ModelHandlerInterface
from .svd_train_eval_pred import SVDModelHandler
from .gnn_train_eval_pred import GNNModelHandler
from .gnn_retrain_strategies import GNNRetrainModelHandler

__all__ = [
    "GNNEncoderInterface", "GCNEncoder", "GraphSAGEEncoder", "GATEncoder", "EdgeRegressionDecoder", "GNNModel",
    "ModelHandlerInterface", "SVDModelHandler", "GNNModelHandler", "GNNRetrainModelHandler",
]
