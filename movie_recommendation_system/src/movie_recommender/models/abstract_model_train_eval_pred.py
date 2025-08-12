from abc import ABC, abstractmethod
import os


class ModelHandlerInterface(ABC):
    """
    Abstract class to implement a factory of model handlers for Collaborative Filtering.

    This abstract class defines the structure of the handler which performs the training, 
    evaluation, and prediction for the model used by a movie recommendation system based 
    on collaborative filtering.

    Attributes:
        _dataset: The dataset used for collaborative filtering.
        _model: The model used for collaborative filtering.
        _algo: The algorithm used for collaborative filtering.
        _trainset: The training set used for collaborative filtering.
    """
    
    def __init__(self, datset=None, model=None, algo=None, trainset=None):
        """
        Initializes the AbstractModelHandlerInterface with the given dataset, model, algorithm, 
        and training set.

        Parameters:
            datset: The dataset for collaborative filtering.
            model: The model for collaborative filtering.
            algo: The algorithm for collaborative filtering.
            trainset: The training set for collaborative filtering.
        """
        self._dataset = datset
        self._model = model
        self._algo = algo
        self._trainset = trainset

    @abstractmethod
    def train(self):
        """
        Abstract method for training the collaborative filtering model.
        """
        pass

    @abstractmethod
    def evaluate_performance(self):
        """
        Abstract method for evaluating the performance of the collaborative filtering model.

        Returns:
            Tuple: A tuple containing evaluation metrics such as MAE (Mean Absolute Error) 
                   and RMSE (Root Mean Square Error).
        """
        pass

    @abstractmethod
    def predict(self, user_id, movie_id, gt_rating=None):
        """
        Abstract method for making predictions using the collaborative filtering model.

        Parameters:
            user_id: The ID of the user for whom the prediction is to be made.
            movie_id: The ID of the movie for which the prediction is to be made.
            gt_rating (float, optional): Ground truth rating (if available).

        Returns:
            The prediction result, specific to the implementation.
        """
        pass
