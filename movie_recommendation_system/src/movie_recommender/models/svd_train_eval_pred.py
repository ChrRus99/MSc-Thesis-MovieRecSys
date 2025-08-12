# Util
import os
from overrides import overrides

# Machine learning
from surprise import Reader, Dataset, SVD, dump
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

# My scripts
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.models import ModelHandlerInterface

"""
References:
    - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class SVDModelHandler(ModelHandlerInterface):
    """
    This class handles training, evaluation, and prediction for a Singular Value Decomposition (SVD)
    model, from the Surprise library, in a movie recommendation system.          

    Attributes:
        __users_ratings_df (pd.DataFrame): A copy of the users ratings Dataframe from the dataset.
        __movies_df (pd.DataFrame): A copy of the movies metadata DataFrame from the dataset.
        __dataset (surprise.dataset.Dataset): The dataset created from users ratings dataframe.
        _model (surprise.prediction_algorithms.matrix_factorization.SVD): SVD model for collaborative filtering.
        __trainset (surprise.Trainset): The training set for training the SVD model.
        __testset (list): The test set for evaluating the SVD model performance.
    """

    def __init__(
        self,
        tabular_dataset_handler: TabularDatasetHandler,
    ):
        """
        Initializes the SVD model handler.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of TabularDatasetHandler.
        """
        super().__init__()

        # Initialize some brand-new copies of the required dataframes from dataset
        self.__users_ratings_df = tabular_dataset_handler.get_users_ratings_df_deepcopy()
        self.__movies_df = tabular_dataset_handler.get_movies_df_deepcopy()

        # Initialize a graph dataset handler (for collaborative filtering)
        self._gdh = HeterogeneousGraphDatasetHandler(tabular_dataset_handler)

        # Create a dataset from the 'users_ratings_df' for training and test the recommender system 
        # models provided by the 'surprise' library
        reader = Reader()
        self.__dataset = Dataset.load_from_df(self.__users_ratings_df[['userId', 'movieId', 'rating']], reader)

        # Initialize the model, the trainset and the testset
        self._model = None
        self.__trainset = None
        self.__testset = None

    @overrides
    def train(self, trained_model_path: str=None):
        """
        Trains the SVD model on the training set and stores the trained model in the specified path.

        Parameters:
            trained_model_path (str): The path where to store the trained models.
        """
        # Initialize the SVD model
        self._model = SVD()

        # Split the dataset into train and test sets
        trainset, testset = train_test_split(self.__dataset, test_size=0.2)

        # Train the SVD model on the train set
        self._model.fit(trainset)

        if trained_model_path:
            # Store the trained model
            model_filename = "SVD_model.pkl"
            trained_model_filename = os.path.join(trained_model_path, model_filename)
            dump.dump(trained_model_filename, algo=self._model)

        # Save the test set for later evaluation
        self.__testset = testset

    @overrides
    def evaluate_performance(self):
        """
        Evaluates the performance of the trained model on the test set using RMSE and MAE metrics.

        Returns:
            rmse (float): Root Mean Squared Error.
            mae (float): Mean Absolute Error.
        """
        # Check whether the model is valid
        #self._model_check()

        # Make predictions on the saved test set
        predictions = self._model.test(self.__testset)

        # Evaluate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)

        return rmse, mae

    def train_with_cross_validation(self, trained_model_path: str):
        """
        Trains the SVD model using cross-validation and stores the trained model.

        Parameters:
            trained_model_path (str): The path where to store the trained models.
        """
        # Initialize the SVD model
        self._model = SVD()

        # Train the SVD model
        self.__trainset = self.__dataset.build_full_trainset()
        self._model.fit(self.__trainset)

        # Store the trained model
        model_filename = "SVD_with_cross_validation_model.pkl"
        trained_model_filename = os.path.join(trained_model_path, model_filename)
        dump.dump(trained_model_filename, algo=self._model)

    def evaluate_performance_with_cross_validation(self):
        """
        Evaluates the performance of the model with cross-validation using RMSE and MAE metrics.

        Returns:
            rmse (float): Mean RMSE across folds.
            mae (float): Mean MAE across folds.
        """
        # Check whether the model is valid
        #self._model_check()

        # Perform cross-validation, for evaluating RMSE and MAE
        results = cross_validate(self._model, self.__dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        # Extract mean RMSE and mean MAE
        rmse = sum(results['test_rmse']) / len(results['test_rmse'])
        mae = sum(results['test_mae']) / len(results['test_mae'])

        return rmse, mae

    @overrides
    def predict(self, user_id, movie_id, gt_rating=None):
        """
        Predicts the rating for a given user and movie.

        Parameters:
            user_id (int): The ID of the user.
            movie_id (int): The ID of the movie.
            gt_rating (float, optional): Ground truth rating (if available).

        Returns:
            prediction (surprise.prediction_algorithms.predictions.Prediction): Prediction result.
        """
        # Check whether the model is valid
        #self._model_check()

        if gt_rating is not None:
            return self._model.predict(user_id, movie_id, gt_rating)
        else:
            return self._model.predict(user_id, movie_id)

    # def _model_check(self):
    #     """ Check whether the SVD model has been trained and is available to be preloaded. """
    #     # If the model has not been trained yet
    #     if self._model == None:
    #         # If a pretrained model exists in memory, load it
    #         if os.path.exists(self.__trained_model_filename):
    #             self._model, self.__algo, self.__trainset = dump.load(self.__trained_model_filename)
    #         else:
    #             raise FileNotFoundError(f"The model file '{self.__trained_model_filename}' does not exist.")