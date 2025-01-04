# Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Others
from contextlib import redirect_stdout
from io import StringIO

# My scripts
from Src.scripts.data.tabular_dataset_handler import TabularDatasetHandler
from Src.scripts.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from Src.scripts.approaches.filters.content_based_filtering import ContentBasedFiltering
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface
from Src.scripts.approaches.collaborative_filters.SVD_based_CF import SVD_Based_CollaborativeFilter
from Src.scripts.approaches.collaborative_filters.GNN_based_CF import GNN_Based_CollaborativeFilter

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class HybridFiltering:
    """
        HybridFiltering - A Movie Recommendation System using Hybrid Filtering approach

        This class implements a movie recommendation system based on a hybrid filtering approach.
        It combines collaborative filtering using collaborative filtering models and content-based 
        filtering to provide personalized movie recommendations for users.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of TabularDatasetHandler.
            model (CollaborativeFilteringInterface): The collaborative filtering model.

        Attributes:
            __tdh (TabularDatasetHandler): Instance of TabularDatasetHandler.
            __movies_df (pd.DataFrame): A copy of the movies dataframe from the dataset.
            __users_ratings_df (pd.DataFrame): A copy of the users ratings dataframe from the dataset.
            __model (CollaborativeFilteringInterface): The collaborative filtering model (SVD or GNN).
            __gdh (HeterogeneousGraphDatasetHandler): Instance of HeterogeneousGraphDatasetHandler for GNN-based CF.
    """

    def __init__(self, tabular_dataset_handler: TabularDatasetHandler, model: CollaborativeFilteringInterface):
        self.__tdh = tabular_dataset_handler

        # Initialize some brand-new copies of the required dataframes from dataset
        self.__movies_df = tabular_dataset_handler.get_movies_df_deepcopy()
        self.__users_ratings_df = tabular_dataset_handler.get_users_ratings_df_deepcopy()

        # Handle the different possible collaborative filter models and their training
        self.__model = None
        self.__gdh = None

        if isinstance(model, SVD_Based_CollaborativeFilter):
            self.__model = model
            self.__model.train()
        elif isinstance(model, GNN_Based_CollaborativeFilter):
            self.__model = model
            # Suppress prints
            with StringIO() as fake_stdout:
                with redirect_stdout(fake_stdout):
                    self.__model.train()
            self.__gdh = HeterogeneousGraphDatasetHandler(tabular_dataset_handler)

    def predict(self, user_id, movie_title, N):
        """
            Suggests the target user N movies similar to the selected target movie, based on his/her
            tastes (i.e., on user's past ratings).

            The predictions are based on a hybrid approach combining collaborative and content-based
            filtering.

            Parameters:
                user_id (int): The ID of the user.
                movie_title (str): The title of the target movie.
                N (int): The number of movie recommendations to return.

            Returns:
                pd.DataFrame: DataFrame containing movie recommendations.
        """
        # Get the movies suggestion of the metadata_based_recommender
        new_movies_df = ContentBasedFiltering.metadata_based_recommender(
            self.__tdh, 
            movie_title, N, 
            improved=False
        )

        # Predict the N top most similar movies sorted on the basis of expected ratings by that particular user
        if isinstance(self.__model, SVD_Based_CollaborativeFilter):
            # Do prediction on 'new_movies_df'
            new_movies_df['est_rating'] = new_movies_df['id'].apply(
                lambda x: self.__model.predict(user_id, x).est
            )
        elif isinstance(self.__model, GNN_Based_CollaborativeFilter):
            # Do prediction on 'new_movies_df'
            new_movies_df['est_rating'] = new_movies_df['id'].apply(
                lambda x: self.__model.predict(user_id, x)[0]
            )
        new_movies_df = new_movies_df.sort_values('est_rating', ascending=False)

        # Add ground truth ratings to the output
        new_movies_df['gt_rating'] = new_movies_df['id'].apply(
            lambda x: self.__users_ratings_df[
                (self.__users_ratings_df['userId'] == user_id) &
                (self.__users_ratings_df['movieId'] == x)
            ]['rating'].values[0]
            if ((user_id in self.__users_ratings_df['userId'].values) and
                (x in self.__users_ratings_df['movieId'].values) and
                (len(self.__users_ratings_df[
                     (self.__users_ratings_df['userId'] == user_id) &
                     (self.__users_ratings_df['movieId'] == x)
                ]) > 0))
            else np.nan
        )

        return new_movies_df