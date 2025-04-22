# Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Others
from contextlib import redirect_stdout
from io import StringIO

# My scripts
from movie_recommender.models.svd_train_eval_pred import SVDModelHandler
from movie_recommender.models.gnn_train_eval_pred import GNNModelHandler
from movie_recommender.recommenders.content_based_filtering import ContentBasedFiltering
from movie_recommender.recommenders.collaborative_filtering import CollaborativeFiltering

"""
References:
    - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class HybridFiltering:
    """
    HybridFiltering - Movie Recommendation System based on User-Based Hybrid Filtering.

    This class implements a movie recommendation system based on a hybrid filtering approach.
    It combines collaborative filtering and content-based filtering to provide and improved 
    user-based movie recommendation.

    Attributes:
        __tdh (TabularDatasetHandler): Instance of TabularDatasetHandler.
        __movies_df (pd.DataFrame): A copy of the movies dataframe from the dataset.
        __users_ratings_df (pd.DataFrame): A copy of the users ratings dataframe from the dataset.
        __model (CollaborativeFilteringInterface): The collaborative filtering model (SVD or GNN).
        __gdh (HeterogeneousGraphDatasetHandler): Instance of HeterogeneousGraphDatasetHandler for GNN-based CF.
    """

    def __init__(self, collaborative_filtering: CollaborativeFiltering):
        """
        Initializes the collaborative filtering system.

        Parameters:
            collaborative_filtering (CollaborativeFiltering): The collaborative filtering model.
        """
        self.__tdh = collaborative_filtering._gdh._tdh
        self.__collaborative_filtering = collaborative_filtering

        # Initialize some brand-new copies of the required dataframes from dataset
        self.__movies_df = self.__tdh.get_movies_df_deepcopy()
        self.__users_ratings_df = self.__tdh.get_users_ratings_df_deepcopy()

        # Handle the different possible collaborative filter models and their training
        self.__model = None

        if isinstance(self.__collaborative_filtering._model_handler, SVDModelHandler):
            self.__model = collaborative_filtering._model_handler
        elif isinstance(self.__collaborative_filtering._model_handler, GNNModelHandler):
            self.__model = collaborative_filtering._model_handler

    def suggest_similar_movies(self, user_id, movie_title, top_n):
        """
        Suggests N movies similar to the selected target movie, based on the user's tastes (i.e.,
        on the user's past ratings).

        Parameters:
            user_id (int): The ID of the user.
            movie_title (str): The title of the target movie.
            top_n (int): The number of movie recommendations to return.

        Returns:
            pd.DataFrame: DataFrame containing movie recommendations.
        """
        # Get the movies suggestion of the metadata_based_recommender
        new_movies_df = ContentBasedFiltering.metadata_based_recommender(
            self.__tdh, 
            movie_title, top_n, 
            improved=False
        )

        # Predict the n top most similar movies sorted on the basis of expected ratings by that particular user
        if isinstance(self.__collaborative_filtering._model_handler, SVDModelHandler):
            # Do prediction on 'new_movies_df'
            new_movies_df['est_rating'] = new_movies_df['id'].apply(
                lambda x: self.__model.predict(user_id, x).est
            )
        elif isinstance(self.__collaborative_filtering._model_handler, GNNModelHandler):
            # Do prediction on 'new_movies_df'
            # new_movies_df['est_rating'] = new_movies_df['id'].apply(
            #     lambda x: self.__model.predict(user_id, x)[0]
            # )
            movie_ids = new_movies_df['id'].tolist()
            predicted_ratings = self.__model.predict_batch(user_id, movie_ids, include_gt_ratings=False)
            new_movies_df['est_rating'] = predicted_ratings

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