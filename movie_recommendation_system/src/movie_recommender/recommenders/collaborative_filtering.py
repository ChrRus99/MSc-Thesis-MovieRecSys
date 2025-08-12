# Dataset
import pandas as pd

# Pytorch geometric
import torch

# My scripts
from movie_recommender.models.abstract_model_train_eval_pred import ModelHandlerInterface

"""
References:
    - Link Regression on MovieLens:
        https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2?usp=sharing
    - Link Prediction on MovieLens:
        https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing
    - Link Prediction on Heterogeneous Graphs with PyG:
        https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
    - Heterogeneous Graph Learning (Pytorch-geometric):
        https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
    - GRAPH Link Prediction w/ DGL on Pytorch and PyG Code Example | GraphML | GNN:
        https://www.youtube.com/watch?v=wxJ84sMJfUA&ab_channel=code_your_own_AI
    - PyTorch Geometric tutorial: Graph Autoencoders & Variational Graph Autoencoders:
        https://www.youtube.com/watch?v=qA6U4nIK62E&ab_channel=AntonioLonga
"""


class CollaborativeFiltering:
    """
    CollaborativeFiltering - Movie Recommendation System based on User-Based Collaborative Filtering
    using a backbone model (SVD or GNN).

    This class implements a movie recommendation system based on a collaborative filtering approach
    which leverages on any backbone models which implements the ModelHandlerInterface, such as:
        - SVDModelHandler: SVD (Singular Value Decomposition) model from the Surprise library.
        - GNNModelHandler: GNN-based model from PyTorch Geometric library.

    Attributes:
        _gdh (HeterogeneousGraphDatasetHandler): An instance of HeterogeneousGraphDatasetHandler managing the dataset.
        __graph_dataset (HeteroData): The heterogeneous graph dataset.
        _model_handler (ModelHandlerInterface): The model handler for the recommendation system.
        _model (Model): The edge regression model.
        __device (torch.device): The device on which to perform computations (GPU if available, else CPU).
    """      

    def __init__(self, model_handler: ModelHandlerInterface):
        """
        Initializes the collaborative filtering system.

        Parameters:
            model_handler (ModelHandlerInterface): A pre-trained model to be used in the
                recommendation logic which implements the ModelHandlerInterface interface.
        """
        super().__init__()

        # Trained model for movie recommendation
        self._model_handler = model_handler
        self._model = self._model_handler._model

        # Heterogeneous graph dataset
        self._gdh = self._model_handler._gdh
        self.__graph_dataset = self._model_handler._gdh.get_graph_dataset()

        # Enable GPU computation (if available)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, user_id, movie_id):
        """
        Predicts a rating for a given user and movie.

        Parameters:
            user_id (int): The ID of the user.
            movie_id (int): The ID of the movie.

        Returns:
            Tuple containing the predicted rating and the ground truth rating (if available).
        """
        return self._model_handler.predict(user_id, movie_id)

    def predict_ratings(
        self,
        user_id,
        subset_movies_df,
        include_gt_ratings: bool=True,
        use_batch: bool=False
    ):
        """
        Predicts ratings for a user on a subset of movies provided as a DataFrame.

        Parameters:
            user_id (int): ID of the user.
            subset_movies_df (pd.DataFrame): DataFrame containing movie IDs to predict ratings for.
            include_gt_ratings (bool, optional): Whether to include ground truth ratings for seen
                movies. Default is True.
            use_batch (bool, optional): If True, predictions are computed in batch via predict_batch. 
                If False, predictions are computed sequentially. Default is False.

        Returns:
            pd.DataFrame: DataFrame with columns 'movieId', 'predicted_rating' and, if requested, 
                'ground_truth_rating'.
        """
        # Extract movie IDs from the input DataFrame
        movie_ids = subset_movies_df['id'].tolist()

        if use_batch:
            # Use the batched prediction function
            predicted_ratings, ground_truth_ratings = self._model_handler.predict_batch(user_id, movie_ids, include_gt_ratings)
        else:
            # Sequential prediction fallback
            predicted_ratings = []
            ground_truth_ratings = []
            for movie_id in movie_ids:
                pred, gt = self.predict(user_id, movie_id)
                predicted_ratings.append(pred)
                ground_truth_ratings.append(gt)

        # Build the result DataFrame
        results_df = pd.DataFrame({
            'movieId': movie_ids,
            'predicted_rating': predicted_ratings
        })

        if include_gt_ratings:
            results_df['ground_truth_rating'] = ground_truth_ratings

        return results_df

    def suggest_new_movie(self, user_id):
        """
        Suggests a new movie for a given user, based on the history of user's past ratings.

        Parameters:
            user_id (int): The ID of the user.

        Returns:
            Tuple containing the suggested movie and the predicted rating.
        """
        # Upload some (updated) useful data from the graph dataset
        users_ratings_df = self._gdh.users_ratings_df
        movies_df = self._gdh.movies_df.rename(columns={'id': 'movieId'}, inplace=False)
        mapped_users_ids_df = self._gdh.mapped_users_ids_df
        mapped_movies_ids_df = self._gdh.mapped_movies_ids_df

        # Select movies that user has never watched before
        movies_rated = users_ratings_df[users_ratings_df['userId'] == user_id]
        movies_not_rated = movies_df[~movies_df.index.isin(movies_rated['movieId'])]
        movies_not_rated = movies_not_rated.merge(mapped_movies_ids_df, on='movieId')

        # Extract the mapped user id value
        mapped_user_id = mapped_users_ids_df[mapped_users_ids_df['userId'] == user_id]['mappedUserId'].values[0]

        # Select a sample movie from the set of 'movies_not_rated'
        movie = movies_not_rated.sample(1)

        # Create new `edge_label_index` between the user and the movie
        edge_label_index = (
            torch.tensor([mapped_user_id], dtype=torch.long),
            torch.tensor([movie['mappedMovieId'].item()], dtype=torch.long)
        )

        # Move the input data to the device
        data = self.__graph_dataset.to(self.__device)

        # Predict the user_id's rating for that movie
        with torch.no_grad():
            pred = self._model(
                data.x_dict,
                data.edge_index_dict,
                edge_label_index
            )
            pred = pred.clamp(min=0, max=5).detach().cpu().numpy()

        return movie, pred.item()
