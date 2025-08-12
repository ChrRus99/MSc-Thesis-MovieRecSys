# Dataset
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Util
import os
import pickle
import random
import numpy as np
from typing import Tuple

# Pytorch Geometric
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Machine learning
from sentence_transformers import SentenceTransformer

# My scripts
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.utils import generate_users_external_features

"""
References:
    - Link Prediction on Heterogeneous Graphs with PyG:
        https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
    - Converting Tabular Dataset(CSV file) to Graph Dataset with Pytorch Geometric:
        https://medium.com/@tejpal.abhyuday/converting-tabular-dataset-csv-file-to-graph-dataset-with-pytorch-geometric-b3aea6b64100
    - Heterogeneous graph learning [Advanced PyTorch Geometric Tutorial 4]:
        https://www.youtube.com/watch?v=qL09oshDKww&ab_channel=AntonioLonga
    - Pytorch.org - DATASETS & DATALOADERS:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - Heterogeneous Graph Learning (Pytorch-geometric):
        https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
    - Python Classes: The Power of Object-Oriented Programming:
        https://realpython.com/python-classes/
"""


class HeterogeneousGraphDatasetHandler:
    """
    This class handles the creation of a heterogeneous graph dataset for edge regression tasks,
    starting from MovieLens tabular datasets containing users ratings and movies information.

    Heterogeneous Graph Data structure:
        - Nodes: Users and Movies
        - Edges: if a user has rated a movie
        - Node features:
            - Movie: graph topological features + external features from movies_df tabular dataset data
            - User: graph topological features
        - Labels: the ratings (edges weights)
        - Target supervised task: edge regression (i.e., prediction of user-movie ratings)

    Attributes:
        _tdh (PreprocessedTDH): Preprocessed Tabular Data Handler containing users ratings and movies information.
        _movies_df (pd.DataFrame): DataFrame containing movies information.
        _users_ratings_df (pd.DataFrame): DataFrame containing users ratings for movies.
        _data (HeteroData): The heterogeneous graph dataset created by this class starting from the tabular dataset.
        _X_movies (torch.Tensor): Tensor containing movie nodes features.
        _X_users (torch.Tensor): Tensor containing user nodes features.
        _y_ratings (torch.Tensor): Tensor containing edge labels (ratings).
        _user_to_movie_edge_indices (torch.Tensor): Edge indices representing user-movie interactions.
        _mapped_users_ids_df (pd.DataFrame): DataFrame mapping original user IDs to consecutive integers.
        _mapped_movies_ids_df (pd.DataFrame): DataFrame mapping original movie IDs to consecutive integers.
    """

    def __init__(self, preprocessed_tdh: TabularDatasetHandler):
        # Tabular datasets
        self._tdh = preprocessed_tdh
        self._movies_df = self._tdh.get_movies_df_deepcopy()
        self._users_ratings_df = self._tdh.get_valid_ratings() #self._tdh.get_users_ratings_df_deepcopy()

        # Initialize a new heterogeneous graph dataset
        self._data = HeteroData()       

        # Heterogeneous graph dataset nodes and edges data
        self._X_movies = None
        self._X_users = None
        self._y_ratings = None
        self._user_to_movie_edge_indices = None

        # Mapped nodes ids
        self._mapped_users_ids_df = None
        self._mapped_movies_ids_df = None

        # Nodes external features
        self._unique_genres = set()

    @property
    def users_ratings_df(self):
        return self._users_ratings_df

    @property
    def movies_df(self):
        return self._movies_df

    @property
    def mapped_users_ids_df(self):
        return self._mapped_users_ids_df

    @property
    def mapped_movies_ids_df(self):
        return self._mapped_movies_ids_df

    def build_graph_dataset(self):
        """
        Builds the heterogeneous graph dataset with user and movie nodes indices and features and
        with edges labels and indices.

        Returns:
            None
        """
        ## RESET THE HETEROGENEOUS GRAPH DATASET
        # Note: this is useful to re-build the graph dataset from scratch each time this function is called
        self._reset_graph_dataset()

        ## BUILD THE HETEROGENEOUS GRAPH DATASET
        # Build and store movie nodes features
        movie_features = self._build_movie_nodes_features(self._movies_df)
        self._X_movies = movie_features  # X_movies.shape = [num_movie_nodes x movie_node_feature_dim

        # Build and store user nodes features
        user_features = self._build_user_nodes_features(self._users_ratings_df)
        self._X_users = user_features  # X_users.shape = [num_users_nodes x user_node_feature_dim]

        # Build and store mapped user and movie IDs
        mapped_user_ids_df, mapped_movies_ids_df = self._build_user_and_movie_nodes_indices(self._users_ratings_df, self._movies_df)
        self._mapped_users_ids_df = mapped_user_ids_df
        self._mapped_movies_ids_df = mapped_movies_ids_df

        # Build and store user-movie edges labels and indices
        edge_indices, edge_labels = self._build_edges_indices_and_labels(self._users_ratings_df)
        self._user_to_movie_edge_indices = edge_indices  # _user_to_movie_edge_indices.shape = [2, num_ratings]
        self._y_ratings = edge_labels  # y_ratings.shape = [num_ratings] = [tot_num_of_edges]

        ## STORE THE HETEROGENEOUS GRAPH DATASET DATA
        # Add mapped node indices in the heterogeneous graph dataset
        self._data['user'].node_id = torch.tensor(self._mapped_users_ids_df['mappedUserId'].to_list())
        self._data['movie'].node_id = torch.tensor(self._mapped_movies_ids_df['mappedMovieId'].to_list())

        # Add nodes features in the heterogeneous graph dataset
        self._data['user'].x = self._X_users  # dim = [num_users, num_features_users]
        self._data['movie'].x = self._X_movies  # dim = [num_movies, num_features_movies]

        # Add edge indices in the heterogeneous graph dataset
        self._data['user', 'rating', 'movie'].edge_index = self._user_to_movie_edge_indices  # dim = [2, num_ratings]
        # edge_type = [source='user', type='rating', destination='movie'] is the user-movie edge

        # Add edge labels in the heterogeneous graph dataset
        self._data['user', 'rating', 'movie'].edge_label = self._y_ratings
        self._data['user', 'movie'].y = self._y_ratings

        # Make the heterogeneous graph dataset bidirectional by adding reverse edges from movies to users.
        # In this a way the GNN will be able to pass messages in both directions.
        self._data = T.ToUndirected()(self._data)
        # edge_type = [source='movie', type='rev_rating', destination='user'] is the movie-user edge

        # out = [model(data.X_dict, data.edge_index_dict)]

        """
        # Debugging print mappings
        print("Mapped Users IDs:")
        print(self._mapped_users_ids_df.head())

        print("Mapped Movies IDs:")
        print(self._mapped_movies_ids_df.head())

        print("Mapped Users in Graph Dataset:")
        print(self._data["user"].node_id)

        print("Mapped Movies in Graph Dataset:")
        print(self._data["movie"].node_id)

        # Debugging print edges
        print("Edge Labels (y_ratings):", self._y_ratings)
        print("Edge Index (user-movie):", self._user_to_movie_edge_indices)
        """

    def get_graph_dataset(self):
        """
        Returns the constructed heterogeneous graph dataset.

        Returns:
            HeteroData: Constructed graph dataset.
        """
        return self._data

    def save_graph_dataset(self, filepath: str):
        """
        Saves the constructed heterogeneous graph dataset to a .pth file.

        Parameters:
            filepath (str): Path to the .pkl file where the constructed heterogeneous graph dataset
                will be saved.

        Returns:
            None
        """
        # Save the dataset to disk
        torch.save(self._data, filepath)

        print(f"Graph dataset saved to {filepath}")

    def load_graph_dataset(self, filepath: str):
        """
        Loads the heterogeneous graph dataset from a .pth file.

        Parameters:
            filepath (str): Path to the .pkl file from which the the constructed heterogeneous graph
                dataset will be loaded.

        Returns:
            HeteroData: Loaded graph dataset.
        """
        # Load the dataset from disk
        self._data = torch.load(filepath)
        return self._data

    def store_class_instance(self, filepath: str):
        """
        Store the entire class instance to a .pkl file.

        Parameters:
            filepath (str): Path to the .pkl file where the instance will be saved.

        Returns:
            None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Class instance saved to {filepath}")

    @staticmethod
    def load_class_instance(filepath: str):
        """
        Loads a class instance from a .pkl file.

        Parameters:
            filepath (str): Path to the .pkl file from which the instance will be loaded.

        Returns:
            HeterogeneousGraphDatasetHandler: The loaded class instance.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def plot_graph_dataset(self, node_sample_ratio: float=1.0, with_labels: bool=True):
        """
        Plots the sampled heterogeneous graph dataset.

        Parameters:
            node_sample_ratio (float, optional): Ratio of nodes to sample. Default is 1.
            with_labels (bool, optional): Whether to include node labels. Default is True.

        Returns:
            None
        """
        # Create a NetworkX graph
        G = nx.MultiDiGraph()

        # Node and edge colors
        node_colors = {'user': 'lightgreen', 'movie': 'lightblue'}
        edge_color = 'red'

        # Sample nodes
        sampled_user_nodes = random.sample(range(len(self._X_users)), int(node_sample_ratio * len(self._X_users)))
        sampled_movie_nodes = random.sample(range(len(self._X_movies)), int(node_sample_ratio * len(self._X_movies)))

        # Add sampled nodes with their features and colors
        for node_type in self._data.node_types:
            for node_idx, node_feature in enumerate(self._data[node_type].x):
                if node_type == 'user' and node_idx not in sampled_user_nodes:
                    continue
                elif node_type == 'movie' and node_idx not in sampled_movie_nodes:
                    continue

                G.add_node(f"{node_type}_{node_idx}", features=node_feature.numpy(), color=node_colors[node_type])

        # Add edges with their labels and color
        for edge_type in self._data.edge_types:
            if hasattr(self._data[edge_type], 'y'):
                edge_index = self._data[edge_type].edge_index
                labels = self._data[edge_type].y

                for src, dst, label in zip(edge_index[0], edge_index[1], labels):
                    if f"user_{src}" in G.nodes and f"movie_{dst}" in G.nodes:
                        G.add_edge(f"user_{src}", f"movie_{dst}", label=label, color=edge_color)

        # Draw the graph
        pos = nx.spring_layout(G)
        # pos = nx.kamada_kawai_layout(G)

        # Extract node colors
        node_colors = [G.nodes[node]['color'] for node in G.nodes]

        # Extract edge colors
        edge_colors = [G.edges[edge]['color'] for edge in G.edges]

        nx.draw(G, pos, with_labels=with_labels, font_weight='bold', node_color=node_colors, edge_color=edge_colors)
        plt.show()

    def _build_movie_nodes_features(self, movies_df: pd.DataFrame) -> torch.Tensor:
        """
        Constructs movie node features from tabular movies_df.

        Parameters:
            movies_df (pd.DataFrame): Movies Dataframe.

        Returns:
            torch.Tensor: Movie nodes features.
        """
        ## BUILD MOVIE TITLE EXTERNAL FEATURES [FIXED-SIZE VECTORS]
        # Encode the movie titles into fixed-size vectors using the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        with torch.no_grad():
            titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=False)
            titles = titles.cpu()

        ## BUILD MOVIE COLLECTION EXTERNAL FEATURES [ONE-HOT ENCODING] (deprecated, to update!)
        # Encode the collection each movie belongs to a unique number
        # collections = movies_df['belongs_to_collection'].map(
        #     {collection: index for index, collection in enumerate(movies_df['belongs_to_collection'].unique())}
        # ).fillna(0).astype(float).tolist()  # int

        # Convert to PyTorch tensor
        #collections = torch.from_numpy(np.array(collections)).view(-1, 1)

        ## BUILD MOVIE PRODUCTION COMPANY EXTERNAL FEATURES [ONE-HOT ENCODING] (deprecated, to update!)
        # One-hot encode and merge the production companies of each movie
        # production_companies = (movies_df['production_companies']
        #     .apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
        #     .str.get_dummies('|')
        #     .values
        # )

        # Convert to PyTorch tensor
        #production_companies = torch.from_numpy(production_companies).to(torch.float)  # int

        ## BUILD MOVIE GENRES EXTERNAL FEATURES [ONE-HOT ENCODING]
        # Extract genres from the new dataframe
        new_genres = set()
        movies_df['genres'] = movies_df['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
        for genre_string in movies_df['genres'].dropna():
            new_genres.update(genre_string.split('|'))
        
        # Update the global genre list to store new (unique) genres
        self._unique_genres.update(new_genres)
        
        # Ensure all genres seen so far are used in one-hot encoding
        genre_columns = sorted(self._unique_genres)  # Maintain consistent ordering
        genre_df = movies_df['genres'].str.get_dummies(sep='|')
        
        # Add missing columns to align with unique_genres
        for genre in genre_columns:
            if genre not in genre_df:
                genre_df[genre] = 0  # Add missing genre columns with zero values
        
        # Reorder columns to match unique_genres
        genre_df = genre_df[genre_columns]

         # Convert to PyTorch tensor
        genres = torch.from_numpy(genre_df.values).to(torch.int)  # int

        # # One-hot encode and merge the genres of each movie
        # genres = (movies_df['genres']
        #     .apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
        #     .str.get_dummies('|')
        #     .values
        # )

        # # Convert to PyTorch tensor
        # genres = torch.from_numpy(genres).to(torch.int)  # int

        ## CONCATENATE MOVIE EXTERNAL FEATURES
        # Build the external features of the movie node by concatenating: genres and titles features
        movie_features = torch.cat([titles, genres], dim=-1)
        # Other tested solutions:
        # movie_features = torch.cat([titles, collections, genres], dim=-1)
        # movie_features = torch.cat([titles, genres, production_companies], dim=-1)
        # movie_features = torch.cat([titles, collections, genres, production_companies], dim=-1)       

        return movie_features

    # NOTE: This approach is the one which provides the best performance, but does not support 
    # re-training, due to variable size of feature vectors.
    # def _build_user_nodes_features(self, users_ratings_df: pd.DataFrame) -> torch.Tensor:
    #     """
    #     Constructs user node features (identity matrix, since we do not have users' external 
    #     features).

    #     Parameters:
    #         users_ratings_df (pd.DataFrame): User ratings DataFrame.

    #     Returns:
    #         torch.Tensor: User nodes features.
    #     """
    #     # We don't have user features, which is why we use an identity matrix
    #     user_features = torch.eye(len(self._users_ratings_df['userId'].unique()))

    #     return user_features

    def _build_user_nodes_features(self, users_ratings_df: pd.DataFrame) -> torch.Tensor:
        """
        Constructs user node features.

        Parameters:
            users_ratings_df (pd.DataFrame): User ratings DataFrame.

        Returns:
            torch.Tensor: User nodes features.
        """
        # Build fixed-size user feature vectors using a predefined random projection matrix
        user_features = generate_users_external_features(
            users_ratings_df=users_ratings_df,
            #max_users=10000,
            feature_dim=600,
            sparse_vector=True,
            sparsity=0.01,
        )

        return user_features

    def _build_user_and_movie_nodes_indices(
            self,
            users_ratings_df: pd.DataFrame,
            movies_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructs mappings for user and movie nodes, assigning a unique index to each user to each
        movie.

        This function ensures that all users and movies from the input datasets are correctly mapped
        to unique IDs in the graph. The user IDs are derived from the `users_ratings_df`, while the
        movie IDs are derived from `movies_df`.

        Parameters:
            users_ratings_df (pd.DataFrame): User ratings DataFrame.
            movies_df (pd.DataFrame): Movies Dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the mapped user and movie IDs.
        """
        ## GENERATE UNIQUE (USER AND MOVIE) NODE INDICES
        # Check whether at least one of the DataFrames is not None
        if users_ratings_df is None and movies_df is None:
            raise ValueError("Both users_ratings_df and movies_df cannot be None.")
        
        mapped_movies_ids_df = mapped_user_ids_df = None

        # Create a mapping from the userId to a unique consecutive value in the range [0, num_users]
        if users_ratings_df is not None:
            last_user_index = self._mapped_users_ids_df['mappedUserId'].max() + 1 if self._mapped_users_ids_df is not None else 0
            users_ids = users_ratings_df['userId'].unique()
            mapped_user_ids_df = pd.DataFrame(data={
                'userId': users_ids,
                'mappedUserId': pd.RangeIndex(last_user_index, last_user_index + len(users_ids))
            })

        # Create a mapping from the movieId to a unique consecutive value in the range [0, num_movies]
        if movies_df is not None:
            last_movie_index = self._mapped_movies_ids_df['mappedMovieId'].max() + 1 if self._mapped_movies_ids_df is not None else 0
            movies_ids = movies_df['id'].unique()
            mapped_movies_ids_df = pd.DataFrame(data={
                'movieId': movies_ids,
                'mappedMovieId': pd.RangeIndex(last_movie_index, last_movie_index + len(movies_ids))
            })

        return (mapped_user_ids_df, mapped_movies_ids_df)
    
    def _build_edges_indices_and_labels(
            self,
            users_ratings_df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs and indices edges between user and movie nodes. Each edge has a user rating as
        label.

        This function filters out edges corresponding to rated movies which are not present in 
        `movies_df`. This ensures that all movie nodes have valid features and prevents featureless
        nodes from being included in the graph.

        Notes: 
            - Not all rated movies are used as edges in the graph, only the ones present in 
              `movies_df`.
            - Not all movies in `movies_df` are used as have a corresponding edge in the graph, only
              the ones rated by at least one user in `users_ratings_df`.
            - Including only nodes with valid features is meant to improve the learning process of 
              the GNN, as nodes without features can introduce noise and reduce model performance.

        Parameters:
            users_ratings_df (pd.DataFrame): User ratings DataFrame.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices and edge labels.
        """
        ## CONNECT HETEROGENEOUS NODES THROUGH EDGES
        # Merge (user and movies unique indices) mappings in the users_ratings_df
        users_ratings_df = users_ratings_df.merge(self._mapped_users_ids_df, on='userId').merge(self._mapped_movies_ids_df, on='movieId')
        
        # Create the edge_index representation in COO format, following the PyG semantics
        edge_indices = torch.stack([
            torch.tensor(users_ratings_df['mappedUserId'].values),
            torch.tensor(users_ratings_df['mappedMovieId'].values)
        ], dim=0)

        ## BUILD EDGE LABELS (= RATINGS OF RATED MOVIES)
        # Build edge labels: users ratings (which are the edges features)
        edge_labels = torch.from_numpy(users_ratings_df['rating'].values).to(torch.float)

        return (edge_indices, edge_labels)

    def _reset_graph_dataset(self):
        """
        Resets all internal attributes to their initial state.
        """
        self._data = HeteroData()
        
        self._X_movies = None
        self._X_users = None
        self._y_ratings = None
        self._user_to_movie_edge_indices = None

        self._mapped_users_ids_df = None
        self._mapped_movies_ids_df = None
