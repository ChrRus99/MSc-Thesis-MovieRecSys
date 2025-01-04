# Dataset
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Util
import os
import pickle
import random
import numpy as np

# Pytorch Geometric
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Machine learning
from sentence_transformers import SentenceTransformer

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
        Heterogeneous Graph Dataset.

        This class handles the creation of a heterogeneous graph dataset for edge regression tasks,
        focusing on user-movie interactions.

        Heterogeneous Graph Data structure:
            - Nodes: Users and Movies
            - Edges: if a user has rated a movie
            - Node features:
                - Movie: graph topological features + external features from movies_df tabular dataset data
                - User: graph topological features
            - Labels: the ratings (edges weights)
            - Target supervised task: edge regression (i.e., prediction of user-movie ratings)

        Attributes:
            __tdh (PreprocessedTDH): Preprocessed Tabular Data Handler containing users ratings and movies information.
            __movies_df (pd.DataFrame): DataFrame containing movies information.
            __users_ratings_df (pd.DataFrame): DataFrame containing users ratings for movies.
            __unique_users_ids (pd.DataFrame): DataFrame mapping original user IDs to consecutive integers.
            __unique_movies_ids (pd.DataFrame): DataFrame mapping original movie IDs to consecutive integers.
            __X_movies (torch.Tensor): Tensor containing movie nodes features.
            __X_users (torch.Tensor): Tensor containing user nodes features.
            __y_ratings (torch.Tensor): Tensor containing edge labels (ratings).
            __edge_index_user_to_movie (torch.Tensor): Edge indices representing user-movie interactions.
            __data (HeteroData): The heterogeneous graph dataset created by this class starting from the tabular dataset.
    """

    def __init__(self, preprocessed_tdh):
        # Tabular datasets
        self.__tdh = preprocessed_tdh
        self.__movies_df = self.__tdh.get_movies_df_deepcopy()
        self.__users_ratings_df = self.__tdh.get_users_ratings_df_deepcopy()

        self.__unique_users_ids = None
        self.__unique_movies_ids = None

        # Initialize a new heterogeneous graph dataset
        self.__data = HeteroData()

        # Graph features of the heterogeneous graph dataset
        self.__X_movies = None
        self.__X_users = None
        self.__y_ratings = None
        self.__edge_index_user_to_movie = None

    @property
    def users_ratings_df(self):
        return self.__users_ratings_df

    @property
    def movies_df(self):
        return self.__movies_df

    @property
    def unique_users_ids(self):
        return self.__unique_users_ids

    @property
    def unique_movies_ids(self):
        return self.__unique_movies_ids

    # TODO: da sistemare
    def add_users_ratings_data(self, new_ratings_df: pd.DataFrame):
        """
            Adds new user ratings data to the existing tabular dataset.

            Parameters:
                new_ratings_df (pd.DataFrame): New user ratings DataFrame to be added.

            Returns:
                None
        """
        self.__users_ratings_df = pd.concat([self.__users_ratings_df, pd.DataFrame.from_records(new_ratings_df)])

    def build_graph_dataset(self):
        """
            Builds the heterogeneous graph dataset with user and movie nodes features and with edges
            labels.

            Returns:
                None
        """
        # Build the heterogeneous graph dataset: nodes features and indixes, edges labels and indixes
        self.__build_movie_nodes_features()
        self.__build_user_nodes_features()
        self.__build_user_and_movie_nodes_indices()
        self.__build_edges_indices_and_labels()

        # Save node indices 
        self.__data['user'].node_id = torch.tensor(self.__unique_users_ids['mappedUserId'].to_list())
        self.__data['movie'].node_id = torch.tensor(self.__unique_movies_ids['mappedMovieId'].to_list())

        # Add the nodes features to the heterogeneous graph dataset
        self.__data['user'].x = self.__X_users  # dim = [num_users, num_features_users]
        self.__data['movie'].x = self.__X_movies  # dim = [num_movies, num_features_movies]

        # Add edge indices to the heterogeneous graph dataset
        self.__data['user', 'rating', 'movie'].edge_index = self.__edge_index_user_to_movie  # dim = [2, num_ratings]
        # edge_type = [source='user', type='rating', destination='movie'] is the user-movie edge

        # Add the edge rating labels to the heterogeneous graph dataset
        self.__data['user', 'rating', 'movie'].edge_label = self.__y_ratings
        self.__data['user', 'movie'].y = self.__y_ratings

        # Add the reverse edges from movies to users the heterogeneous graph dataset.
        # In this a way the GNN will be able to pass messages in both directions.
        self.__data = T.ToUndirected()(self.__data)
        # edge_type = [source='movie', type='rev_rating', destination='user'] is the movie-user edge

        # Remove the reverse edges from the heterogeneous graph dataset
        #del self.__data['movie', 'rev_rating', 'user'].edge_label

        # out = [model(data.X_dict, data.edge_index_dict)]

        """
        # Debugging print mappings
        print("Mapped Users IDs:")
        print(self.__unique_users_ids.head())

        print("Mapped Movies IDs:")
        print(self.__unique_movies_ids.head())

        print("Mapped Users in Graph Dataset:")
        print(self.__data["user"].node_id)

        print("Mapped Movies in Graph Dataset:")
        print(self.__data["movie"].node_id)
        """

    def get_graph_dataset(self):
        """
            Returns the constructed heterogeneous graph dataset.

            Returns:
                HeteroData: Constructed graph dataset.
        """
        return self.__data

    def save_graph_dataset(self, filepath):
        """
            Saves the constructed heterogeneous graph dataset to a .pth file.

            Parameters:
                filepath (str): Path to the .pkl file where the constructed heterogeneous graph 
                dataset will be saved.

            Returns:
                None
        """
        # Save the dataset to disk
        torch.save(self.__data, filepath)

    def load_graph_dataset(self, filepath):
        """
            Loads the heterogeneous graph dataset from a .pth file.

            Parameters:
                filepath (str): Path to the .pth file from which the the constructed heterogeneous 
                graph dataset will be loaded.

            Returns:
                HeteroData: Loaded graph dataset.
        """
        # Load the dataset from disk
        self.__data = torch.load(filepath)
        return self.__data

    def store_class_instance(self, filepath):
        """
            Store the entire class instance to a .pkl file.

            Parameters:
                filepath (str): Path to the .pkl file where the instance will be saved.

            Returns:
                None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_class_instance(filepath):
        """
            Loads a class instance from a .pkl file.

            Parameters:
                filepath (str): Path to the .pkl file from which the instance will be loaded.

            Returns:
                HeterogeneousGraphDatasetHandler: The loaded class instance.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def plot_graph_dataset(self, node_sample_ratio=1, with_labels=True):
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
        sampled_user_nodes = random.sample(range(len(self.__X_users)), int(node_sample_ratio * len(self.__X_users)))
        sampled_movie_nodes = random.sample(range(len(self.__X_movies)), int(node_sample_ratio * len(self.__X_movies)))

        # Add sampled nodes with their features and colors
        for node_type in self.__data.node_types:
            for node_idx, node_feature in enumerate(self.__data[node_type].x):
                if node_type == 'user' and node_idx not in sampled_user_nodes:
                    continue
                elif node_type == 'movie' and node_idx not in sampled_movie_nodes:
                    continue

                G.add_node(f"{node_type}_{node_idx}", features=node_feature.numpy(), color=node_colors[node_type])

        # Add edges with their labels and color
        for edge_type in self.__data.edge_types:
            if hasattr(self.__data[edge_type], 'y'):
                edge_index = self.__data[edge_type].edge_index
                labels = self.__data[edge_type].y

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

    def __build_movie_nodes_features(self):
        """
            Constructs movie node features from tabular movies_df.

            Returns:
                None
        """
        ## PROCESS EXTERNAL FEATURES
        # Encode the movie titles into fixed-size vectors using the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with torch.no_grad():
            titles = model.encode(self.__movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=False)
            titles = titles.cpu()

        # Encode the collection each movie belongs to a unique number
        collections = self.__movies_df['belongs_to_collection'].map(
            {collection: index for index, collection in enumerate(self.__movies_df['belongs_to_collection'].unique())}
        ).fillna(0).astype(int).tolist()

        # Convert to PyTorch tensor
        collections = torch.from_numpy(np.array(collections)).view(-1, 1)

        # One-hot encode and merge the genres of each movie
        genres = (self.__movies_df['genres']
            .apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
            .str.get_dummies('|')
            .values
        )

        # Convert to PyTorch tensor
        genres = torch.from_numpy(genres).to(torch.int)

        # One-hot encode and merge the production companies of each movie
        production_companies = (self.__movies_df['production_companies']
            .apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
            .str.get_dummies('|')
            .values
        )

        # Convert to PyTorch tensor
        production_companies = torch.from_numpy(production_companies).to(torch.int)

        ## BUILD NODES EXTERNAL FEATURES
        # Build the external features of the movie node by concatenating: genres and titles features
        movie_features = torch.cat([titles, genres], dim=-1)
        # Other tested solutions:
        # movie_features = torch.cat([titles, collections, genres], dim=-1)
        # movie_features = torch.cat([titles, genres, production_companies], dim=-1)
        # movie_features = torch.cat([titles, collections, genres, production_companies], dim=-1)

        self.__X_movies = movie_features
        # X_movies.shape = [num_movie_nodes x movie_node_feature_dim]

    def __build_user_nodes_features(self):
        """
            Constructs user node features (identity matrix, since we do not have users' external 
            features).

            Returns:
                None
        """
        # We don't have user features, which is why we use an identity matrix
        user_features = torch.eye(len(self.__users_ratings_df['userId'].unique()))

        # Convert PyTorch tensor
        self.__X_users = user_features
        # X_users.shape = [num_users_nodes x user_node_feature_dim]

    def __build_user_and_movie_nodes_indices(self):
        """
            Constructs mappings for user and movie nodes, assigning a unique index to each user to
            each movie.

            This function ensures that all users and movies from the input datasets are correctly 
            mapped to unique IDs in the graph. The user IDs are derived from the `users_ratings_df`,
            while the movie IDs are derived from `movies_df`.

            Returns:
                None
        """
        ## GENERATE UNIQUE (USER AND MOVIE) NODE INDICES
        # Create a mapping from the userId to a unique consecutive value in the range [0, num_users]
        self.__unique_users_ids = self.__users_ratings_df['userId'].unique()
        self.__unique_users_ids = pd.DataFrame(data={
            'userId': self.__unique_users_ids,
            'mappedUserId': pd.RangeIndex(len(self.__unique_users_ids))
        })

        # Create a mapping from the movieId to a unique consecutive value in the range [0, num_movies]
        self.__unique_movies_ids = self.__movies_df['id'].unique()
        self.__unique_movies_ids = pd.DataFrame(data={
            'movieId': self.__unique_movies_ids,
            'mappedMovieId': pd.RangeIndex(len(self.__unique_movies_ids))
        })
        
    def __build_edges_indices_and_labels(self):
        """
            Constructs and indices edges between user and movie nodes. Each edge has a user rating 
            as label.

            This function filters out edges corresponding to rated movies which are not present in 
            `movies_df`. This ensures that all movie nodes have valid features and prevents 
            featureless nodes from being included in the graph.

            Notes: 
                - Not all rated movies are used as edges in the graph, only the ones present in 
                  `movies_df`.
                - Not all movies in `movies_df` are used as have a corresponding edge in the graph,
                  only the ones rated by at least one user in `users_ratings_df`.
                - Including only nodes with valid features is meant to improve the learning process
                  of the GNN, as nodes without features can introduce noise and reduce model 
                  performance.

            Returns:
                None
        """
        ## CONNECT HETEROGENEOUS NODES THROUGH EDGES
        # Merge (user and movies unique indices) mappings in the users_ratings_df
        self.__users_ratings_df = self.__users_ratings_df.merge(self.__unique_users_ids, on='userId')
        self.__users_ratings_df = self.__users_ratings_df.merge(self.__unique_movies_ids, on='movieId')

        # Create the edge_index representation in COO format, following the PyG semantics
        self.__edge_index_user_to_movie = torch.stack([
            torch.tensor(self.__users_ratings_df['mappedUserId'].values),
            torch.tensor(self.__users_ratings_df['mappedMovieId'].values)
        ], dim=0)
        # edge_index_user_to_movie.shape = [2, num_ratings]

        ## BUILD EDGE LABELS (= RATINGS OF RATED MOVIES)
        # Extract the labels: the ratings (which are the edges features)
        ratings = self.__users_ratings_df['rating'].values

        # Convert edge labels in PyTorch tensor
        self.__y_ratings = torch.from_numpy(ratings).to(torch.float)
        # y_ratings.shape = [num_ratings] = [tot_num_of_edges]

        """
        # Debugging print edges
        print("Edge Labels (y_ratings):", self.__y_ratings)
        print("Edge Index (user-movie):", self.__edge_index_user_to_movie)
        """
