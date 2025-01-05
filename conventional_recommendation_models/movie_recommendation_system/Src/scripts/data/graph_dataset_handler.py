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
            __mapped_users_ids_df (pd.DataFrame): DataFrame mapping original user IDs to consecutive integers.
            __mapped_movies_ids_df (pd.DataFrame): DataFrame mapping original movie IDs to consecutive integers.
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
        self.__users_ratings_df = self.__tdh.get_valid_ratings() #self.__tdh.get_users_ratings_df_deepcopy()

        # Initialize a new heterogeneous graph dataset
        self.__data = HeteroData()       

        # Heterogeneous graph dataset nodes and edges data
        self.__X_movies = None
        self.__X_users = None
        self.__y_ratings = None
        self.__edge_index_user_to_movie = None

        # Mapped nodes ids
        self.__mapped_users_ids_df = None
        self.__mapped_movies_ids_df = None

    @property
    def users_ratings_df(self):
        return self.__users_ratings_df

    @property
    def movies_df(self):
        return self.__movies_df

    @property
    def mapped_users_ids_df(self):
        return self.__mapped_users_ids_df

    @property
    def mapped_movies_ids_df(self):
        return self.__mapped_movies_ids_df

    def add_new_movies(self, new_movies_df):
        """
            Adds new movies to the graph dataset.

            This function takes a DataFrame of new movies and updates the internal dataset by adding
            the new movie nodes and their features. 
            
            Notes: 
                - After storing the new movies, this function rebuild the graph dataset from scratch
                  using the function 'build_graph_dataset'. 
                - Notice that (unlike user's ratings in the function 'add_new_users_ratings'), here
                  it is not possible to just update the movie nodes, and this is due to the one-hot
                  encoding procedure used in the movie node features construction which would cause 
                  feature size inconsistencies.

            Parameters:
                new_movies_df (pd.DataFrame): A DataFrame containing the new movies to be added.

            Returns:
                None
        """
        # Extract the ids of the new movies to store
        all_new_movies_ids = new_movies_df["id"].to_list()

        # Filter out movies that are already in the dataset
        new_movies_df = new_movies_df[~new_movies_df['id'].isin(self.__movies_df['id'])]
        
        # Extract the ids of the filtered new movies to store
        filtered_new_movies_ids = new_movies_df["id"].to_list()

        # If all movies are already present in 'movies_df' do nothing and return
        if len(filtered_new_movies_ids) == 0:
            print(f"No movie added: Movies ids {all_new_movies_ids} are all already present in 'movies_df'.")
            return
        
        print(f"Found {len(filtered_new_movies_ids)} new movies to add to 'movies_df': Movies ids {filtered_new_movies_ids}")

        # Update 'movies_df' by appending the new movies to the existing ones
        self.__movies_df = pd.concat([self.__movies_df, new_movies_df])

        # Note: we cannot just update movie nodes because to build their features we use one-hot encoding,
        # hence new features would have different feature sizes, thus causing inconsistencies!
        """
        # Build features for the new movies
        self.__build_movie_nodes_features(new_movies_df)
        
        # Update the indices for user and movie nodes
        self.__build_user_and_movie_nodes_indices(None, new_movies_df)

        # Update the graph dataset with new movie nodes
        self.__data['movie'].node_id = torch.tensor(self.__mapped_movies_ids_df['mappedMovieId'].to_list())
        self.__data['movie'].x = self.__X_movies
        """

        # The only way is to rebuild the whole graph dataset (with updated 'movies_df')
        self.build_graph_dataset()
        
    def add_new_users_ratings(self, new_ratings_df: pd.DataFrame):
        """
            Adds new user ratings to the graph dataset.

            This function updates the graph dataset with new user ratings, ensuring that only 
            ratings for movies in 'movies_df' are considered. It also updates user nodes, movie 
            nodes, and the edges between them in the graph.

            Notes:
                - The function concatenates the new ratings with the existing user ratings.
                - It filters out ratings for movies that do not exist in 'movies_df'.
                - It ensures that only new unique ratings are added.
                - It updates user nodes features, user and movie nodes indices, and edges indices 
                  and labels.
                - Finally, it updates the internal graph data structure with the new nodes and edges.

            Parameters:
                new_ratings_df (pd.DataFrame): New user ratings DataFrame to be added.

            Returns:
                None
        """
        # Check whether the graph has been initialized
        if (self.__mapped_users_ids_df is None) or (self.__mapped_movies_ids_df is None):
            print("The graph has not yet been initialized.")
            print("You need to build the graph by calling the function 'build_graph_dataset' first")
            return

        # Filter new ratings to include only those movies are in 'movies_df'
        new_ratings_df = new_ratings_df[new_ratings_df['movieId'].isin(self.__movies_df['id'])]

        # Filter new ratings to include only those that are not already in the dataset
        existing_ratings = self.__users_ratings_df[['userId', 'movieId']].drop_duplicates()
        existing_ratings = existing_ratings[existing_ratings['movieId'].isin(self.__movies_df['id'])]

        new_ratings_df = new_ratings_df.merge(existing_ratings, on=['userId', 'movieId'], how='left', indicator=True)
        new_ratings_df = new_ratings_df[new_ratings_df['_merge'] == 'left_only'].drop('_merge', axis=1)

        # Extract the number of filtered new ratings to store
        num_filtered_new_ratings = new_ratings_df.shape[0]

        # If all (user, movie) ratings are already present in 'users_ratings_df' do nothing and return
        if num_filtered_new_ratings == 0:
            print("No ratings added: All (user, movie) ratings are already present in 'users_ratings_df'.")
            print("Re-build the graph with the function 'build_graph_dataset' if you want to update those ratings.")
            return

        print(f"Found {num_filtered_new_ratings} new ratings to add to 'users_ratings_df'")

        # Update 'users_ratings_df' by appending the new ratings to the existing ones
        self.__users_ratings_df = pd.concat([self.__users_ratings_df, new_ratings_df])

        # Build features and indices for new user nodes and edges
        self.__build_user_nodes_features(new_ratings_df)
        self.__build_user_and_movie_nodes_indices(new_ratings_df, None)
        self.__build_edges_indices_and_labels(new_ratings_df)

        # Update self.__data with new user nodes and edges
        self.__data['user'].x = self.__X_users
        self.__data['user'].node_id = torch.tensor(self.__mapped_users_ids_df['mappedUserId'].to_list())
        self.__data['user', 'rating', 'movie'].edge_index = self.__edge_index_user_to_movie
        self.__data['user', 'rating', 'movie'].edge_label = self.__y_ratings
        self.__data['user', 'movie'].y = self.__y_ratings

        # Re-build reverse edges from scratch
        reverse_edge_type = ('movie', 'rev_rating', 'user')
        if reverse_edge_type in self.__data.edge_types:
            del self.__data[reverse_edge_type]

        self.__data = T.ToUndirected()(self.__data)

    def build_graph_dataset(self):
        """
            Builds the heterogeneous graph dataset with user and movie nodes features and with edges
            labels.

            Returns:
                None
        """
        # Reset the heterogeneous graph dataset (this is useful to re-build the graph dataset from scratch each time)
        self.__reset_graph_dataset()

        # Build the heterogeneous graph dataset: nodes features and indixes, edges labels and indixes
        self.__build_movie_nodes_features(self.__movies_df)
        self.__build_user_nodes_features(self.__users_ratings_df)
        self.__build_user_and_movie_nodes_indices(self.__users_ratings_df, self.__movies_df)
        self.__build_edges_indices_and_labels(self.__users_ratings_df)

        # Save node mapped indices 
        self.__data['user'].node_id = torch.tensor(self.__mapped_users_ids_df['mappedUserId'].to_list())
        self.__data['movie'].node_id = torch.tensor(self.__mapped_movies_ids_df['mappedMovieId'].to_list())

        # Add the nodes features to the heterogeneous graph dataset
        self.__data['user'].x = self.__X_users  # dim = [num_users, num_features_users]
        self.__data['movie'].x = self.__X_movies  # dim = [num_movies, num_features_movies]

        # Add edge indices to the heterogeneous graph dataset
        self.__data['user', 'rating', 'movie'].edge_index = self.__edge_index_user_to_movie  # dim = [2, num_ratings]
        # edge_type = [source='user', type='rating', destination='movie'] is the user-movie edge

        # Add edge labels to the heterogeneous graph dataset
        self.__data['user', 'rating', 'movie'].edge_label = self.__y_ratings
        self.__data['user', 'movie'].y = self.__y_ratings

        # Make the heterogeneous graph dataset bidirectional by adding the reverse edges from movies to users
        # In this a way the GNN will be able to pass messages in both directions.
        self.__data = T.ToUndirected()(self.__data)
        # edge_type = [source='movie', type='rev_rating', destination='user'] is the movie-user edge

        # out = [model(data.X_dict, data.edge_index_dict)]

        """
        # Debugging print mappings
        print("Mapped Users IDs:")
        print(self.__mapped_users_ids_df.head())

        print("Mapped Movies IDs:")
        print(self.__mapped_movies_ids_df.head())

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

    def __build_movie_nodes_features(self, movies_df):
        """
            Constructs movie node features from tabular movies_df.

            Parameters:
                movies_df (pd.DataFrame): Movies Dataframe.

            Returns:
                None
        """
        ## PROCESS EXTERNAL FEATURES
        # Encode the movie titles into fixed-size vectors using the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with torch.no_grad():
            titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=False)
            titles = titles.cpu()

        # Encode the collection each movie belongs to a unique number
        collections = movies_df['belongs_to_collection'].map(
            {collection: index for index, collection in enumerate(movies_df['belongs_to_collection'].unique())}
        ).fillna(0).astype(int).tolist()

        # Convert to PyTorch tensor
        collections = torch.from_numpy(np.array(collections)).view(-1, 1)

        # One-hot encode and merge the genres of each movie
        genres = (movies_df['genres']
            .apply(lambda x: '|'.join(x) if isinstance(x, list) else None)
            .str.get_dummies('|')
            .values
        )

        # Convert to PyTorch tensor
        genres = torch.from_numpy(genres).to(torch.int)

        # One-hot encode and merge the production companies of each movie
        production_companies = (movies_df['production_companies']
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

        # Store or update movie's features
        self.__X_movies = movie_features
        #self.__X_movies = self._append_or_initialize(self.__X_movies, movie_features, dim=0)
        # X_movies.shape = [num_movie_nodes x movie_node_feature_dim]

    def __build_user_nodes_features(self, users_ratings_df):
        """
            Constructs user node features (identity matrix, since we do not have users' external 
            features).

            Parameters:
                users_ratings_df (pd.DataFrame): User ratings DataFrame.

            Returns:
                None
        """
        # We don't have user features, which is why we use an identity matrix
        user_features = torch.eye(len(self.__users_ratings_df['userId'].unique()))

        # Store or update user's features
        self.__X_users = user_features
        # X_users.shape = [num_users_nodes x user_node_feature_dim]

    def __build_user_and_movie_nodes_indices(self, users_ratings_df, movies_df):
        """
            Constructs mappings for user and movie nodes, assigning a unique index to each user to
            each movie.

            This function ensures that all users and movies from the input datasets are correctly 
            mapped to unique IDs in the graph. The user IDs are derived from the `users_ratings_df`,
            while the movie IDs are derived from `movies_df`.

            Parameters:
                users_ratings_df (pd.DataFrame): User ratings DataFrame.
                movies_df (pd.DataFrame): Movies Dataframe.

            Returns:
                None
        """
        ## GENERATE UNIQUE (USER AND MOVIE) NODE INDICES
        # Create a mapping from the userId to a unique consecutive value in the range [0, num_users]
        if users_ratings_df is not None:
            last_user_index = self.__mapped_users_ids_df['mappedUserId'].max() + 1 if self.__mapped_users_ids_df is not None else 0
            new_users_ids = users_ratings_df['userId'].unique()
            new_users_df = pd.DataFrame(data={
                'userId': new_users_ids,
                'mappedUserId': pd.RangeIndex(last_user_index, last_user_index + len(new_users_ids))
            })
            self.__mapped_users_ids_df = self._append_or_initialize(self.__mapped_users_ids_df, new_users_df)

        # Create a mapping from the movieId to a unique consecutive value in the range [0, num_movies]
        if movies_df is not None:
            last_movie_index = self.__mapped_movies_ids_df['mappedMovieId'].max() + 1 if self.__mapped_movies_ids_df is not None else 0
            new_movies_ids = movies_df['id'].unique()
            new_movies_df = pd.DataFrame(data={
                'movieId': new_movies_ids,
                'mappedMovieId': pd.RangeIndex(last_movie_index, last_movie_index + len(new_movies_ids))
            })
            self.__mapped_movies_ids_df = self._append_or_initialize(self.__mapped_movies_ids_df, new_movies_df)
    
    def __build_edges_indices_and_labels(self, users_ratings_df):
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

            Parameters:
                users_ratings_df (pd.DataFrame): User ratings DataFrame.

            Returns:
                None
        """
        ## CONNECT HETEROGENEOUS NODES THROUGH EDGES
        # Merge (user and movies unique indices) mappings in the users_ratings_df
        users_ratings_df = users_ratings_df.merge(self.__mapped_users_ids_df, on='userId').merge(self.__mapped_movies_ids_df, on='movieId')
        
        # Create the edge_index representation in COO format, following the PyG semantics
        new_edge_index = torch.stack([
            torch.tensor(users_ratings_df['mappedUserId'].values),
            torch.tensor(users_ratings_df['mappedMovieId'].values)
        ], dim=0)

        # Store or update user-movie edge indices
        self.__edge_index_user_to_movie = self._append_or_initialize(self.__edge_index_user_to_movie, new_edge_index, dim=1)
        # edge_index_user_to_movie.shape = [2, num_ratings]

        ## BUILD EDGE LABELS (= RATINGS OF RATED MOVIES)
        # Build edge labels: users ratings (which are the edges features)
        new_ratings = torch.from_numpy(users_ratings_df['rating'].values).to(torch.float)

        # Store or update user-movie edge labels
        self.__y_ratings = self._append_or_initialize(self.__y_ratings, new_ratings, dim=0)
        # y_ratings.shape = [num_ratings] = [tot_num_of_edges]

        """
        # Debugging print edges
        print("Edge Labels (y_ratings):", self.__y_ratings)
        print("Edge Index (user-movie):", self.__edge_index_user_to_movie)
        """

    def __reset_graph_dataset(self):
        """
            Resets all internal attributes to their initial state.
        """
        self.__data = HeteroData()
        
        self.__X_movies = None
        self.__X_users = None
        self.__y_ratings = None
        self.__edge_index_user_to_movie = None

        self.__mapped_users_ids_df = None
        self.__mapped_movies_ids_df = None

    def _append_or_initialize(self, existing, new, dim=None):
        """
            Appends new data to existing data, or initializes it if it doesn't exist.

            Parameters:
                existing (Union[torch.Tensor, pd.DataFrame]): The existing data.
                new (Union[torch.Tensor, pd.DataFrame]): The new data to be added.
                dim (Optional[int]): The dimension along which to concatenate (only used for tensors).

            Returns:
                Union[torch.Tensor, pd.DataFrame]: The updated data.
        """
        if existing is None:
            return new
        if isinstance(existing, torch.Tensor) and isinstance(new, torch.Tensor):
            return torch.cat([existing, new], dim=dim)
        elif isinstance(existing, pd.DataFrame) and isinstance(new, pd.DataFrame):
            return pd.concat([existing, new], ignore_index=True)
        else:
            raise TypeError("Both inputs must be of the same type (either Tensor or DataFrame).")
