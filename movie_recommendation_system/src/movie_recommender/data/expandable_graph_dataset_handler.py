# Dataset
import pandas as pd

# Pytorch Geometric
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# My scripts
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler


class ExpandableHeterogeneousGraphDatasetHandler(HeterogeneousGraphDatasetHandler):
    """
    This class handles a heterogeneous graph dataset that can be expanded with new nodes and edges.
    
    This class extends the HeterogeneousGraphDatasetHandler class by adding the capability to expand
    the graph dataset with new nodes and edges. It allows to add new movie and user nodes, and to 
    add new user-movie ratings.

    This class also provides a subgraph dataset containing all the updated nodes in the graph 
    dataset (old and new ones), and only the new edges. This subgraph is meant to be be used for
    re-training tasks: full retraining, incremental training, continuous learning, distillation 
    learning or fine-tuning of a pre-trained model.

    Attributes:
        ... same as HeterogeneousGraphDatasetHandler ...
        _new_data (HeteroData): The heterogeneous suggraph dataset containing all the updated nodes  
            in the graph dataset (old and new ones), and only the new edges.
    """

    def __init__(self, prebuilt_gdh: HeterogeneousGraphDatasetHandler):
        super().__init__(prebuilt_gdh._tdh)

        # Copy the data from the prebuilt HeterogeneousGraphDatasetHandler instance
        self._tdh = prebuilt_gdh._tdh
        self._movies_df = prebuilt_gdh._movies_df
        self._users_ratings_df = prebuilt_gdh._users_ratings_df

        self._data = prebuilt_gdh._data

        self._X_movies = prebuilt_gdh._X_movies
        self._X_users = prebuilt_gdh._X_users
        self._y_ratings = prebuilt_gdh._y_ratings
        self._user_to_movie_edge_indices = prebuilt_gdh._user_to_movie_edge_indices
        
        self._mapped_users_ids_df = prebuilt_gdh._mapped_users_ids_df
        self._mapped_movies_ids_df = prebuilt_gdh._mapped_movies_ids_df

        self._unique_genres = prebuilt_gdh._unique_genres

        # Subgraph data containg only new nodes and edges
        self._new_data = HeteroData()

    def get_subgraph_dataset(self):
        """
        Returns a subgraph dataset containing only the new nodes and edges.

        Returns:
            HeteroData: The subgraph dataset containing only the new nodes and edges.
        """
        return self._new_data

    def clear_suggraph_dataset(self):
        """ 
        Clears the subgraph dataset. 
        
        This functions resets the subgraph dataset containing only the new nodes and edges, without
        affecting the main graph dataset.
        """
        self._new_data = HeteroData()

    # Deprecated function: rebuild the whole graph every time a new movie is added
    # def add_new_movies(self, new_movies_df):
    #     """
    #     Adds new movies to the graph dataset.

    #     This function takes a DataFrame of new movies and updates the internal dataset by adding
    #     the new movie nodes and their features. 
        
    #     Notes: 
    #         - After storing the new movies, this function rebuild the graph dataset from scratch
    #           using the function 'build_graph_dataset'. 
    #         - Notice that (unlike user's ratings in the function 'add_new_users_ratings'), here it
    #           is not possible to just update the movie nodes, and this is due to the one-hot
    #           encoding procedure used in the movie node features construction which would cause 
    #           feature size inconsistencies.

    #     Parameters:
    #         new_movies_df (pd.DataFrame): A DataFrame containing the new movies to be added.

    #     Returns:
    #         None
    #     """
    #     ## FILTER NEW MOVIES
    #     # Extract the ids of the new movies to store
    #     all_new_movies_ids = new_movies_df["id"].to_list()

    #     # Filter out movies that are already in the dataset
    #     new_movies_df = new_movies_df[~new_movies_df['id'].isin(self._movies_df['id'])]
        
    #     # Extract the ids of the filtered new movies to store
    #     filtered_new_movies_ids = new_movies_df["id"].to_list()

    #     # If all movies are already present in 'movies_df' do nothing and return
    #     if len(filtered_new_movies_ids) == 0:
    #         print(f"No movie added: Movies ids {all_new_movies_ids} are all already present in 'movies_df'.")
    #         return
        
    #     print(f"Found {len(filtered_new_movies_ids)} new movies to add to 'movies_df': Movies ids {filtered_new_movies_ids}")

    #     ## STORE NEW MOVIES IN 'movies_df'
    #     # Update 'movies_df' by appending the new movies to the existing ones
    #     self._movies_df = pd.concat([self._movies_df, new_movies_df])

    #     ## EXTEND THE HETEROGENEOUS GRAPH DATASET WITH THE NEW MOVIES (BY RE-BUILDING IT FROM SCRATCH)
    #     # Note: we cannot just update movie nodes because to build their features we use one-hot encoding,
    #     # hence new features would have different feature sizes, thus causing inconsistencies!
    #     """
    #     # Build features for the new movies
    #     self._build_movie_nodes_features(new_movies_df)
        
    #     # Update the indices for user and movie nodes
    #     self._build_user_and_movie_nodes_indices(None, new_movies_df)

    #     # Update the graph dataset with new movie nodes
    #     self._data['movie'].node_id = torch.tensor(self._mapped_movies_ids_df['mappedMovieId'].to_list())
    #     self._data['movie'].x = self._X_movies
    #     """

    #     # The only way is to rebuild the whole graph dataset (with updated 'movies_df')
    #     self.build_graph_dataset()

    #     ## STORE THE HETEROGENEOUS SUBGRAPH DATASET CONTAINING ONLY THE NEW EDGES DATA (FOR INCREMENTAL TRAINING)
    #     # Add ALL mapped movie node indices to the heterogeneous subgraph dataset
    #     self._new_data['movie'].node_id = self._data['movie'].node_id

    #     # Add ALL movie node features to the heterogeneous subgraph dataset
    #     self._new_data['movie'].x = self._data['movie'].x

    def add_new_movies(self, new_movies_df):
        """
        Adds new movies to the heterogeneous graph dataset.

        Note: careful not to introduce movies with a genre not present in the MovieLens dataset.
        This behaviour lead to a mismatch between the number of external features of the new movies
        due to the one-hot encoding of the genres, thus causing inconsistency errors!

        Parameters:
            new_movies_df (pd.DataFrame): A DataFrame containing new movie metadata with an 'id' column.
        
        Workflow:
            1. Filter out movies that already exist in the dataset.
            2. Update 'movies_df' with new movie entries.
            3. Extend movie node features and indices.
            4. Update the full graph dataset.
            5. Store new movies in a subgraph dataset for re-training.
        """
        # Check whether the graph has been initialized
        if (self._mapped_users_ids_df is None) or (self._mapped_movies_ids_df is None):
            print("The graph has not yet been initialized.")
            print("You need to build the graph by calling the function 'build_graph_dataset' first!")
            return
        
        ## FILTER NEW MOVIES
        # Extract the ids of the new movies to store
        all_new_movies_ids = new_movies_df["id"].to_list()

        # Filter out movies that are already in the dataset
        new_movies_df = new_movies_df[~new_movies_df['id'].isin(self._movies_df['id'])]
        
        # Extract the ids of the filtered new movies to store
        filtered_new_movies_ids = new_movies_df["id"].to_list()

        # If all movies are already present in 'movies_df' do nothing and return
        if len(filtered_new_movies_ids) == 0:
            print(f"No movie added: Movies ids {all_new_movies_ids} are all already present in 'movies_df'.")
            return
        
        print(f"Found {len(filtered_new_movies_ids)} new movies to add to 'movies_df': Movies ids {filtered_new_movies_ids}")

        ## STORE NEW MOVIES IN 'movies_df'
        # Update 'movies_df' by appending the new movies to the existing ones
        self._movies_df = pd.concat([self._movies_df, new_movies_df])

        ## EXTEND THE HETEROGENEOUS GRAPH DATASET WITH THE NEW MOVIES 
        # Strong assumption: the set of possible genres for movies does not change over time, hence
        # the length of the one-hot encoding of the genres, and thus the length of the external
        # feature vector of movie nodes is preserved over time for new movies.

        # Extend movie nodes features
        movie_features = self._build_movie_nodes_features(new_movies_df)
        self._X_movies = self._append_or_initialize(self._X_movies, movie_features, dim=0)

        # Extend movie nodes indices
        _, mapped_movie_ids_df = self._build_user_and_movie_nodes_indices(None, new_movies_df)
        self._mapped_movies_ids_df = self._append_or_initialize(self._mapped_movies_ids_df, mapped_movie_ids_df)

        ## STORE THE FULL UPDATED HETEROGENEOUS GRAPH DATASET DATA
        # Update mapped movie node indices in the heterogeneous graph dataset
        self._data['movie'].node_id = torch.tensor(self._mapped_movies_ids_df['mappedMovieId'].to_list())

        # Update movie node features in the heterogeneous graph dataset
        self._data['movie'].x = self._X_movies

        ## STORE THE HETEROGENEOUS SUBGRAPH DATASET CONTAINING ONLY THE NEW EDGES DATA (FOR RE-TRAINING)
        # Add ALL mapped movie node indices to the heterogeneous subgraph dataset
        self._new_data['movie'].node_id = self._data['movie'].node_id

        # Add ALL movie node features to the heterogeneous subgraph dataset
        self._new_data['movie'].x = self._data['movie'].x

    def add_new_user_movie_ratings(self, new_ratings_df: pd.DataFrame):
        """
        Adds new user-movie rating edges to the heterogeneous graph dataset.
        
        Parameters:
            new_ratings_df (pd.DataFrame): A DataFrame containing new user-movie ratings with 
                'userId' and 'movieId' columns.
        
        Workflow:
            1. Filter out ratings for non-existing movies.
            2. Remove duplicate ratings.
            3. Update 'users_ratings_df' with new ratings.
            4. Extend user features, indices, and edge connections.
            5. Update the full graph dataset.
            6. Store new edges in a subgraph dataset for re-training.
        """
        # Check whether the graph has been initialized
        if (self._mapped_users_ids_df is None) or (self._mapped_movies_ids_df is None):
            print("The graph has not yet been initialized.")
            print("You need to build the graph by calling the function 'build_graph_dataset' first")
            return

        ## FILTER NEW RATINGS
        # Filter new ratings to extract only the ones about movies which are present in 'movies_df'
        new_ratings_df = new_ratings_df[new_ratings_df['movieId'].isin(self._movies_df['id'])]

        # Filter new ratings to include only those that are not already present in the graph dataset
        existing_ratings = self._users_ratings_df[['userId', 'movieId']].drop_duplicates()
        existing_ratings = existing_ratings[existing_ratings['movieId'].isin(self._movies_df['id'])]

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

        # Filter new ratings to extract only the ones of new users
        new_users = ~new_ratings_df['userId'].isin(self._users_ratings_df['userId'])
        new_users_ratings_df = new_ratings_df[new_users]

        # Extract the number of new users to store
        new_user_ids = list(set(new_users_ratings_df['userId'].to_list()))
        num_new_user_ids = len(new_user_ids)

        ## STORE NEW RATINGS IN 'users_ratings_df'
        # Update 'users_ratings_df' by appending the new ratings to the existing ones
        self._users_ratings_df = pd.concat([self._users_ratings_df, new_ratings_df])

        ## EXTEND THE HETEROGENEOUS GRAPH DATASET WITH THE NEW RATINGS
        # If there are new users, we need to update the user nodes features and indices
        if num_new_user_ids > 0:
            # Extend user nodes features
            user_features = self._build_user_nodes_features(new_users_ratings_df)
            self._X_users = self._append_or_initialize(self._X_users, user_features, dim=0) 

            # Extend user nodes indices
            mapped_user_ids_df, _ = self._build_user_and_movie_nodes_indices(new_users_ratings_df, None)
            self._mapped_users_ids_df = self._append_or_initialize(self._mapped_users_ids_df, mapped_user_ids_df)

            print(f"Found {num_new_user_ids} new users to add to 'users_ratings_df': Users ids {new_user_ids}")
        else:
            print(f"No new users to add. Users {list(set(new_ratings_df['userId'].to_list()))} are already present in 'users_ratings_df'.")

        # Extend user-movie edges indices and labels
        edge_indices, edge_labels = self._build_edges_indices_and_labels(new_ratings_df)
        self._user_to_movie_edge_indices = self._append_or_initialize(self._user_to_movie_edge_indices, edge_indices, dim=1)
        self._y_ratings = self._append_or_initialize(self._y_ratings, edge_labels, dim=0)

        ## STORE THE FULL UPDATED HETEROGENEOUS GRAPH DATASET DATA 
        # Update mapped user node indices in the heterogeneous graph dataset
        self._data['user'].node_id = torch.tensor(self._mapped_users_ids_df['mappedUserId'].to_list())

        # Update user node features in the heterogeneous graph dataset
        self._data['user'].x = self._X_users

        # Update edge indices in the heterogeneous graph dataset
        self._data['user', 'rating', 'movie'].edge_index = self._user_to_movie_edge_indices

        # Update edge labels in the heterogeneous graph dataset
        self._data['user', 'rating', 'movie'].edge_label = self._y_ratings
        self._data['user', 'movie'].y = self._y_ratings

        # Re-build reverse edges from scratch
        reverse_edge_type = ('movie', 'rev_rating', 'user')
        if reverse_edge_type in self._data.edge_types:
            del self._data[reverse_edge_type]

        self._data = T.ToUndirected()(self._data)

        ## STORE THE HETEROGENEOUS SUBGRAPH DATASET CONTAINING ONLY THE NEW EDGES DATA (FOR RE-TRAINING)
        # Add ALL mapped user node indices in the heterogeneous subgraph dataset
        self._new_data['user'].node_id = self._data['user'].node_id

        # Add ALL user node features in the heterogeneous subgraph dataset
        self._new_data['user'].x = self._data['user'].x

        # Add/update ONLY NEW edge indices in the heterogeneous subgraph dataset
        if hasattr(self._new_data['user', 'rating', 'movie'], 'edge_index'):
            self._new_data['user', 'rating', 'movie'].edge_index = torch.cat((
                self._new_data['user', 'rating', 'movie'].edge_index,
                edge_indices
            ), dim=1)
        else:
            self._new_data['user', 'rating', 'movie'].edge_index = edge_indices

        # Add/update ONLY NEW edge labels in the heterogeneous subgraph dataset
        if hasattr(self._new_data['user', 'rating', 'movie'], 'edge_label'):
            self._new_data['user', 'rating', 'movie'].edge_label = torch.cat((
                self._new_data['user', 'rating', 'movie'].edge_label,
                edge_labels
            ), dim=0)
        else:
            self._new_data['user', 'rating', 'movie'].edge_label = edge_labels
        if hasattr(self._new_data['user', 'movie'], 'y'):
            self._new_data['user', 'movie'].y = torch.cat((
                self._new_data['user', 'movie'].y,
                edge_labels
            ), dim=0)
        else:
            self._new_data['user', 'movie'].y = edge_labels

        # Re-build reverse edges from scratch
        reverse_edge_type = ('movie', 'rev_rating', 'user')
        if reverse_edge_type in self._new_data.edge_types:
            del self._new_data[reverse_edge_type]

        self._new_data = T.ToUndirected()(self._new_data)

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