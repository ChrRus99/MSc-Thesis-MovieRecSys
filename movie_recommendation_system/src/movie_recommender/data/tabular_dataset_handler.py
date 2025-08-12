# Dataset
import copy
import pandas as pd

# Util
import ast
import numpy as np
import os
import pickle

"""
References:
    - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class TabularDatasetHandler:
    """
    This class handles the loading, preprocessing, and saving of MovieLens tabular datasets for 
    movie recommendations.

    Datasets:
        - movies_metadata.csv: Contains information on 45,000 movies featured in the Full MovieLens dataset.
        - keywords.csv: Contains movie plot keywords in the form of a stringified JSON Object.
        - credits.csv: Consists of Cast and Crew Information for all movies.
        - links.csv: Contains the TMDB and IMDB IDs of all movies in the Full MovieLens dataset.
        - links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies.
        - ratings_small.csv: Subset of 100,000 ratings from 700 users on 9,000 movies.
    """

    def __init__(self, data_path: str):
        self.__data_path = data_path

    def get_users_ratings_df_deepcopy(self) -> pd.DataFrame:
        """ Returns a deep copy of the users_ratings DataFrame. """
        return copy.deepcopy(self.__users_ratings_df)

    def get_movies_df_deepcopy(self) -> pd.DataFrame:
        """ Returns a deep copy of the movies DataFrame. """
        return copy.deepcopy(self.__movies_df)

    def get_links_df_deepcopy(self) -> pd.DataFrame:
        """ Returns a deep copy of the links DataFrame. """
        return copy.deepcopy(self.__links_df)

    def get_credits_df_deepcopy(self) -> pd.DataFrame:
        """ Returns a deep copy of the credits DataFrame. """
        return copy.deepcopy(self.__credits_df)

    def get_keywords_df_deepcopy(self) -> pd.DataFrame:
        """ Returns a deep copy of the keywords DataFrame."""
        return copy.deepcopy(self.__keywords_df)

    def get_valid_ratings(self) -> pd.DataFrame:
        """ 
        Extracts and returns only the ratings in 'users_ratings_df' for movies that are present
        in the 'movies_df'.
        """
        # Filter ratings to include only those with movieId present in movies_df
        valid_ratings_df = self.__users_ratings_df[self.__users_ratings_df['movieId'].isin(self.__movies_df['id'])]
        return valid_ratings_df

    def get_rated_movies_df(self) -> pd.DataFrame:
        """ Extracts and returns only the movies that have been rated by users. """
        # Get valid ratings
        filtered_users_ratings_df = self.get_valid_ratings()

        # Merge valid ratings with movies_df to get rated movies
        rated_movies_df = filtered_users_ratings_df.merge(self.__movies_df, left_on='movieId', right_on='id')
        return rated_movies_df

    def load_datasets(self, custom_filepaths: dict = None):
        """
        Load the datasets.

        This function allows to select custom file paths for the datasets. If no custom paths are
        provided, the default paths will be used.

        Parameters:
            custom_filepaths (dict, optional): A dictionary where keys are dataset names 
                ('users_ratings_df', 'movies_df', 'links_df', 'credits_df', 'keywords_df') and 
                values are custom filepaths.

        """
        loader = self.__DatasetLoader(self.__data_path, custom_filepaths)
        (
            self.__users_ratings_df,
            self.__movies_df,
            self.__links_df,
            self.__credits_df,
            self.__keywords_df,
        ) = loader.load_datasets()
        
        return self
    
    def update_datasets(
        self,
        users_ratings_df: pd.DataFrame = None,
        movies_df: pd.DataFrame = None,
        links_df: pd.DataFrame = None,
        credits_df: pd.DataFrame = None,
        keywords_df: pd.DataFrame = None
    ):
        """
        Updates the class's DataFrames with the provided DataFrames if they are not None.
        """
        if users_ratings_df is not None:
            self.__users_ratings_df = users_ratings_df
        if movies_df is not None:
            self.__movies_df = movies_df
        if links_df is not None:
            self.__links_df = links_df
        if credits_df is not None:
            self.__credits_df = credits_df
        if keywords_df is not None:
            self.__keywords_df = keywords_df    
    
    def preprocess_datasets(self):
        """
        Preprocesses the loaded datasets.

        This function should be called after loading the datasets to perform necessary preprocessing
        steps.
        """
        preprocessor = self.__DatasetPreprocessor(
            self.__users_ratings_df,
            self.__movies_df,
            self.__links_df,
            self.__credits_df,
            self.__keywords_df,
        )

        preprocessor.preprocess_datasets()

        self.__users_ratings_df = preprocessor.users_ratings_df
        self.__movies_df = preprocessor.movies_df
        self.__links_df = preprocessor.links_df
        self.__credits_df = preprocessor.credits_df
        self.__keywords_df = preprocessor.keywords_df
        
        return self

    def save_preprocessed_tabular_datasets(self, path: str):
        """
        Saves all datasets to the given path in CSV format.

        Parameters:
            path (str): The directory path to save the datasets.
        """
        datasets = {
            "users_ratings.csv": self.__users_ratings_df,
            "movies.csv": self.__movies_df,
            "links.csv": self.__links_df,
            "credits.csv": self.__credits_df,
            "keywords.csv": self.__keywords_df,
        }

        os.makedirs(path, exist_ok=True)

        for filename, df in datasets.items():
            try:
                df.to_csv(os.path.join(path, filename), index=False)
            except Exception as e:
                raise Exception(f"Error saving dataset '{filename}' to path '{path}': {e}")
            
        print(f"Preprocessed datasets saved to '{path}'")

    @staticmethod
    def load_preprocessed_tabular_datasets(path: str):
        """
        Loads all datasets from the given directory and returns a new TabularDatasetHandler instance.

        Parameters:
            path (str): The directory path to load the datasets from.

        Returns:
            TabularDatasetHandler: A new instance with loaded datasets.
        """
        handler = TabularDatasetHandler(data_path=path)

        datasets = {
            "users_ratings.csv": "__users_ratings_df",
            "movies.csv": "__movies_df",
            "links.csv": "__links_df",
            "credits.csv": "__credits_df",
            "keywords.csv": "__keywords_df",
        }

        for filename, attr in datasets.items():
            file_path = os.path.join(path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file '{filename}' not found in directory '{path}'")
            try:
                setattr(handler, attr, pd.read_csv(file_path))
            except Exception as e:
                raise Exception(f"Error loading dataset '{filename}' from path '{path}': {e}")

        return handler

    def store_class_instance(self, filepath):
        """
        Store the entire class instance to a .pkl file.

        Args:
            filepath (str): Path to the .pkl file where the instance will be saved.

        Returns:
            None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Class instance saved to '{filepath}'")

    @staticmethod
    def load_class_instance(filepath):
        """
        Loads a class instance from a .pkl file.

        Args:
            filepath (str): Path to the .pkl file from which the instance will be loaded.

        Returns:
            TabularDatasetHandler: The loaded class instance.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    class __DatasetLoader:
        """
        A private class responsible for loading datasets.
        """

        def __init__(self, data_path, custom_filepaths=None):
            self.__data_path = data_path
            self.__custom_filepaths = custom_filepaths

        def load_datasets(self):
            """
            Loads datasets from either the default paths or user-specified custom paths.


            Returns:
                Tuple: A tuple containing DataFrames for users_ratings, movies, links, credits, 
                    and keywords.
            """
            # Default dataset paths
            default_paths = {
                "users_ratings_df": os.path.join(self.__data_path, "ratings_small.csv"),
                "movies_df": os.path.join(self.__data_path, "movies_metadata.csv"),
                "links_df": os.path.join(self.__data_path, "links_small.csv"),
                "credits_df": os.path.join(self.__data_path, "credits.csv"),
                "keywords_df": os.path.join(self.__data_path, "keywords.csv"),
            }

            # Override defaults with custom paths if provided
            if self.__custom_filepaths is not None:
                dataset_paths = {key: self.__custom_filepaths.get(key, default_paths[key]) for key in default_paths}
            else:
                dataset_paths = default_paths

            # Load datasets
            try:
                users_ratings_df = pd.read_csv(dataset_paths["users_ratings_df"])
                movies_df = pd.read_csv(dataset_paths["movies_df"])
                links_df = pd.read_csv(dataset_paths["links_df"])
                credits_df = pd.read_csv(dataset_paths["credits_df"])
                keywords_df = pd.read_csv(dataset_paths["keywords_df"])
            except Exception as e:
                raise Exception(f"Error loading datasets: {e}")

            return users_ratings_df, movies_df, links_df, credits_df, keywords_df

    class __DatasetPreprocessor:
        """
        A private class responsible for preprocessing datasets.
        """

        def __init__(self, users_ratings_df, movies_df, links_df, credits_df, keywords_df):
            self.users_ratings_df = users_ratings_df
            self.movies_df = movies_df
            self.links_df = links_df
            self.credits_df = credits_df
            self.keywords_df = keywords_df

        def preprocess_datasets(self):
            """ Preprocesses the loaded datasets in place. """
            self.__preprocess_users_ratings_df()
            self.__preprocess_movies_df()
            self.__preprocess_links_df()
            self.__preprocess_credits_df()
            self.__preprocess_keywords_df()

        def __preprocess_users_ratings_df(self):
            """ Preprocesses the users_ratings DataFrame. """
            pass

        def __preprocess_movies_df(self):
            """ Preprocesses the movies DataFrame. """
            # Remove bad 'id'
            self.movies_df = self.movies_df.drop([19730, 29503, 35587], errors='ignore')

            # Drop duplicates based on the 'id' column
            self.movies_df = self.movies_df.drop_duplicates(subset='id')

            # Drop columns 'imdb_id', 'poster_path', 'video'
            self.movies_df = self.movies_df.drop(columns=['imdb_id', 'poster_path', 'video'])

            # Process the 'id' column
            self.movies_df['id'] = pd.to_numeric(   # Handle non-numeric values by converting them to NaN
                self.movies_df['id'],
                errors='coerce'
            )
            self.movies_df = self.movies_df.dropna( # Drop rows with NaN values in the 'id' column
                subset=['id']
            )
            self.movies_df['id'] = self.movies_df['id'].astype('int')

            # Process the 'title' column: substitute NaN and empty titles with the corresponding 'original_title'
            invalid_titles_df = self.movies_df[self.movies_df['title'].isnull() | (self.movies_df['title'] == '')]
            self.movies_df.loc[invalid_titles_df.index, 'title'] = invalid_titles_df['original_title']

            # Process the 'genres' column
            self.movies_df['genres'] = (self.movies_df['genres']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda genres: [genre['name'] for genre in genres] if isinstance(genres, list) else [])
            )

            # Add a new column 'year'
            self.movies_df['year'] = (pd.to_datetime(self.movies_df['release_date'], errors='coerce')
                .apply(lambda date: str(date).split('-')[0] if date != np.nan else np.nan)
            )

            # Process the 'production_companies' column
            self.movies_df['production_companies'] = (self.movies_df['production_companies']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda prod_companies: [prod_company['name'] for prod_company in prod_companies]
                    if isinstance(prod_companies, list) else [])
            )

            # Process the 'production_countries' column
            self.movies_df['production_countries'] = (self.movies_df['production_countries']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda prod_countries: [prod_country['name'] for prod_country in prod_countries]
                    if isinstance(prod_countries, list) else [])
            )

            # Process the 'belongs_to_collection' column
            self.movies_df['belongs_to_collection'] = (self.movies_df['belongs_to_collection']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda collections: collections.get('name') if isinstance(collections, dict) else None)
            )

            # Process the 'spoken_languages' column
            self.movies_df['spoken_languages'] = (self.movies_df['spoken_languages']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda spoken_langs: [spoken_lang['name'] for spoken_lang in spoken_langs])
            )

        def __preprocess_links_df(self):
            """ Preprocesses the links DataFrame. """
            # Process the 'tmdbId' column
            self.links_df.dropna(subset=['tmdbId'], inplace=True)
            self.links_df['tmdbId'] = self.links_df['tmdbId'].astype('int')

        def __preprocess_credits_df(self):
            """ Preprocesses the credits DataFrame. """
            # Process the 'id' column
            self.credits_df['id'] = self.credits_df['id'].astype('int')

            # Process 'cast' and 'crew' columns
            self.credits_df['cast'] = self.credits_df['cast'].apply(ast.literal_eval)
            self.credits_df['crew'] = self.credits_df['crew'].apply(ast.literal_eval)

            # Process 'cast' column to extract only relevant data
            self.credits_df['cast'] = self.credits_df['cast'].apply(
                lambda cast_list: [
                    {'character': x['character'], 'name': x['name']}
                    for x in cast_list
                ] if isinstance(cast_list, list) else None
            )
            
            # Process 'crew' column to extract only relevant data
            self.credits_df['crew'] = self.credits_df["crew"].apply(
                lambda crew_list: [
                    {'department': x['department'], 'job': x['job'], 'name': x['name']}
                    for x in crew_list
                ] if isinstance(crew_list, list) else None
            )

        def __preprocess_keywords_df(self):
            """ Preprocesses the keywords DataFrame. """
            # Process the 'id' column
            self.keywords_df['id'] = self.keywords_df['id'].astype('int')

            # Process 'keywords' columns
            self.keywords_df['keywords'] = self.keywords_df['keywords'].apply(ast.literal_eval)

            # Process the 'keywords' column, for extracting only the names of the keywords
            self.keywords_df['keywords'] = self.keywords_df['keywords'].apply(
                lambda keyword_row: [
                    {'name': x['name']}
                    for x in keyword_row
                ] if isinstance(keyword_row, list) else []
            )