# Dataset
import pandas as pd
import numpy as np

# Util
import ast
from nltk.stem.snowball import SnowballStemmer

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# My scripts
from movie_recommender.recommenders.popularity_ranking import PopularityRanking

"""
References:
    - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""

class ContentBasedFiltering:
    """
    ContentBasedFiltering - Movie Recommendation System based on Content-Based Filtering.

    This class provides functions for implementing a movie recommendation system based on 
    content-based filtering.
    It includes methods for recommending movies based on movie descriptions and metadata (cast,
    crew, keywords, and genre).
    """

    @staticmethod
    def description_based_recommender(tabular_dataset_handler, movie_title, N, improved=False):
        """
        Recommends the N top most similar movies to the target movie based on movie descriptions
        and taglines.

        This function DOES NOT provide personalized recommendations based on the user.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of DatasetHandler.
            movie_title (str): The title of the movie.
            N (int): The number of top most similar movies to recommend.
            improved (bool): Whether to use improved recommendation using the weighted rating (wr)
                when True.

        Returns:
            pd.DataFrame: The N top most similar movies based on content.
        """
        # Initialize a brand-new copy of the 'movies_df'
        movies_df = tabular_dataset_handler.get_movies_df_deepcopy()

        # Process the 'tagline' column
        movies_df['tagline'] = movies_df['tagline'].fillna('')

        # Add a new column 'description' containing the concatenation of the 'overview' and of the 'tagline' columns
        movies_df['description'] = movies_df['overview'] + movies_df['tagline']
        movies_df['description'] = movies_df['description'].fillna('')

        # Create a TF-IDF vectorizer and generate embeddings for the descriptions
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        description_embeddings = vectorizer.fit_transform(movies_df['description'])

        # Check if the movie title exists in the dataset
        if movie_title not in movies_df['title'].values:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")

        # Get the index of the target movie
        idx = movies_df[movies_df['title'] == movie_title].index[0]

        # Calculate the cosine similarities between the target movie and all the other movies
        cosine_similarities = linear_kernel(description_embeddings[idx], description_embeddings).flatten()
        similar_indices = cosine_similarities.argsort()[-N-1:-1][::-1]
        similar_scores = cosine_similarities[similar_indices]

        # Retrieve the recommended movies
        recommended_movies = movies_df.iloc[similar_indices][['id', 'title', 'year', 'description']]
        recommended_movies['similarity_score'] = similar_scores

        # Return the improved recommendations if requested
        if improved:
            return ContentBasedFiltering.__get_improved_recommendations(
                recommended_movies, movies_df, N
            )[['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'wr']]
        else:
            return recommended_movies[['id', 'title', 'year', 'description']]


    @staticmethod
    def metadata_based_recommender(tabular_dataset_handler, movie_title, N, improved=False):
        """
        Recommends the N top most similar movies to the target movie based on movie metadata 
        (cast, crew, keywords, and genre).

        This function DOES NOT provide personalized recommendations based on the user.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of DatasetHandler.
            movie_title (str): The title of the movie.
            N (int): The number of top most similar movies to recommend.
            improved (bool): Whether to use improved recommendation using the weighted rating (wr)
                when True.

        Returns:
            pd.DataFrame: The N top most similar movies based on content.
        """
        # Initialize some brand-new copies of the required dataframes from dataset
        movies_df = tabular_dataset_handler.get_movies_df_deepcopy()
        credits_df = tabular_dataset_handler.get_credits_df_deepcopy()
        keywords_df = tabular_dataset_handler.get_keywords_df_deepcopy()

        # Add the columns 'cast', 'crew', 'keywords' (from the corresponding dataframes) to the 'movies_df'
        movies_df = movies_df.merge(credits_df, on='id')
        movies_df = movies_df.merge(keywords_df, on='id')

        # Add the columns 'cast_size' and 'crew_size' to the 'movies_df'
        movies_df['cast_size'] = movies_df['cast'].apply(lambda x: len(x))
        movies_df['crew_size'] = movies_df['crew'].apply(lambda x: len(x))

        # Add the column 'director' to the 'movies_df'
        movies_df['director'] = movies_df['crew'].apply(ContentBasedFiltering.__get_director)

        # Process the 'cast' column, for extracting only the names of the actors
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: [i['name'] for i in cast_row]
            if isinstance(cast_row, list) else []
        )
        # Process the 'cast' column, for extracting only the 3 top actors that appear in the credits list for each movie
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: cast_row[:3]
            if len(cast_row) >= 3 else cast_row
        )

        # Process the 'keywords' column, for extracting only the names of the keywords
        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda keyword_row: [i['name'] for i in keyword_row]
            if isinstance(keyword_row, list) else []
        )

        # Process the 'cast' column, for stripping spaces (so that NameSurname is recognised as the same actor)
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: [str.lower(actor.replace(" ", "")) for actor in cast_row]
        )

        # Process the 'director' column, for mentioning each director 3 times to give it more weight (w.r.t. to the entire cast)
        movies_df['director'] = movies_df['director'].astype('str').apply(
            lambda director: str.lower(director.replace(" ", ""))
        )
        movies_df['director'] = movies_df['director'].apply(
            lambda director: [director, director, director]
        )

        # Create a serie 'keyword_s' of movies' keywords
        keyword_s = (movies_df
                     .apply(lambda row: pd.Series(row['keywords']), axis=1)
                     .stack()
                     .reset_index(level=1, drop=True)
        )
        keyword_s.name = 'keyword'

        # Process the 'keyword_s' serie, by calculating the frequency counts of every keyword
        keyword_s = keyword_s.value_counts()

        # Remove keywords that occur only once
        keyword_s = keyword_s[keyword_s > 1]

        # Convert every keyword word to its stem (so that "similar" words are considered the same)
        stemmer = SnowballStemmer('english')

        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda keyword : ContentBasedFiltering.__filter_keywords(keyword, keyword_s)
        )
        movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])

        # Process the 'keywords' column, for stripping spaces
        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )

        # Add the column 'soup' to the 'movies_df', containing metadata dump for every movie
        movies_df['soup'] = (movies_df['keywords'] +
                             movies_df['cast'] +
                             movies_df['director'] +
                             movies_df['genres']
        )
        movies_df['soup'] = movies_df['soup'].apply(lambda x: ' '.join(x))

        # Create a TF-IDF vectorizer and generate embeddings for the soups
        vectorizer = TfidfVectorizer(stop_words='english')
        soup_embeddings = vectorizer.fit_transform(movies_df['soup'])

        # Check if the movie title exists in the dataset
        if movie_title not in movies_df['title'].values:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")

        # Get the index of the target movie
        idx = movies_df[movies_df['title'] == movie_title].index[0]

        # Calculate the cosine similarities between the target movie and all the other movies
        cosine_similarities = linear_kernel(soup_embeddings[idx], soup_embeddings).flatten()
        similar_indices = cosine_similarities.argsort()[-N-1:-1][::-1]
        similar_scores = cosine_similarities[similar_indices]

        # Retrieve the recommended movies
        recommended_movies = movies_df.iloc[similar_indices][['id', 'title', 'year', 'soup']]
        recommended_movies['similarity_score'] = similar_scores

        # Return the improved recommendations if requested
        if improved:
            return ContentBasedFiltering.__get_improved_recommendations(
                recommended_movies, movies_df, N
            )[['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'wr']]
        else:
            return recommended_movies[['id', 'title', 'year', 'soup']]

    @staticmethod
    def __get_improved_recommendations(recommended_movies_df, movies_df, N, percentile=0.60):
        """
        Returns improved recommendations by filtering bad movies and considering popularity and 
        critical response.

        To do that it calculates the top 25 movies based on similarity scores and calculate the i-th
        percentile movies, that is used as value m for calculating the weighted rating of each movie
        using the IMDb's formula.

        Parameters:
            recommended_movies_df (pd.DataFrame): DataFrame containing recommended movies.
            movies_df (pd.DataFrame): DataFrame containing movie data.
            N (int): Number of top most similar movies to recommend.
            percentile (float): Percentile value for calculating weighted rating.

        Returns:
            pd.DataFrame: Improved N most similar movies.

        """
        # Integrate columns in 'recommended_movies_df' from 'movies_df'
        enriched_df = recommended_movies_df.merge(
            movies_df[['id', 'genres', 'vote_count', 'vote_average', 'popularity']], 
            on='id', 
            how='left'
        )
        
        # Use _top_movies_IMDB_wr_formula to improve the recommendation
        return PopularityRanking._top_movies_IMDB_wr_formula(enriched_df, N, percentile)

    @staticmethod
    def __get_director(crew_row):
        """
        Returns the director from the 'crew' row.

        Parameters:
            crew_row (list): List of crew members for a movie.

        Returns:
            str or np.nan: Director's name or NaN if not found.
        """
        for x in crew_row:
            if x['job'] == 'Director':
                return x['name']

        return np.nan

    @staticmethod
    def __filter_keywords(keyword_s, keyword_list):
        """
        Models the keywords of a movie that are not present in 'keyword_s' and returns the 
        other ones.

        Parameters:
            keyword_s (pd.Series): Movie keywords.
            keyword_list (pd.Series): Frequency counts of every keyword.

        Returns:
            list: List of filtered keywords.
        """
        words = []

        for k in keyword_list:
            if k in keyword_s:
                words.append(k)

        return words
