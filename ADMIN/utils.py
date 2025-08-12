import pandas as pd
from difflib import SequenceMatcher

def is_title_present(movies_df: pd.DataFrame, title: str, max_errors: int=3) -> tuple:
    """
    Determines if a given title is present in the movies DataFrame, allowing for a specified number
    of character-level errors.

    Args:
        movies_df (pd.DataFrame): A DataFrame containing a 'title' column with movie titles.
        title (str): The title to search for in the DataFrame.
        max_errors (int): The maximum number of character-level differences allowed for a match.

    Returns:
        tuple: A tuple where the first element is a boolean indicating if an exact match was found,
            and the second element is the best matching title if no exact match is found, or None
            if no match exists.
    """
    def calculate_similarity(str1, str2):
        # Calculate the similarity ratio
        return SequenceMatcher(None, str1, str2).ratio()

    best_match = None
    highest_similarity = 0

    for movie_title in movies_df['title']:
        similarity_ratio = calculate_similarity(movie_title, title)
        if similarity_ratio > highest_similarity:
            highest_similarity = similarity_ratio
            best_match = movie_title

    # Check if the best match meets the max_errors threshold
    max_allowed_similarity = 1 - (max_errors / max(len(title), len(best_match or "")))
    if highest_similarity >= max_allowed_similarity:
        if best_match == title:
            return True, title
        else:
            return False, best_match
    else:
        return False, None

