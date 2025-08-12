from pathlib import Path
from typing import List, Dict, Tuple, Optional


def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

environment = "docker" if is_docker() else "local"


def group_ratings_by_user(ratings: List[Tuple[str, str, int, int]]) -> Dict[str, List[Tuple[str, str, int, int]]]:
    """
    Groups a list of ratings by user_id.
    
    Args:
        ratings (List[Tuple[str, str, int, int]]): List of tuples containing (user_id, movie_id, 
            rating, timestamp).
    
    Returns:
        Dict[str, List[Tuple[str, str, int, int]]]: Dictionary where the key is user_id and the 
            value is a list of tuples (user_id, movie_id, rating, timestamp).
    """
    grouped_ratings = {}

    for rating in ratings:
        user_id = rating[0]
        if user_id not in grouped_ratings:
            grouped_ratings[user_id] = []
        grouped_ratings[user_id].append(rating)
    
    return grouped_ratings


def remove_lucene_chars(text: str) -> str:
    """ Removes Lucene special characters """
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


def generate_full_text_query(input: str) -> str:
    """
    Generates a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a similarity threshold 
    (~0.8) to each word, then combines them using the AND operator. Useful for mapping movies and
    people from user questions to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]

    for word in words[:-1]:
        full_text_query += f" {word}~0.8 AND"
    full_text_query += f" {words[-1]}~0.8"
    
    return full_text_query.strip()