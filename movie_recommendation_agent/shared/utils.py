import csv
import os
from datetime import datetime
from typing import Any, Annotated, Optional, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


USERS_REGISTER_CSV_FILE_PATH = os.path.abspath("users_register.csv")
RATINGS_CSV_FILE_PATH  = os.path.abspath("users_movie_ratings.csv")


def is_user_registered(user_id: str) -> bool:
    """Check if a user is already registered.

    Args:
        user_id (str): Unique ID of the user.

    Returns:
        bool: True if the user is registered, False otherwise.
    """
    if not os.path.isfile(USERS_REGISTER_CSV_FILE_PATH):
        return False

    with open(USERS_REGISTER_CSV_FILE_PATH, mode='r') as file:
        # Attempt to read headers
        try:
            reader = csv.DictReader(file)
        except csv.Error:
            # Fallback in case the file does not have headers
            reader = csv.reader(file)
            fieldnames = ["user_id", "first_name", "last_name", "email", "registered_at"]
            # Skip the first line if it's just data without headers
            next(reader, None)  # skip the header

        for row in reader:
            if str(row.get("user_id") or row[0]) == str(user_id):
                return True
    return False


def save_user_info(user_data: dict) -> None:
    """Save user information to a CSV file.

    Args:
        user_data (dict): Dictionary containing user information with keys:
                          "user_id", "first_name", "last_name", "email", "registered_at".
    """
    # Check if the file exists, if not, create it with the header
    if not os.path.isfile(USERS_REGISTER_CSV_FILE_PATH):
        with open(USERS_REGISTER_CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "first_name", "last_name", "email", "registered_at"])

    # Append the user information to the CSV file
    with open(USERS_REGISTER_CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            user_data.get("user_id"),
            user_data.get("first_name"),
            user_data.get("last_name"),
            user_data.get("email"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


def load_user_info(user_id: str) -> Optional[Dict]:
    """Load user information from a CSV file.

    Args:
        user_id (str): The unique ID of the user.

    Returns:
        Optional[dict]: A dictionary containing user information if the user exists, 
                        or None if no matching user ID is found.
    """
    if not os.path.isfile(USERS_REGISTER_CSV_FILE_PATH):
        return None  # Return None if the file doesn't exist

    try:
        with open(USERS_REGISTER_CSV_FILE_PATH, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["user_id"] == user_id:
                    # Return the matching user as a dictionary
                    return {
                        "user_id": row["user_id"],
                        "first_name": row["first_name"],
                        "last_name": row["last_name"],
                        "email": row["email"],
                        "registered_at": row["registered_at"]
                    }
    except (FileNotFoundError, csv.Error) as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return None  # Return None if no match is found


def save_user_seen_movies(user_id: str, movies: List[Dict]) -> None:
    """Save user seen movies information to a CSV file.
    
    This function saves user seen movies information to a CSV file, overwriting if the combination 
    of user_id and movie_id already exists.

    Args:
        user_id (str): Unique ID of the user.
        movies (list): List of movies seen by the user, each represented as a dict with keys 
                       "movie_id" and "rating".
    """
    # Check if the file exists, and if not, create it with headers
    if not os.path.isfile(RATINGS_CSV_FILE_PATH):
        with open(RATINGS_CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "movie_id", "rating", "timestamp"])

    # Read existing data from the CSV file
    rows = []
    with open(RATINGS_CSV_FILE_PATH, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        rows = list(reader)    # Read the rest of the file into memory

    # Prepare new data to write
    for movie in movies:
        found = False
        for i, row in enumerate(rows):
            # Compare user_id and movie_id as strings (without conversion)
            if row[0] == user_id and row[1] == movie["movie_id"]:
                # Update existing entry
                rows[i] = [
                    user_id,
                    movie["movie_id"],
                    movie["rating"],
                    int(datetime.now().timestamp())
                ]
                found = True
                break

        if not found:
            # If the entry doesn't exist, add it as a new row
            rows.append([
                user_id,
                movie["movie_id"],
                movie["rating"],
                int(datetime.now().timestamp())
            ])

    # Write all data back to the file
    with open(RATINGS_CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "movie_id", "rating", "timestamp"])
        writer.writerows(rows)


def load_user_seen_movies(user_id: str) -> List[Dict[str, str]]:
    """Load movies seen by a specific user from a CSV file.

    Args:
        user_id (str): The unique ID of the user.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the movie information.
                              Each dictionary has keys "user_id", "movie_id", "rating", "timestamp".
    """
    if not os.path.isfile(RATINGS_CSV_FILE_PATH):
        return []

    movies_seen = []

    try:
        with open(RATINGS_CSV_FILE_PATH, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["user_id"] == user_id:
                    movies_seen.append({
                        "user_id": row["user_id"],
                        "movie_id": int(row["movie_id"]),
                        "rating": row["rating"],
                        "timestamp": row["timestamp"]
                    })
    except (FileNotFoundError, csv.Error) as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return movies_seen


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)