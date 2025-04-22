import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from db_handlers.utils import (
    is_docker,
    environment,
)


try:
    # If running in Airflow use the MongoHook
    from airflow.providers.mongo.hooks.mongo import MongoHook
    AIRFLOW_AVAILABLE = "AIRFLOW_HOME" in os.environ
except ImportError:
    # Fallback to using pymongo
    AIRFLOW_AVAILABLE = False

AIRFLOW_AVAILABLE = False  # TEMP: forces to avoid using Airflow Hooks

if AIRFLOW_AVAILABLE:
    print(f"[LOG] Detected Airflow environment, with Docker: [{is_docker()}]")
else:   
    print(f"[LOG] Detected local environment, with Docker: [{is_docker()}]")

    from dotenv import load_dotenv
    from pymongo import MongoClient

    # Dynamically find the project root (assumes .env is always in recsys)
    project_root = Path(__file__).resolve().parents[2]  # Move up two levels
    dotenv_path = project_root / ".env"  # Path to .env

    # Load environment variables from .env file
    load_dotenv(dotenv_path)


# Collection names
USER_MOVIE_RATINGS_COLLECTION = "user_movie_ratings"
USER_PREFERENCES_COLLECTION = "user_preferences"


def get_db_connection():
    """Establishes and returns a MongoDB database connection, using Airflow's MongoHook if available."""
    if AIRFLOW_AVAILABLE:
        # Note: Ensure 'mongo_default' is set up in Airflow Connections
        hook = MongoHook(conn_id='mongo_default')
        client = hook.get_conn()
    else:
        # Change host if using Docker networking
        os.environ["MONGODB_URI"] = os.environ["DOCKER_MONGODB_URI"] if is_docker() else os.environ["MONGODB_URI"]
        client = MongoClient(os.environ["MONGODB_URI"])
    
    db = client["movie_app"]
    return client, db


def create_user_movie_ratings_collection() -> None:
    """Creates 'user_movie_ratings' collection if it does not exist"""
    client, db = get_db_connection()
    
    # Create 'user_movie_ratings' collection if it does not exist
    if USER_MOVIE_RATINGS_COLLECTION not in db.list_collection_names():
        db.create_collection(USER_MOVIE_RATINGS_COLLECTION)

    mongodb_url = client.address
    print(f"[LOG] Collection '{USER_MOVIE_RATINGS_COLLECTION}' created successfully in [{environment}] at URL: [{mongodb_url}].")


def create_user_preferences_collection() -> None:
    """Creates 'user_preferences' collection if it does not exist"""
    client, db = get_db_connection()

    # Create 'user_preferences' collection if it does not exist
    if USER_PREFERENCES_COLLECTION not in db.list_collection_names():
        db.create_collection(USER_PREFERENCES_COLLECTION)

    mongodb_url = client.address
    print(f"[LOG] Collection '{USER_PREFERENCES_COLLECTION}' created successfully in [{environment}] at URL: [{mongodb_url}].")


def reset_user_collections() -> None:
    """Removes all documents from a collection while keeping the structure."""
    client, db = get_db_connection()
    db[USER_MOVIE_RATINGS_COLLECTION].delete_many({})
    db[USER_PREFERENCES_COLLECTION].delete_many({})
    client.close()
    print(f"[LOG] Collections '{USER_MOVIE_RATINGS_COLLECTION}' and '{USER_PREFERENCES_COLLECTION}' reset successfully.")


def drop_user_collections() -> None:
    """Deletes an entire collection from the database."""
    client, db = get_db_connection()
    db[USER_MOVIE_RATINGS_COLLECTION].drop()
    db[USER_PREFERENCES_COLLECTION].drop()
    client.close()
    print(f"[LOG] Collections '{USER_MOVIE_RATINGS_COLLECTION}' and '{USER_PREFERENCES_COLLECTION}' dropped successfully.")


def store_user_movie_rating(user_id: str, movie_id: str, rating: int, timestamp: int=-1) -> None:
    """
    Stores or updates a user's movie rating in MongoDB.
    If the movie already has a rating from the user, this function updates the rating and the
    timestamp avoiding creating duplicated user-movie ratings. 
    """
    client, db = get_db_connection()
    ratings_collection = db[USER_MOVIE_RATINGS_COLLECTION]
    
    # Find if the user has already rated the movie
    existing_rating = ratings_collection.find_one(
        {"user_id": user_id, "ratings.movie_id": movie_id},
        {"ratings.$": 1}  # Only retrieve the matched movie rating
    )

    if timestamp < 0:
        timestamp = int(time.time())

    if existing_rating:
        # Update the rating and timestamp
        ratings_collection.update_one(
            {"user_id": user_id, "ratings.movie_id": movie_id},
            {"$set": {"ratings.$.rating": rating, "ratings.$.timestamp": timestamp}}
        )
    else:
        # Insert new rating if it doesn't exist
        ratings_collection.update_one(
            {"user_id": user_id},
            {"$push": {"ratings": {"movie_id": movie_id, "rating": rating, "timestamp": timestamp}}},
            upsert=True
        )

    client.close()
    print(f"[LOG] New user rating stored successfully in '{USER_MOVIE_RATINGS_COLLECTION}' collection.")


def store_user_preference(user_id: str, preference: str, timestamp: int=-1) -> None:
    """
    Stores or updates a user preference in MongoDB.
    If the preference already exists, this function updates its timestamp instead of inserting a
    duplicate.
    """
    client, db = get_db_connection()
    preferences_collection = db[USER_PREFERENCES_COLLECTION]
    
    # Find if the preference already exists for the user
    existing_preference = preferences_collection.find_one(
        {"user_id": user_id, "preferences.preference": preference},
        {"preferences.$": 1}  # Only retrieve the matched preference
    )

    if timestamp < 0:
        timestamp = int(time.time())

    if existing_preference:
        # Update the timestamp of the existing preference
        preferences_collection.update_one(
            {"user_id": user_id, "preferences.preference": preference},
            {"$set": {"preferences.$.timestamp": timestamp}}
        )
    else:
        # Insert new preference if it doesn't exist
        preferences_collection.update_one(
            {"user_id": user_id},
            {"$push": {"preferences": {"preference": preference, "timestamp": timestamp}}},
            upsert=True
        )

    client.close()
    print(f"[LOG] New user preference stored successfully in '{USER_PREFERENCES_COLLECTION}' collection.")


def get_user_movie_ratings(user_id: str, after_timestamp: int=-1, last_k: int=-1) -> List[Tuple[str, str, int, int]]:
    """
    Retrieves movie ratings of a user as a list of tuples (user_id, movie_id, rating, timestamp).
    Filtering is applied in two stages:
        1. If after_timestamp > 0, only ratings with a timestamp greater than after_timestamp are 
            returned.
        2. If last_k > 0, only the last k ratings (i.e., the most recent ones) from the filtered 
            list are returned.
    If both filters are inactive (<= 0), all user's ratings are returned.
    """
    client, db = get_db_connection()
    ratings_collection = db[USER_MOVIE_RATINGS_COLLECTION]

    # Retrieve user's ratings (if they exist)
    user_ratings = ratings_collection.find_one({"user_id": user_id})

    client.close()

    if user_ratings and "ratings" in user_ratings:
        ratings = user_ratings["ratings"]

        # Filter by after_timestamp if specified
        if after_timestamp > 0:
            ratings = [r for r in ratings if r["timestamp"] > after_timestamp]

        # Sort ratings by timestamp in ascending order
        ratings = sorted(ratings, key=lambda r: r["timestamp"])

        # Filter by last_k if specified (take the most recent last_k ratings)
        if last_k > 0:
            ratings = ratings[-last_k:]

        return [(user_id, r["movie_id"], r["rating"], r["timestamp"]) for r in ratings]

    return []


def get_user_preferences(user_id: str) -> List[Tuple[str, str, int]]:
    """Retrieves all preferences of a user as a list of tuples (user_id, preference, timestamp)."""
    client, db = get_db_connection()
    preferences_collection = db[USER_PREFERENCES_COLLECTION]
    
    # Retrieve user's preferences (if they exist)
    user_pref = preferences_collection.find_one({"user_id": user_id})

    client.close()

    if user_pref and "preferences" in user_pref:
        return [(user_id, p["preference"], p["timestamp"]) for p in user_pref["preferences"]]
    return []


def get_all_last_user_movie_ratings(after_timestamp: int = -1) -> List[Tuple[str, str, int, int]]:
    """
    Retrieves all movie ratings from all users as a list of tuples (user_id, movie_id, rating, 
    timestamp).
    Only ratings after the specified timestamp are returned. If after_timestamp < 0, all ratings are
    returned.
    """
    client, db = get_db_connection()
    ratings_collection = db[USER_MOVIE_RATINGS_COLLECTION]
    
    # Query to filter ratings by timestamp
    if after_timestamp < 0:
        # Retrieve all ratings
        query = {}
    else:
        # Only retrieve ratings after the specified timestamp
        query = {"ratings.timestamp": {"$gt": after_timestamp}}
    
    # Retrieve all user ratings matching the query
    all_ratings = ratings_collection.find(query)
    
    result = []
    for user_ratings in all_ratings:
        user_id = user_ratings["user_id"]
        for rating in user_ratings["ratings"]:
            if after_timestamp < 0 or rating["timestamp"] > after_timestamp:
                result.append((user_id, rating["movie_id"], rating["rating"], rating["timestamp"]))
    
    client.close()
    return result
