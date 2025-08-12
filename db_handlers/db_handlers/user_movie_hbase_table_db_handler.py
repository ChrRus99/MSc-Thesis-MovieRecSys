import os
import pandas as pd
from pathlib import Path

from db_handlers.utils import (
    is_docker,
    environment,
)


try:
    # If running in Airflow use
    # ---> there is no pre-built HBASE hook for Airflow
    # TODO: see https://registry.astronomer.io/

    AIRFLOW_AVAILABLE = "AIRFLOW_HOME" in os.environ
except ImportError:
    # Fallback to using psycopg2
    AIRFLOW_AVAILABLE = False

AIRFLOW_AVAILABLE = False  # TEMP: forces to avoid using Airflow Hooks

if AIRFLOW_AVAILABLE:
    print(f"[LOG] Detected Airflow environment, with Docker: [{is_docker()}]")
else:
    print(f"[LOG] Detected local environment, with Docker: [{is_docker()}]")
    
    import happybase
    from dotenv import load_dotenv

    # Dynamically find the project root (assumes .env is always in recsys)
    project_root = Path(__file__).resolve().parents[2]  # Move up two levels
    dotenv_path = project_root / ".env"  # Path to .env

    # Load environment variables from .env file
    load_dotenv(dotenv_path)


# HBase table and column family names
TABLE_NAME = "user_movie_ratings"
COLUMN_FAMILY = "ratings"  # HBase requires at least one column family


def get_db_connection():
    """Establishes and returns a connection to HBase."""
    # Change host if using Docker networking
    host = os.environ.get("DOCKER_HBASE_HOST") if is_docker() else os.environ.get("HBASE_HOST")

    return happybase.Connection(host)


def create_user_movie_ratings_table():
    """Creates the HBase table for storing user-movie rating predictions."""
    conn = get_db_connection()

    tables = conn.tables()
    if TABLE_NAME.encode() in tables:
        print(f"[LOG] Table '{TABLE_NAME}' already exists.")
        return
    conn.create_table(TABLE_NAME, {COLUMN_FAMILY: dict()})

    hbase_host = conn.host
    print(f"[LOG] Table '{TABLE_NAME}' created successfully in [{environment}] at URL: [{hbase_host}].")

    conn.close()

# def create_user_movie_ratings_table():
#     """Creates or overrides the HBase table for storing user-movie rating predictions."""
#     conn = get_db_connection()

#     tables = conn.tables()
#     if TABLE_NAME.encode() in tables:
#         print(f"[LOG] Table '{TABLE_NAME}' already exists. Overriding it.")
#         try:
#             # Check if table is enabled before disabling
#             if conn.is_table_enabled(TABLE_NAME):
#                 conn.disable_table(TABLE_NAME)
#                 print(f"[LOG] Table '{TABLE_NAME}' disabled successfully.")
#             else:
#                 print(f"[LOG] Table '{TABLE_NAME}' was already disabled.")

#             conn.delete_table(TABLE_NAME)
#             print(f"[LOG] Table '{TABLE_NAME}' deleted successfully.")
#         except Exception as e:
#             print(f"[ERROR] Error deleting existing table: {e}")
#             conn.close()
#             return  # Exit the function if there's an error deleting the table.

#     try:
#         conn.create_table(TABLE_NAME, {COLUMN_FAMILY: dict()})
#         print(f"[LOG] Table '{TABLE_NAME}' created successfully.")
#     except Exception as e:
#         print(f"[ERROR] Error creating table: {e}")

#     conn.close()

def delete_all_hbase_tables():
    """Deletes the table specified by TABLE_NAME if it exists."""
    conn = get_db_connection()
    tables = conn.tables()

    if not tables:
        print("No HBase tables found.")
        conn.close()
        return

    print("Deleting all HBase tables...")

    for table_name_bytes in tables:
        table_name = table_name_bytes.decode()  # Decode table name from bytes
        try:
            if conn.is_table_enabled(table_name):
                conn.disable_table(table_name)
                print(f"[LOG] Table '{table_name}' disabled successfully.")
            else:
                print(f"[LOG] Table '{table_name}' was already disabled.")

            conn.delete_table(table_name)
            print(f"[LOG] Table '{table_name}' deleted successfully.")
        except Exception as e:
            print(f"[ERROR] Error deleting table '{table_name}': {e}")

    print("Deletion process completed.")
    conn.close()

def drop_user_movie_ratings_table():
    """Drops the user-movie ratings table if it exists."""
    conn = get_db_connection()

    if TABLE_NAME.encode() in conn.tables():
        conn.disable_table(TABLE_NAME)
        conn.delete_table(TABLE_NAME)
        print(f"[LOG] Table '{TABLE_NAME}' dropped.")
    else:
        print(f"[ERROR] Table '{TABLE_NAME}' does not exist.")
    
    conn.close()

def reset_user_movie_ratings_table():
    """Drops and recreates the user-movie ratings table."""
    drop_user_movie_ratings_table()
    create_user_movie_ratings_table()


def store_user_movie_prediction(user_id: int, movie_id: int, rating: float):
    """
    Stores or updates a user-movie rating prediction.
    
    Args:
        user_id (int): The user ID.
        movie_id (int): The movie ID.
        rating (float): The predicted rating.
    """
    conn = get_db_connection()

    table = conn.table(TABLE_NAME)
    row_key = f"user_{user_id}"  # Row key for HBase
    column = f"{COLUMN_FAMILY}:movie_{movie_id}"  # Column qualifier
    table.put(row_key, {column.encode(): str(rating).encode()})
    print(f"[LOG] Stored rating {rating} for user {user_id} and movie {movie_id}.")

    conn.close()


def store_user_movie_predictions(predictions_df):
    """
    Stores multiple user-movie rating predictions in batch.
    
    Args:
        predictions_df (DataFrame): A DataFrame with columns ["userId", "movieId", "rating"].
    """
    conn = get_db_connection()

    table = conn.table(TABLE_NAME)
    batch = table.batch()
    for idx, row in predictions_df.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        row_key = f"user_{user_id}"
        column = f"{COLUMN_FAMILY}:movie_{movie_id}"
        batch.put(row_key, {column.encode(): str(rating).encode()})
    batch.send()
    print(f"[LOG] Stored {len(predictions_df)} predictions successfully.")
    
    conn.close()


def get_user_movie_prediction(user_id: int, movie_id: int) -> float:
    """
    Retrieves a specific movie rating prediction for a given user.
    
    Args:
        user_id (int): The user ID.
        movie_id (int): The movie ID.

    Returns:
        float: The predicted rating, or None if not found.
    """
    conn = get_db_connection()
    table = conn.table(TABLE_NAME)
    row_key = f"user_{user_id}"
    column = f"{COLUMN_FAMILY}:movie_{movie_id}".encode()
    row = table.row(row_key)
    conn.close()
    
    if column in row:
        return float(row[column].decode())
    return None


def get_all_user_predictions(user_id: int) -> pd.DataFrame:
    """
    Retrieves all movie rating predictions for a given user.
    
    Args:
        user_id (int): The user ID.

    Returns:
        pd.DataFrame: A DataFrame with columns ["userId", "movieId", "rating"].
    """
    conn = get_db_connection()

    table = conn.table(TABLE_NAME)
    row_key = f"user_{user_id}"  # Row key
    row = table.row(row_key)

    conn.close()
    
    predictions = [{
        'userId': user_id,
        'movieId': int(col.decode().split(':')[1].replace('movie_', '')),
        'rating': float(val.decode())
    } for col, val in row.items()]
    return pd.DataFrame(predictions)


def get_user_ids() -> list[int]:
    """
    Retrieves all unique user IDs stored in the HBase table.
    
    Returns:
        list: A list of user IDs.
    """
    conn = get_db_connection()
    table = conn.table(TABLE_NAME)
    user_ids = [key.decode().split('_')[1] for key, _ in table.scan()]
    conn.close()
    
    return user_ids