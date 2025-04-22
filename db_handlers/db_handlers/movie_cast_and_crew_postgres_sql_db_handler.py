import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from db_handlers.utils import (
    is_docker,
    environment,
)


try:
    # If running in Airflow use the PostgresHook
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    AIRFLOW_AVAILABLE = "AIRFLOW_HOME" in os.environ
except ImportError:
    # Fallback to using psycopg2
    AIRFLOW_AVAILABLE = False

AIRFLOW_AVAILABLE = False  # TEMP: forces to avoid using Airflow Hooks

if AIRFLOW_AVAILABLE:
    print(f"[LOG] Detected Airflow environment, with Docker: [{is_docker()}]")
else:
    print(f"[LOG] Detected local environment, with Docker: [{is_docker()}]")
    
    import psycopg2
    from dotenv import load_dotenv

    # Dynamically find the project root (assumes .env is always in recsys)
    project_root = Path(__file__).resolve().parents[2]  # Move up two levels
    dotenv_path = project_root / ".env"  # Path to .env

    # Load environment variables from .env file
    load_dotenv(dotenv_path)


# Table names
MOVIES_METADATA_TABLE = "movies_metadata"
CAST_AND_CREW_TABLE = "cast_and_crew"


def get_db_connection():
    """Establishes and returns a PostgreSQL database connection, using Airflow's PostgresHook if available."""
    if AIRFLOW_AVAILABLE:
        # Note: Ensure 'postgres_default' is set up in Airflow Connections
        try:
            hook = PostgresHook(postgres_conn_id='postgres_default')
            return hook.get_conn()
        except Exception as e:
            print(f"[ERROR] Airflow PostgresHook failed: {e}")
            raise
    else:
        # Change host if using Docker networking
        host = os.getenv("DOCKER_POSTGRESQL_HOST") if is_docker() else os.getenv("POSTGRESQL_HOST")
        try:
            return psycopg2.connect(
                host=host,
                dbname=os.environ["POSTGRESQL_DBNAME"],
                user=os.environ["POSTGRESQL_USER"],
                password=os.environ["POSTGRESQL_PASSWORD"],
                port=os.environ["POSTGRESQL_PORT"],
            )
        except psycopg2.OperationalError as e:
            print(f"[ERROR] Database connection failed: {e}")
            raise


def create_movie_metadata_table() -> None:
    """Creates the 'movie_metadata' table in the database if it does not exist."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create the movies_metadata table
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {MOVIES_METADATA_TABLE} (
                id INTEGER PRIMARY KEY,
                adult BOOLEAN,
                belongs_to_collection TEXT,
                budget TEXT,
                genres JSONB,
                homepage TEXT,
                original_language TEXT,
                original_title TEXT,
                overview TEXT,
                popularity FLOAT,
                production_companies JSONB,
                production_countries JSONB,
                release_date TEXT,
                revenue FLOAT,
                runtime FLOAT,
                spoken_languages JSONB,
                status TEXT,
                tagline TEXT,
                title TEXT,
                vote_average FLOAT,
                vote_count FLOAT,
                year TEXT
            )
        ''')

    postgres_host = conn.get_dsn_parameters().get('host', 'unknown')
    print(f"[LOG] Table '{MOVIES_METADATA_TABLE}' created successfully in [{environment}] at URL: [{postgres_host}].")

    conn.commit()
    conn.close()


def create_cast_and_crew_table() -> None:
    """Creates the 'cast_and_crew' table in the database if it does not exist."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create the cast_and_crew table
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {CAST_AND_CREW_TABLE} (
                id SERIAL PRIMARY KEY,
                movie_id INTEGER UNIQUE REFERENCES {MOVIES_METADATA_TABLE}(id),
                cast_list JSONB,
                crew_list JSONB
            )
        ''')

    postgres_host = conn.get_dsn_parameters().get('host', 'unknown')
    print(f"[LOG] Table '{CAST_AND_CREW_TABLE}' created successfully in [{environment}] at URL: [{postgres_host}].")

    conn.commit()
    conn.close()


def reset_movie_cast_and_crew_tables() -> None:
    """Resets 'movies_metadata' and 'cast_and_crew' tables by dropping and recreating them."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Truncate both tables at the same time, cascading the operation
        cur.execute(f'TRUNCATE TABLE {MOVIES_METADATA_TABLE}, {CAST_AND_CREW_TABLE} RESTART IDENTITY CASCADE')
    conn.commit()
    conn.close()
    print(f"[LOG] Tables '{MOVIES_METADATA_TABLE}' and '{CAST_AND_CREW_TABLE}' reset successfully.")


def drop_movie_cast_and_crew_tables() -> None:
    """Drops 'movies_metadata' and 'cast_and_crew' tables if they exist."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS {CAST_AND_CREW_TABLE} CASCADE')
        cur.execute(f'DROP TABLE IF EXISTS {MOVIES_METADATA_TABLE} CASCADE')
    conn.commit()
    conn.close()
    print(f"[LOG] Tables '{CAST_AND_CREW_TABLE}' and '{MOVIES_METADATA_TABLE}' dropped successfully.")


def store_movielens_movies(movies_df: pd.DataFrame) -> None:
    """Stores movies metadata from a DataFrame into the database."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        inserted_count = 0
        error_count = 0
        for _, row in movies_df.iterrows():
            try:
                # Ensure JSON fields are valid JSON
                genres_json = json.dumps(row['genres']) if isinstance(row['genres'], (list, dict)) else None
                production_companies_json = json.dumps(row['production_companies']) if isinstance(row['production_companies'], (list, dict)) else None
                production_countries_json = json.dumps(row['production_countries']) if isinstance(row['production_countries'], (list, dict)) else None
                spoken_languages_json = json.dumps(row['spoken_languages']) if isinstance(row['spoken_languages'], (list, dict)) else None

                cur.execute(f'''
                    INSERT INTO {MOVIES_METADATA_TABLE} (id, adult, belongs_to_collection, budget, genres, homepage,
                    original_language, original_title, overview, popularity, production_companies, 
                    production_countries, release_date, revenue, runtime, spoken_languages, status, 
                    tagline, title, vote_average, vote_count, year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                ''', (
                    row['id'], row['adult'], row['belongs_to_collection'], row['budget'], genres_json, 
                    row['homepage'], row['original_language'], row['original_title'], 
                    row['overview'], row['popularity'], production_companies_json, 
                    production_countries_json, row['release_date'], row['revenue'], row['runtime'],
                    spoken_languages_json, row['status'], row['tagline'], row['title'], 
                    row['vote_average'], row['vote_count'], row['year']
                ))
                inserted_count += 1
            except Exception as e:
                print(f"[ERROR] Error inserting row {row['id']}: {e}")
                error_count += 1
    conn.commit()
    conn.close()
    print(f"[LOG] Stored {inserted_count} movies successfully, with {error_count} errors.")


def store_movielens_cast_and_crew(credits_df: pd.DataFrame) -> None:
    """Stores cast and crew data from a DataFrame into the database."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        inserted_count = 0
        error_count = 0
        for _, row in credits_df.iterrows():
            try:
                # Convert dict to JSON string
                cast_json = json.dumps(row['cast']) if isinstance(row['cast'], (list, dict)) else None
                crew_json = json.dumps(row['crew']) if isinstance(row['crew'], (list, dict)) else None

                cur.execute(f'''
                    INSERT INTO {CAST_AND_CREW_TABLE} (movie_id, cast_list, crew_list)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (movie_id) DO NOTHING
                ''', (row['id'], cast_json, crew_json))
                inserted_count += 1
            except Exception as e:
                print(f"[ERROR] Error inserting row {row['id']}: {e}")
                error_count += 1
    conn.commit()
    conn.close()
    print(f"[LOG] Stored {inserted_count} cast and crew records successfully, with {error_count} errors.")


def store_new_movie(params: Tuple[Any, ...]) -> None:
    """Stores a new movie record into the database."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'''
            INSERT INTO {MOVIES_METADATA_TABLE} (id, adult, belongs_to_collection, budget, genres, homepage,
            original_language, original_title, overview, popularity, production_companies, 
            production_countries, release_date, revenue, runtime, spoken_languages, status, 
            tagline, title, vote_average, vote_count, year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', params)
    conn.commit()
    conn.close()
    print(f"[LOG] New movie record stored successfully in '{MOVIES_METADATA_TABLE}' table.")


def store_new_cast_and_crew(params: Tuple[int, Any, Any]) -> None:
    """Stores a new cast and crew record into the database."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'''
            INSERT INTO {CAST_AND_CREW_TABLE} (movie_id, cast_list, crew_list)
            VALUES (%s, %s, %s)
        ''', params)
    
    conn.commit()
    conn.close()
    print(f"[LOG] New cast and crew record for movie_id {params[0]} stored successfully in '{CAST_AND_CREW_TABLE}' table.")


def get_movie_metadata(identifier: Any) -> Optional[Dict[str, Any]]:
    """Retrieves movie information by ID or title."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        if isinstance(identifier, int):
            cur.execute(f'SELECT * FROM {MOVIES_METADATA_TABLE} WHERE id = %s', (identifier,))
        else:
            cur.execute(f'SELECT * FROM {MOVIES_METADATA_TABLE} WHERE title = %s', (identifier,))
        
        result = cur.fetchone()
        if result:
            col_names = [desc[0] for desc in cur.description]
            result_dict = dict(zip(col_names, result))
        else:
            result_dict = None
            print(f"[LOG] Movie not found.")

    conn.close()

    return result_dict


def get_cast_and_crew_metadata(movie_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves cast and crew information for a given movie ID."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {CAST_AND_CREW_TABLE} WHERE movie_id = %s', (movie_id,))
    
        result = cur.fetchone()
        if result:
            col_names = [desc[0] for desc in cur.description]
            result_dict = dict(zip(col_names, result))
        else:
            result_dict = None
            print(f"[LOG] Cast and crew not found.")

    conn.close()

    return result_dict


def get_all_movies() -> pd.DataFrame:
    """Retrieves all movies from the database."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {MOVIES_METADATA_TABLE}')
        result = cur.fetchall()
    
    conn.close()

    col_names = [desc[0] for desc in cur.description]
    movies_df = pd.DataFrame(result, columns=col_names)

    return movies_df


def get_all_cast_and_crew() -> pd.DataFrame:
    """Retrieves all cast and crew from the database."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {CAST_AND_CREW_TABLE}')
        result = cur.fetchall()
    
    conn.close()

    col_names = [desc[0] for desc in cur.description]
    cast_and_crew_df = pd.DataFrame(result, columns=col_names)

    return cast_and_crew_df


def get_joined_movies_and_cast_and_crew() -> pd.DataFrame:
    """Retrieves all movies and their corresponding cast and crew from the database."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'''
            SELECT m.*, c.cast_list, c.crew_list
            FROM {MOVIES_METADATA_TABLE} m
            LEFT JOIN {CAST_AND_CREW_TABLE} c ON m.id = c.movie_id
        ''')
        result = cur.fetchall()
    
    conn.close()

    col_names = [desc[0] for desc in cur.description]
    joined_df = pd.DataFrame(result, columns=col_names)

    return joined_df


def get_movie_ids() -> List[int]:
    """Retrieves all movie IDs from the database."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'SELECT id FROM {MOVIES_METADATA_TABLE}')
        result = cur.fetchall()
    
    conn.close()
    
    movie_ids = [row[0] for row in result]
    return movie_ids