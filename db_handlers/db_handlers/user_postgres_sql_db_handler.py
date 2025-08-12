import os
import pandas as pd
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
USER_REGISTER_TABLE = "user_register"
USER_MODEL_LOOKUP_TABLE = "user_model_lookup"
# LANGGRAPH_THREADS_TABLE = "langgraph_threads"

MIN_USER_MODEL_ID = 1000
MAX_USER_MODEL_ID = 1000000


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


def create_user_register_table() -> None:
    """Creates the 'user_register' table in the database if it does not exist."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create the user register table
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {USER_REGISTER_TABLE} (
                user_id UUID PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                email TEXT UNIQUE,
                timestamp BIGINT
            )
        ''')       

    postgres_host = conn.get_dsn_parameters().get('host', 'unknown')
    print(f"[LOG] Table '{USER_REGISTER_TABLE}' created successfully in [{environment}] at URL: [{postgres_host}].")

    conn.commit()
    conn.close()


def create_user_model_lookup_table():
    """Creates the user-model lookup table if it does not exist."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {USER_MODEL_LOOKUP_TABLE} (
                user_id UUID PRIMARY KEY,
                model_id INT UNIQUE NOT NULL
            )
        ''')

    postgres_host = conn.get_dsn_parameters().get('host', 'unknown')
    print(f"[LOG] Table '{USER_MODEL_LOOKUP_TABLE}' created successfully in [{environment}] at URL: [{postgres_host}].")

    conn.commit()
    conn.close()


def create_langgraph_threads_table() -> None:
    # TODO: see langgraph inmemory storage postgres: https://www.youtube.com/watch?v=hE8C2M8GRLo
    pass
    

def reset_user_tables() -> None:
    """Resets the 'user_register' and 'langgraph_threads' tables by dropping and recreating them."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Truncate both tables at the same time, cascading the operation
        cur.execute(f'TRUNCATE TABLE {USER_REGISTER_TABLE} RESTART IDENTITY CASCADE')
        cur.execute(f'TRUNCATE TABLE {USER_MODEL_LOOKUP_TABLE} RESTART IDENTITY CASCADE')
    conn.commit()
    conn.close()
    print(f"[LOG] Tables '{USER_REGISTER_TABLE}' and '{USER_MODEL_LOOKUP_TABLE}' reset successfully.")


def drop_user_tables() -> None:
    """Drops 'user_register' table if it exists."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS {USER_REGISTER_TABLE} CASCADE')
        cur.execute(f'DROP TABLE IF EXISTS {USER_MODEL_LOOKUP_TABLE} CASCADE')
    conn.commit()
    conn.close()
    print(f"[LOG] Tables '{USER_REGISTER_TABLE}' and '{USER_MODEL_LOOKUP_TABLE}' dropped successfully.")


def store_new_user(user_id: uuid, first_name: str, last_name: str, email: str, timestamp: int=-1) -> None:
    """Stores a new user in the database or update the existing user if needed.
       If the user already exists with identical data, do nothing.
       If an email conflict occurs, handle it gracefully.
    """
    conn = get_db_connection()
    
    if timestamp < 0:
        timestamp = int(time.time())

    try:
        # Store user info
        with conn.cursor() as cur:
            cur.execute(f'''
                INSERT INTO {USER_REGISTER_TABLE} (user_id, first_name, last_name, email, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    email = EXCLUDED.email,
                    timestamp = EXCLUDED.timestamp
            ''', (user_id, first_name, last_name, email, timestamp))
        
        # Store the user model ID
        model_id = store_new_user_model_id(user_id)

        conn.commit()
        print(f"[LOG] New user {user_id} stored successfully in '{USER_REGISTER_TABLE}' table")
    except psycopg2.errors.UniqueViolation as e:
        # Handle duplicate email error
        if 'user_register_email_key' in str(e):
            print(f"[LOG] User with email {email} already exists. Skipping insertion.")
        else:
            raise  # Re-raise unexpected UniqueViolation errors
    finally:
        conn.close()


def store_new_user_model_id(user_id: uuid.UUID) -> int:
    """Store a new user model ID in the lookup table and return the assigned model ID.
       If the user already has a model ID, do nothing and return the existing model ID."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Check if the user already has a model ID
        cur.execute(f'''
            SELECT model_id FROM {USER_MODEL_LOOKUP_TABLE} WHERE user_id = %s
        ''', (user_id,))
        existing_model_id = cur.fetchone()
        
        if existing_model_id:
            # User already exists in the table, return the existing model ID
            print(f"[LOG] User {user_id} already has an associated model ID {existing_model_id[0]}. Skipping insertion.")
            conn.close()
            return existing_model_id[0]
        
        # Find the next available model ID (starting from MIN_USER_MODEL_ID)
        cur.execute(f'''
            SELECT COALESCE(MAX(model_id), %s) FROM {USER_MODEL_LOOKUP_TABLE}
        ''', (MIN_USER_MODEL_ID - 1,))  # Start from MIN_USER_MODEL_ID
        next_model_id = cur.fetchone()[0] + 1
        
        # Ensure the next model ID is within bounds
        if next_model_id > MAX_USER_MODEL_ID:
            raise ValueError(f"Model ID exceeds maximum allowed ID: {MAX_USER_MODEL_ID}")
        
        # Insert new user model ID
        cur.execute(f'''
            INSERT INTO {USER_MODEL_LOOKUP_TABLE} (user_id, model_id)
            VALUES (%s, %s)
        ''', (user_id, next_model_id))
        
        print(f"[LOG] Assigned new model ID {next_model_id} to user {user_id} in '{USER_MODEL_LOOKUP_TABLE}' table.")
    
    conn.commit()
    conn.close()
    return next_model_id


def is_user_registered(user_id: str) -> bool:
    """Checks if a user is registered in the database."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'''
            SELECT EXISTS(SELECT 1 FROM {USER_REGISTER_TABLE} WHERE user_id = %s)
        ''', (user_id,))
        exists = cur.fetchone()[0]
    
    conn.close()
    return exists


def get_user_info(user_id: str) -> Optional[Dict[str, str]]:
    """Retrieves user information as a dictionary."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute('''
            SELECT user_id, first_name, last_name, email, timestamp FROM user_register WHERE user_id = %s
        ''', (user_id,))
        row = cur.fetchone()
    
    conn.close()

    if row:
        return {
            "user_id": row[0],
            "first_name": row[1],
            "last_name": row[2],
            "email": row[3],
            "timestamp": row[4]
        }
    return None


def get_user_model_id(user_id: uuid.UUID) -> Optional[int]:
    """Retrieve the model ID for a given user."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'''
            SELECT model_id FROM {USER_MODEL_LOOKUP_TABLE} WHERE user_id = %s
        ''', (user_id,))
        row = cur.fetchone()
    
    conn.close()

    if row:
        return row[0]
    return None


def get_user_ids() -> List[str]:
    """Retrieves all user IDs from the user_register table."""
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute(f'''
            SELECT user_id FROM {USER_REGISTER_TABLE}
        ''')
        rows = cur.fetchall()
    
    conn.close()

    return [row[0] for row in rows]


def get_user_model_ids() -> List[Dict[str, int]]:
    """Retrieve all user-model ID mappings."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(f'SELECT user_id, model_id FROM {USER_MODEL_LOOKUP_TABLE}')
        rows = cur.fetchall()
    
    conn.close()
    return [{"user_id": row[0], "model_id": row[1]} for row in rows]