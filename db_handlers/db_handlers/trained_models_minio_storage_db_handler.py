import os
from pathlib import Path

from db_handlers.utils import (
    is_docker,
    environment,
)

try:
    # If running in Airflow use
    # ---> there is no pre-built MinIO hook for Airflow
    # TODO: see https://blog.min.io/apache-airflow-minio/

    AIRFLOW_AVAILABLE = "AIRFLOW_HOME" in os.environ
except ImportError:
    # Fallback to using psycopg2
    AIRFLOW_AVAILABLE = False

AIRFLOW_AVAILABLE = False  # TEMP: forces to avoid using Airflow Hooks

if AIRFLOW_AVAILABLE:
    print(f"[LOG] Detected Airflow environment, with Docker: [{is_docker()}]")
else:
    print(f"[LOG] Detected local environment, with Docker: [{is_docker()}]")
    
    from minio import Minio
    from minio.error import S3Error
    from dotenv import load_dotenv

    # Dynamically find the project root (assumes .env is always in recsys)
    project_root = Path(__file__).resolve().parents[2]  # Move up two levels
    dotenv_path = project_root / ".env"  # Path to .env

    # Load environment variables from .env file
    load_dotenv(dotenv_path)


# Bucket Names
INITIAL_MODEL_BUCKET = "initial-model"                              # Stores the first pre-trained model before any updates
OFFLINE_UPDATED_MODELS_BUCKET = "offline-updated-models"            # Stores periodically updated offline models (incremental learning or continuous training)
ONLINE_UPDATED_USER_MODELS_BUCKET = "online-updated-user-models"    # Stores user-specific online updated models


def get_db_connection():
    """Initialize and return a MinIO client."""
    return Minio(
        os.environ.get("MINIO_ENDPOINT"),
        access_key=os.environ.get("MINIO_ACCESS_KEY"),
        secret_key=os.environ.get("MINIO_SECRET_KEY"),
        secure=os.environ.get("MINIO_SECURE", "False").lower() == "true"
    )


def bucket_exists(client, bucket_name):
    """Check if a bucket exists."""
    return client.bucket_exists(bucket_name)


def create_bucket(client, bucket_name):
    """Create a bucket if it does not exist."""
    if not bucket_exists(client, bucket_name):
        client.make_bucket(bucket_name)
        print(f"[LOG] Bucket '{bucket_name}' created.")
    else:
        print(f"[LOG] Bucket '{bucket_name}' already exists.")


def create_initial_model_bucket():
    """ Create the initial model bucket if it does not exist. """
    create_bucket(get_db_connection(), INITIAL_MODEL_BUCKET)

    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "unknown")
    print(f"[LOG] Bucket '{INITIAL_MODEL_BUCKET}' created successfully in [{environment}] at URL: [{minio_endpoint}]")


def create_offline_updated_models_bucket():
    """ Create the offline updated models bucket if it does not exist. """
    create_bucket(get_db_connection(), OFFLINE_UPDATED_MODELS_BUCKET)

    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "unknown")
    print(f"[LOG] Bucket '{OFFLINE_UPDATED_MODELS_BUCKET}' created successfully in [{environment}] at URL: [{minio_endpoint}]")


def create_online_updated_user_models_bucket():
    """ Create the online updated user models bucket if it does not exist. """
    create_bucket(get_db_connection(), ONLINE_UPDATED_USER_MODELS_BUCKET)

    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "unknown")
    print(f"[LOG] Bucket '{ONLINE_UPDATED_USER_MODELS_BUCKET}' created successfully in [{environment}] at URL: [{minio_endpoint}]")


def drop_buckets():
    """Delete all buckets and their contents."""
    client = get_db_connection()
    for bucket in [INITIAL_MODEL_BUCKET, OFFLINE_UPDATED_MODELS_BUCKET, ONLINE_UPDATED_USER_MODELS_BUCKET]:
        if bucket_exists(client, bucket):
            objects = client.list_objects(bucket, recursive=True)
            for obj in objects:
                client.remove_object(bucket, obj.object_name)
            client.remove_bucket(bucket)
            print(f"[LOG] Bucket '{bucket}' and its contents deleted.")
        else:
            print(f"[LOG] Bucket '{bucket}' does not exist.")


def reset_buckets():
    """Reset all buckets by deleting their contents without deleting the buckets themselves."""
    client = get_db_connection()
    for bucket in [INITIAL_MODEL_BUCKET, OFFLINE_UPDATED_MODELS_BUCKET, ONLINE_UPDATED_USER_MODELS_BUCKET]:
        if bucket_exists(client, bucket):
            objects = client.list_objects(bucket, recursive=True)
            for obj in objects:
                client.remove_object(bucket, obj.object_name)
            print(f"[LOG] Bucket '{bucket}' has been emptied.")
        else:
            print(f"[LOG] Bucket '{bucket}' does not exist.")


def upload_initial_model(model_filepath):
    """
    Upload an initial pre-trained model to MinIO.

    This function overwrites the existing initial model if it already exists.
    
    Args:
        model_filepath (str): Local filepath of the model to be uploaded.
    """
    client = get_db_connection()
    object_name = "init_model.pth"
    client.fput_object(INITIAL_MODEL_BUCKET, object_name, model_filepath)
    print(f"[LOG] Initial pre-trained model '{object_name}' uploaded.")


def upload_offline_updated_model(model_filepath, date, drop_old=False):
    """
    Upload an offline updated model to MinIO with date-based versioning.

    This function overwrites the existing offline model for the provided date if it already exists.
    Moreover, it can remove all models uploaded before the given date if requested.
    
    Args:
        model_filepath (str): Local filepath of the model to be uploaded.
        date (str): Date identifier for versioning the model.
        drop_old (bool): If True, removes all models uploaded before the given date.
    """
    client = get_db_connection()
    object_name = f"{date}_offline_model.pth"
    client.fput_object(OFFLINE_UPDATED_MODELS_BUCKET, object_name, model_filepath)
    print(f"[LOG] Offline updated model '{object_name}' uploaded.")

    # Drop old models if requested
    if drop_old:
        objects = list(client.list_objects(OFFLINE_UPDATED_MODELS_BUCKET, recursive=True))
        for obj in objects:
            obj_date = obj.object_name.split("_")[0]  # Extract date from filename
            if obj_date < date:
                client.remove_object(OFFLINE_UPDATED_MODELS_BUCKET, obj.object_name)
                print(f"[LOG] Old model '{obj.object_name}' removed.")


def upload_online_user_model(model_filepath, user_id):
    """
    Upload an online user-specific model to MinIO.

    This function overwrites the existing online model for the provided user if it already exists.
    
    Args:
        model_filepath (str): Local filepath of the model to be uploaded.
        user_id (str): The ID of the user to whom the model belongs.
    """
    client = get_db_connection()
    object_name = f"{user_id}_online_model.pth"
    client.fput_object(ONLINE_UPDATED_USER_MODELS_BUCKET, object_name, model_filepath)
    print(f"[LOG] Online user model for user '{user_id}' uploaded.")


def download_initial_model(download_path, custom_name: str=None):
    """Download the latest initial model.
    
    Args:
        download_path (str): Local path where the model will be saved.
        custom_name (str, optional): Custom name for the downloaded file. Defaults to the original 
            object name.
    """
    client = get_db_connection()
    object_name = "init_model.pth"
    download_filename = (custom_name + ".pth") if custom_name else object_name
    download_filepath = os.path.join(download_path, download_filename)
    
    client.fget_object(INITIAL_MODEL_BUCKET, object_name, download_filepath)
    print(f"[LOG] Initial model '{object_name}' downloaded as '{download_filename}'.")


def download_offline_updated_model(download_path, date, custom_name: str=None):
    """
    Download an offline updated model from MinIO by date.
    
    Args:
        download_filepath (str): Local path where the model will be saved.
        date (str): Date identifier to locate the specific model version to retrieve from MinIO.
        custom_name (str, optional): Custom name for the downloaded file. Defaults to the original 
            object name.
    """
    client = get_db_connection()
    object_name = f"{date}_offline_model.pth"
    download_filename = (custom_name + ".pth") if custom_name else object_name
    download_filepath = os.path.join(download_path, download_filename)
    
    client.fget_object(OFFLINE_UPDATED_MODELS_BUCKET, object_name, download_filepath)
    print(f"[LOG] Offline updated model '{object_name}' downloaded as '{download_filename}'.")


def download_last_offline_updated_model(download_path, custom_name: str=None):
    """
    Download the most recently uploaded offline updated model.
    
    Args:
        download_path (str): Local directory where the model will be saved.
        custom_name (str, optional): Custom name for the downloaded file. Defaults to the original 
            object name.
    """
    client = get_db_connection()

    objects = list(client.list_objects(OFFLINE_UPDATED_MODELS_BUCKET, recursive=True))
    if not objects:
        print("[LOG] No offline updated models found.")
        return
    
    latest_object = max(objects, key=lambda obj: obj.last_modified)
    download_filename = (custom_name + ".pth") if custom_name else latest_object.object_name
    download_filepath = os.path.join(download_path, download_filename)

    client.fget_object(OFFLINE_UPDATED_MODELS_BUCKET, latest_object.object_name, download_filepath)
    print(f"[LOG] Latest offline updated model '{latest_object.object_name}' downloaded as '{download_filename}'.")


def download_online_user_model(download_path, user_id, custom_name: str=None):
    """
    Download an online user-specific model from MinIO.
    
    Args:
        download_filepath (str): Local path where the model will be saved.
        user_id (str): The ID of the user whose model should be retrieved from MinIO.
        custom_name (str, optional): Custom name for the downloaded file. Defaults to the original 
            object name.
    """
    client = get_db_connection()
    object_name = f"{user_id}_online_model.pth"
    download_filename = (custom_name + ".pth") if custom_name else object_name
    download_filepath = os.path.join(download_path, download_filename)
    
    client.fget_object(ONLINE_UPDATED_USER_MODELS_BUCKET, object_name, download_filepath)
    print(f"[LOG] Online user model for user '{user_id}' downloaded as '{download_filename}'.")


def list_objects_in_bucket(bucket_name):
    """
    List all objects stored in a specific MinIO bucket.
    
    Args:
        bucket_name (str): Name of the bucket to list objects from.
    """
    client = get_db_connection()
    if bucket_exists(client, bucket_name):
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            print(obj.object_name)
    else:
        print(f"[LOG] Bucket '{bucket_name}' does not exist.")
