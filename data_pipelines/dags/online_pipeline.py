from airflow import DAG
from airflow.decorators import task
from airflow.operators.dummy import DummyOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta

import os
import pandas as pd

from shared import *


from db_handlers.user_postgres_sql_db_handler import (
    get_user_model_id,
)
from db_handlers.user_mongodb_nosql_db_handler import (
    get_user_movie_ratings,
)
from db_handlers.trained_models_minio_storage_db_handler import (
    download_last_offline_updated_model, 
    upload_online_user_model, 
    download_online_user_model,
)
from db_handlers.user_movie_hbase_table_db_handler import (
    store_user_movie_predictions,
)


@task()
def load_last_user_ratings(user_id: str, num_new_ratings: int, **kwargs):
    """Load the last user-movie ratings for a specific user from the database."""
    print(f"[LOG] Loading the last {num_new_ratings} user-movie ratings for user {user_id}")

    # Retrieve last user's ratings using the imported DB handler
    user_ratings = get_user_movie_ratings(user_id=user_id, last_k=num_new_ratings)

    # Store the last user's ratings in a DataFrame
    new_user_ratings_df = pd.DataFrame(
        data=user_ratings,
        columns=["userId", "movieId", "rating", "timestamp"]
    )

    # Convert user UUID (str) to model ID (int)
    new_user_ratings_df["userId"] = new_user_ratings_df["userId"].apply(get_user_model_id)

    # Store last user ratings locally for the next task
    new_user_ratings_df_filepath = os.path.join(TEMP_ONLINE_DIR, f"user_{user_id}_ratings.csv")
    new_user_ratings_df.to_csv(new_user_ratings_df_filepath, index=False)

    print(f"[LOG] Saved ratings for user {user_id} to {new_user_ratings_df_filepath}")

    return new_user_ratings_df_filepath

@task()
def load_user_online_gnn_model(user_id: str, **kwargs):
    """Load user's existing GNN model from storage."""
    print(f"[LOG] Loading user {user_id} online GNN model")

    # Define model name and path using shared variables
    old_user_model_name = ONLINE_OLD_USER_MODEL_NAME + user_id
    old_user_online_model_filepath = os.path.join(TEMP_ONLINE_DIR, old_user_model_name + ".pth")

    try:
        # Attempt to download the user-specific online model
        print(f"[LOG] Attempting to download model '{old_user_model_name}' for user {user_id} to {TEMP_ONLINE_DIR}")
        download_online_user_model(download_path=TEMP_ONLINE_DIR, user_id=user_id, custom_name=old_user_model_name)
        print(f"[LOG] Successfully downloaded existing model for user {user_id}")
    except Exception as e:
        # If the model is not found, download the last offline updated model instead
        print(f"[LOG] User {user_id} has no online GNN model assigned yet. Downloading the last offline updated model.")
        download_last_offline_updated_model(download_path=TEMP_ONLINE_DIR, custom_name=old_user_model_name)
        print(f"[LOG] Successfully downloaded last offline updated model for user {user_id}")

    return old_user_online_model_filepath


# TODO: aggiungi un blocco per fare il check per verificare di non essere nel caso in cui tutte le recensioni
# sono già state usate per l'addestramento del modello, in tal caso bisogna saltare direttamente a nodo "end". 


#@task(pool="gpu_pool") # Limit concurrency due to GPU resources
@task
def online_train_gnn_model(user_id: str, new_user_ratings_df_filepath: str, old_user_online_model_filepath: str, **kwargs):
    """
    Update the GNN model using new user ratings via an external script.
    """
    print(f"[LOG] Online training GNN model for user {user_id}, using ratings file: {new_user_ratings_df_filepath} and base model file: {old_user_online_model_filepath}")

    # Define the path for the newly trained model
    new_user_model_name = ONLINE_NEW_USER_MODEL_NAME + user_id
    new_user_online_model_filepath = os.path.join(TEMP_ONLINE_DIR, new_user_model_name + ".pth")

    # Construct script arguments
    script_args = [
        "--user_id", str(user_id),
        "--new_user_ratings_df_filepath", new_user_ratings_df_filepath,
        "--old_user_online_model_filepath", old_user_online_model_filepath,
        "--new_user_online_model_name", new_user_model_name,
    ]
    script_path = os.path.join(SCRIPT_PATH, "online", "train_online_user_gnn_script.py")

    # Run the training script
    print(f"[LOG] Running script: {script_path} with args: {script_args}")
    run_cuda_script(script_path, args=script_args)
    print(f"[LOG] Finished training for user {user_id}. New model should be at: {new_user_online_model_filepath}")

    # Return the path to the newly trained model
    return new_user_online_model_filepath

@task()
def deploy_model(user_id: str, new_user_online_model_filepath: str, **kwargs):
    """Deploy the updated user-specific GNN model to storage."""
    print(f"[LOG] Deploying the updated GNN model for user {user_id} from {new_user_online_model_filepath}")

    # Upload the online GNN model
    upload_online_user_model(model_filepath=new_user_online_model_filepath, user_id=user_id)
    print(f"[LOG] Successfully deployed model for user {user_id}")

@task()
def pre_compute_user_movie_ratings(user_id: str, new_user_online_model_filepath: str, **kwargs):
    """Pre-compute user-movie ratings using the newly trained model via an external script."""
    print(f"[LOG] Pre-computing user-movie ratings for user {user_id} using model {new_user_online_model_filepath}")

    # Define paths for the output file
    precomputed_ratings_filename = f"precomputed_ratings_user_{user_id}.csv"
    precomputed_ratings_filepath = os.path.join(TEMP_ONLINE_DIR, precomputed_ratings_filename)

    # Convert user UUID (str) to model ID (int)
    model_id = get_user_model_id(user_id=user_id)

    # Construct script arguments
    script_args = [
        "--user_id", str(user_id),
        "--model_id", str(model_id),
        "--new_user_online_model_filepath", new_user_online_model_filepath,
        "--precomputed_ratings_filepath", precomputed_ratings_filepath,
    ]
    script_path = os.path.join(SCRIPT_PATH, "online", "pre_compute_user_movie_ratings_script.py")

    # Run the prediction script
    print(f"[LOG] Running script: {script_path} with args: {script_args}")
    run_cuda_script(script_path, args=script_args)
    print(f"[LOG] Finished pre-computing ratings for user {user_id}. Output should be at: {precomputed_ratings_filepath}")
    
    return precomputed_ratings_filepath

@task()
def deploy_pre_computed_user_movie_ratings(user_id: str, precomputed_ratings_filepath: str, **kwargs):
    """Deploy the pre-computed user-movie ratings to the database (e.g., HBase)."""
    print(f"[LOG] Deploying pre-computed ratings for user {user_id} from {precomputed_ratings_filepath}")

    # Load the pre-computed user-movie ratings
    precomputed_ratings_df = pd.read_csv(precomputed_ratings_filepath)

    # Store the pre-computed ratings
    print(f"[LOG] Storing {len(precomputed_ratings_df)} predictions for user {user_id}.")
    store_user_movie_predictions(predictions_df=precomputed_ratings_df)
    print(f"[LOG] Successfully deployed pre-computed ratings for user {user_id}")

@task()
def cleanup_temp_files(
    new_user_ratings_df_filepath: str,
    old_user_online_model_filepath: str,
    new_user_online_model_filepath: str,
    precomputed_ratings_filepath: str,
    **kwargs
):
    """Clean up temporary files created during the pipeline."""
    print(f"[LOG] Cleaning up temporary files.")

    # Clean up temporary files
    if os.path.exists(new_user_ratings_df_filepath):
        os.remove(new_user_ratings_df_filepath)
        print(f"[LOG] Removed temporary ratings file: {new_user_ratings_df_filepath}")

    if os.path.exists(old_user_online_model_filepath):
        os.remove(old_user_online_model_filepath)
        print(f"[LOG] Removed temporary old model file: {old_user_online_model_filepath}")

    if os.path.exists(new_user_online_model_filepath):
        os.remove(new_user_online_model_filepath)
        print(f"[LOG] Removed temporary model file: {new_user_online_model_filepath}")
    
    if os.path.exists(precomputed_ratings_filepath):
        os.remove(precomputed_ratings_filepath)
        print(f"[LOG] Removed temporary pre-computed ratings file: {precomputed_ratings_filepath}")


# --- Define the DAG ---
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),  # Start from now
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'depends_on_past': False,
}

with DAG(
    dag_id='online_pipeline',
    default_args=default_args,
    description='Parallel pipeline for GNN online training per user (using .expand())',
    schedule_interval=None,
    catchup=False,
    max_active_tasks=4,
    #tags=['gnn', 'recommendation', 'online-training', 'production', 'simplified'],
) as dag:

    start = DummyOperator(task_id="start")

    @task(task_id="get_user_info")
    def get_user_info(**context):
        """Extracts validated user information list from the DAG run configuration."""
        # Return list of dicts: [{'user_id': '1', 'num_new_ratings': 5}, ...]
        dag_run = context.get("dag_run")
        user_info = []
        if dag_run:
            conf = dag_run.conf
            raw_user_info = conf.get("users_to_retrain", [])
            print(f"[LOG] Received raw user info: {raw_user_info}")
            for info in raw_user_info:
                if isinstance(info, dict) and "user_id" in info and "num_new_ratings" in info:
                    user_info.append(info)
                else:
                    print(f"[WARN] Invalid user info format skipped: {info}")
        if not user_info:
            print("[WARN] No valid user info provided in DAG run config 'users_to_retrain'.")
        return user_info

    user_info_list = get_user_info()

    # --- Extract individual argument lists needed for .expand() ---
    @task
    def extract_user_ids(user_info: list):
        return [str(user.get("user_id")) for user in user_info if user.get("user_id")]

    @task
    def extract_num_ratings(user_info: list):
        return [int(user.get("num_new_ratings")) for user in user_info if user.get("num_new_ratings") is not None]

    user_ids = extract_user_ids(user_info_list)
    num_ratings = extract_num_ratings(user_info_list)

    # --- Define Workflow using .expand() ---

    # 1. Load Ratings (expands over user_ids and num_ratings)
    loaded_ratings_filepaths = load_last_user_ratings.expand(
        user_id=user_ids, num_new_ratings=num_ratings
    )

    # 2. Load Model (expands over user_ids)
    loaded_model_filepaths = load_user_online_gnn_model.expand(user_id=user_ids)

    # 3. Train Model (expands over user_ids, loaded_ratings_filepaths, loaded_model_filepaths)
    trained_model_filepaths = online_train_gnn_model.expand(
        user_id=user_ids,
        new_user_ratings_df_filepath=loaded_ratings_filepaths,
        old_user_online_model_filepath=loaded_model_filepaths
    )

    # Branch 1: Deploy Model
    # 4. Deploy Model (expands over user_ids, trained_model_filepaths)
    deployed_model_task = deploy_model.expand(
        user_id=user_ids,
        new_user_online_model_filepath=trained_model_filepaths
    )

    # Branch 2: Pre-compute and Deploy Ratings
    # 5. Pre-compute Ratings (expands over user_ids, trained_model_filepaths)
    precomputed_ratings_filepaths = pre_compute_user_movie_ratings.expand(
        user_id=user_ids,
        new_user_online_model_filepath=trained_model_filepaths
    )

    # 6. Deploy Pre-computed Ratings (expands over user_ids, precomputed_ratings_filepaths)
    deployed_ratings_task = deploy_pre_computed_user_movie_ratings.expand(
        user_id=user_ids,
        precomputed_ratings_filepath=precomputed_ratings_filepaths
    )

    # 7. Cleanup Temporary Files (expands over trained_model_filepaths and precomputed_ratings_filepaths)
    cleanup_temp_files_task = cleanup_temp_files.expand(
        new_user_ratings_df_filepath=loaded_ratings_filepaths,
        old_user_online_model_filepath=loaded_model_filepaths,
        new_user_online_model_filepath=trained_model_filepaths,
        precomputed_ratings_filepath=precomputed_ratings_filepaths
    )

    # --- End Task ---
    end = DummyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # --- Define DAG Task Dependencies ---
    # (Dependencies remain the same as in the previous version)
    start >> user_info_list
    user_info_list >> user_ids
    user_info_list >> num_ratings
    [user_ids, num_ratings] >> loaded_ratings_filepaths
    user_ids >> loaded_model_filepaths
    [loaded_ratings_filepaths, loaded_model_filepaths] >> trained_model_filepaths
    trained_model_filepaths >> deployed_model_task # Branch 1
    trained_model_filepaths >> precomputed_ratings_filepaths >> deployed_ratings_task # Branch 2
    [deployed_model_task, deployed_ratings_task] >> cleanup_temp_files_task >>end