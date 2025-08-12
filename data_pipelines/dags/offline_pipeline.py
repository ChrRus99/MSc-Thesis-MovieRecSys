from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.decorators import task
from airflow.operators.python import PythonOperator, BranchPythonOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook
# from airflow.providers.mongo.hooks.mongo import MongoHook
# from airflow.providers.neo4j.hooks.neo4j import Neo4jHook
from datetime import datetime, timedelta

import pandas as pd

# Import data_pipelines shared variables and functions
from shared import *

# Import the necessary modules
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.models.gnn_model import GCNEncoder, GraphSAGEEncoder, GATEncoder
from movie_recommender.models.gnn_train_eval_pred import GNNModelHandler

from db_handlers.movie_cast_and_crew_postgres_sql_db_handler import (
    get_movie_metadata,
    get_cast_and_crew_metadata,
    get_movie_ids,
)
from db_handlers.user_postgres_sql_db_handler import (
    get_user_model_id,
)
from db_handlers.user_mongodb_nosql_db_handler import (
    get_user_movie_ratings,
    get_user_preferences,
    get_all_last_user_movie_ratings,
)
from db_handlers.utils import (
    group_ratings_by_user,
)
from db_handlers.trained_models_minio_storage_db_handler import (
    upload_offline_updated_model,
    download_last_offline_updated_model,
)

## Define the functions for the DAG tasks
# Extract
def load_last_ratings(**kwargs):
    """Load the last user-movie ratings from the database."""
    print("[LOG] Loading the last user-movie ratings")

    # Retrieve all user ratings which are not older than 1 day ago
    current_time = datetime.now().timestamp()
    users_ratings = get_all_last_user_movie_ratings(after_timestamp=current_time - 24 * 60 * 60)

    # Conditional branching
    if len(users_ratings) > 0:
        # Convert user ratings to a DataFrame
        user_ratings_df = pd.DataFrame(
            data=users_ratings,
            columns=["userId", "movieId", "rating", "timestamp"]
        )

        # Convert user UUID (str) to model ID (int)
        user_ratings_df["userId"] = user_ratings_df["userId"].apply(get_user_model_id)

        # Store user preferences locally for the next task
        user_ratings_df.to_csv(TEMP_OFFLINE_LAST_USERS_RATINGS_FILEPATH, index=False)

        return "load_last_offline_gnn_model"
    else:
        return "end"

# Extract
def load_last_offline_gnn_model(**kwargs):
    """Load the last offline GNN model from the database."""
    print("[LOG] Loading the last offline GNN model")

    # Download the last offline updated model and store it locally for the next task
    download_last_offline_updated_model(download_path=TEMP_OFFLINE_DIR, custom_name=OFFLINE_OLD_MODEL_NAME)

# Operation
def offline_train_gnn_model(**kwargs):
    """
    Update the GNN model by performing an offline incremental learning using the new user-movie 
    ratings collected from all users.
    """
    print("[LOG] Offline training GNN model with new user-movie ratings")

    # Run a subprocess to train the GNN model with the new user-movie ratings
    script_path = os.path.join(SCRIPT_PATH, "offline", "train_offline_gnn_script.py")
    run_cuda_script(script_path)

# Load
def deploy_model(**kwargs):
    """Deploy the updated offline GNN model"""
    print("[LOG] Deploying the updated offline GNN model")

    # Upload the updated offline GNN model to the MinIO database
    current_date = datetime.now().strftime("%Y-%m-%d")
    upload_offline_updated_model(TEMP_OFFLINE_NEW_MODEL_FILEPATH, date=current_date, drop_old=False)


## Define the DAG
# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(), # datetime.now() + timedelta(days=1),  # Start tomorrow
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'offline_pipeline',
    default_args=default_args,
    description='Pipeline for GNN offline training',
    schedule_interval='@daily',  # Runs once a day (starting from tomorrow)
    catchup=False
)


## Define tasks
# Dummy start task
start = DummyOperator(task_id="start")

load_last_ratings_task = BranchPythonOperator(
    task_id='load_last_ratings',
    python_callable=load_last_ratings,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

load_last_offline_gnn_model_task = PythonOperator(
    task_id='load_last_offline_gnn_model',
    python_callable=load_last_offline_gnn_model,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

offline_train_gnn_model_task = PythonOperator(
    task_id='offline_train_gnn_model',
    python_callable=offline_train_gnn_model,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

# Dummy end task
end = DummyOperator(task_id="end", dag=dag, trigger_rule="none_failed")


## Define DAG tasks dependencies
start >> load_last_ratings_task >> [load_last_offline_gnn_model_task, end]
load_last_offline_gnn_model_task >> offline_train_gnn_model_task >> deploy_model_task >> end