from airflow import DAG
from airflow.operators.dummy import DummyOperator 
from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook
# from airflow.providers.mongo.hooks.mongo import MongoHook
# from airflow.providers.neo4j.hooks.neo4j import Neo4jHook
from datetime import datetime, timedelta

import numpy as np

# Import data_pipelines shared variables and functions
from shared import *

# Import the necessary modules
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.models.gnn_model import GCNEncoder, GraphSAGEEncoder, GATEncoder
from movie_recommender.models.gnn_train_eval_pred import GNNModelHandler

from db_handlers.movie_cast_and_crew_postgres_sql_db_handler import (
    create_movie_metadata_table,
    create_cast_and_crew_table,
    drop_movie_cast_and_crew_tables,
    store_movielens_movies,
    store_movielens_cast_and_crew,
)
from db_handlers.user_postgres_sql_db_handler import (
    create_user_register_table,
    create_user_model_lookup_table,
    drop_user_tables,
)
from db_handlers.user_mongodb_nosql_db_handler import (
    create_user_movie_ratings_collection,
    create_user_preferences_collection,
    drop_user_collections,
)
from db_handlers.trained_models_minio_storage_db_handler import (
    create_initial_model_bucket,
    create_offline_updated_models_bucket,
    create_online_updated_user_models_bucket,
    drop_buckets,
    upload_initial_model,
    upload_offline_updated_model,
)
from db_handlers.user_movie_hbase_table_db_handler import (
    create_user_movie_ratings_table,
    drop_user_movie_ratings_table,
)
from db_handlers.kg_rag_neo4j_db_handler import (
    create_knowledge_graph,
    reset_knowledge_graph,
)


## Define the functions for the DAG tasks
# Extract + Transform
def load_and_preprocess_datasets(**kwargs):
    """Load and preprocess data from the MovieLens datasets using the 'TabularDatasetHandler'."""
    print("[LOG] Loading and preprocessing MovieLens datasets")

    # Initialize the tabular dataset handler to load and process the movielens datasets
    tdh = TabularDatasetHandler(data_path=DATA_DIR)

    # Load the movielens datasets: CSV files to Pandas DataFrames
    tdh.load_datasets()

    # Perform preprocessing on the datasets
    tdh.preprocess_datasets()

    # Store the current 'TabularDatasetHandler' instance locally for the next tasks
    tdh.store_class_instance(filepath=TDH_FILEPATH)

# Branch 1:
# Reset
def reset_existing_databases(**kwargs):
    """Reset SQL and NoSQL databases, if exist, by dropping tables and collections."""
    print("[LOG] Resetting SQL and NoSQL databases")

    # Drop SQL database tables
    drop_movie_cast_and_crew_tables()
    drop_user_tables()

    # Drop NoSQL database collections
    drop_user_collections()

    # Drop MinIO buckets for storing trained models
    drop_buckets()

    # Drop HBase table user-movie ratings
    drop_user_movie_ratings_table()

    # Reset Neo4j knowledge graph
    reset_knowledge_graph()

# Init
def init_databases(**kwargs):
    """Initialize SQL and NoSQL databases by creating tables and collections."""
    print("[LOG] Initializing SQL and NoSQL databases")

    # Create and initialize SQL database tables
    create_movie_metadata_table()
    create_cast_and_crew_table()
    create_user_register_table()
    create_user_model_lookup_table()

    # Create and initialize NoSQL database collections
    create_user_movie_ratings_collection()
    create_user_preferences_collection()

    # Create MinIO buckets for storing trained models
    create_initial_model_bucket()
    create_offline_updated_models_bucket()
    create_online_updated_user_models_bucket()

    # Create HBase table for storing user-movie ratings
    create_user_movie_ratings_table()

# Load
def store_data_in_databases(**kwargs):
    """Push processed data into SQL and NoSQL databases."""
    print("[LOG] Storing processed data in SQL and NoSQL databases")
    
    # Load the processed data from the previous task
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)

    # Store the processed data into SQL databases
    store_movielens_movies(tdh.get_movies_df_deepcopy())
    store_movielens_cast_and_crew(tdh.get_credits_df_deepcopy())

    # Store the processed data into NoSQL databases
    # We do not have users info at this point, so we do not store user_register, user_ratings, user_preferences

# Branch 2:
# Debug
def debug_cuda():
    """Debug CUDA environment variables and PyTorch CUDA availability."""
    print("[LOG] Debugging CUDA environment variables and PyTorch CUDA availability")

    # Run a subprocess to debug CUDA environment and PyTorch CUDA availability
    script_path = os.path.join(SCRIPT_PATH, "init", "debug_cuda_script.py")
    run_cuda_script(script_path)

# Transform
def build_graph_dataset(**kwargs):
    """Convert the tabular dataset into a graph dataset using the 'HeterogeneousGraphDatasetHandler'."""
    print("[LOG] Building graph dataset for GNN training")

    # Run a subprocess to convert the tabular dataset into a graph dataset
    script_path = os.path.join(SCRIPT_PATH, "init", "build_init_graph_dataset_script.py")
    run_cuda_script(script_path)

# Operation
def train_gnn_model(**kwargs):
    """Train a GNN model on the graph dataset."""
    print("[LOG] Pre-train the initial GNN model on the MovieLens graph dataset")
    
    # Run a subprocess to train the GNN-based collaborative filtering model
    script_path = os.path.join(SCRIPT_PATH, "init", "train_init_gnn_script.py")
    run_cuda_script(script_path)

# Load
def deploy_model(**kwargs):
    """Deploy the pre-trained GNN model on the MinIO database."""
    print("[LOG] Deploying the pre-trained GNN model on MinIO database")

    # Upload the initial GNN model to the MinIO database
    upload_initial_model(INIT_MODEL_FILEPATH)

    # Upload the offline GNN model to the MinIO database (init: same as initial model)
    current_date = datetime.now().strftime("%Y-%m-%d")
    upload_offline_updated_model(INIT_MODEL_FILEPATH, date=current_date, drop_old=False)

# Branch 3:
# Transform + Load
def build_neo4j_knowledge_graph(**kwargs):
    """Build Neo4j knowledge graph containing movies and cast and crew information."""
    print("[LOG] Building Neo4j knowledge graph")

    # Load the processed data from the previous task
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)
    movies_df = tdh.get_movies_df_deepcopy()
    credits_df = tdh.get_credits_df_deepcopy()

    # Merge movies and credits DataFrames
    merged_df = movies_df.merge(credits_df, on='id', how='left')
    merged_df = merged_df[[
        "id", "title", "budget", "genres", "overview", "production_companies", "production_countries",
        "release_date", "revenue", "runtime", "spoken_languages", "tagline", "cast", "crew"
    ]]

    # Preprocessing to handle NaN values
    merged_df['tagline'] = merged_df['tagline'].replace({np.nan: ""})
    merged_df['budget'] = merged_df['budget'].replace({np.nan: -1, 0: -1})
    merged_df['overview'] = merged_df['overview'].replace({np.nan: ""})
    merged_df['release_date'] = merged_df['release_date'].replace({np.nan: ""})
    merged_df['revenue'] = merged_df['revenue'].replace({np.nan: -1, 0: -1})
    merged_df['runtime'] = merged_df['runtime'].replace({np.nan: -1, 0: -1})
    merged_df['cast'] = merged_df['cast'].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )
    merged_df['crew'] = merged_df['crew'].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )

    # Build Neo4j knowledge graph
    create_knowledge_graph(movies_cast_and_crew_df=merged_df)


## Define the DAG
# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),  # Start immediately
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'init_pipeline',
    default_args=default_args,
    description='Pipeline for DBs initialization and GNN offline pre-training',
    schedule_interval=None, # '@once',  # Runs just once when the program starts
    catchup=False
)


## Define tasks
# Dummy start task
start = DummyOperator(task_id="start")

load_and_preprocess_datasets_task = PythonOperator(
    task_id='load_and_preprocess_datasets',
    python_callable=load_and_preprocess_datasets,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

# Branch 1: Create, Initialize and Store data in databases
reset_existing_databases_task = PythonOperator(
    task_id='reset_existing_databases',
    python_callable=reset_existing_databases,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

init_databases_task = PythonOperator(
    task_id='init_databases',
    python_callable=init_databases,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

store_data_in_databases_task = PythonOperator(
    task_id='store_data_in_databases',
    python_callable=store_data_in_databases,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

# Branch 2: Build graph, Train GNN, Deploy model
debug_cuda_task = PythonOperator(
    task_id='debug_cuda',
    python_callable=debug_cuda,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

build_graph_dataset_task = PythonOperator(
    task_id='build_graph_dataset',
    python_callable=build_graph_dataset,
    execution_timeout=timedelta(minutes=30),
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

train_gnn_model_task = PythonOperator(
    task_id='train_gnn_model',
    python_callable=train_gnn_model,
    execution_timeout=timedelta(minutes=30),
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

# Branch 3: Create, Initialize and Store data in Neo4j database
build_neo4j_knowledge_graph_task = PythonOperator(
    task_id='build_neo4j_knowledge_graph',
    python_callable=build_neo4j_knowledge_graph,
    provide_context=True,  # Allow tasks to access context
    dag=dag
)

# Dummy end task
end = DummyOperator(task_id="end")


## Define DAG tasks dependencies
start >> load_and_preprocess_datasets_task
load_and_preprocess_datasets_task >> [reset_existing_databases_task, debug_cuda_task]
reset_existing_databases_task >> init_databases_task >> store_data_in_databases_task
debug_cuda_task >> build_graph_dataset_task >> train_gnn_model_task
init_databases_task >> build_neo4j_knowledge_graph_task
[init_databases_task, train_gnn_model_task] >> deploy_model_task
[store_data_in_databases_task, deploy_model_task, build_neo4j_knowledge_graph_task] >> end