import pandas as pd
import sys
import uuid
from pathlib import Path

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

root_path = '/opt/airflow' if is_docker() else 'D:\\Internship\\recsys\\data_pipelines'
sys.path.append(root_path)

# Import data_pipelines shared variables and functions
from dags.shared import *

from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.models.gnn_retrain_strategies import GNNRetrainModelHandler


# https://chatgpt.com/share/67ddaab0-0ff0-8011-b213-ea555258090b
def _estimate_epochs(N_old: int, N_new: int, E_base: int = 20, alpha: float = 0.7) -> int:
    """
    Estimate the number of epochs for incremental GNN training.

    This uses a heuristic formula to estimate the number of epochs for incremental training based on
    the number of old and new ratings.
    
    Parameters:
    - N_old (int): Number of ratings used in the previous training.
    - N_new (int): Number of new ratings to be incorporated.
    - E_base (int): Base number of epochs for full training. Default is 20.
    - alpha (float): Decay factor controlling training emphasis on new data (0.5–1.0). Default is 0.7.
    
    Returns:
    - int: Estimated number of epochs for incremental training.
    """
    if N_new <= 0:
        return 0  # No new data, no need for training
    
    if N_old <= 0:
        return E_base  # First-time training, use full epochs
    
    E = E_base * (N_new / (N_old + N_new)) ** alpha
    return max(1, round(E))  # Ensure at least 1 epoch


# Define the main function to launch this script as subprocess in the DAG pipeline
def main():
    # TODO: aggiungi caricamento nuovi film se presenti

    # Load new user-movie ratings from the previous task
    new_users_ratings_df = pd.read_csv(TEMP_OFFLINE_LAST_USERS_RATINGS_FILEPATH)
    # Count the number of new user-movie ratings
    num_new_records = new_users_ratings_df.shape[0]

    # Load MovieLens user-movie ratings
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)
     # Count the number of user-movie ratings in the MovieLens dataset
    num_movielens_records = tdh.get_valid_ratings().shape[0]

    # Estimate the number of epochs for incremental training
    num_epochs = _estimate_epochs(
        N_old=num_movielens_records,
        N_new=num_new_records,
        E_base=30,
        alpha=0.2
    )

    # Load the latest offline GNN model from the previous task
    GraphSAGE_model = GNNRetrainModelHandler.load_pretrained_model(
        pretrained_model_filepath=TEMP_OFFLINE_OLD_MODEL_FILEPATH
    )

    # Set the new train set for the re-training of the model
    GraphSAGE_model.add_new_train_data(new_ratings_df=new_users_ratings_df)

    # Incremental train the GNN model and store the updated model locally for the next task
    GraphSAGE_model.incremental_train(
        num_epochs=num_epochs,
        lr=0.001,
        model_name=OFFLINE_NEW_MODEL_NAME,
        trained_model_path=TEMP_OFFLINE_DIR,
        store_tensorboard_training_plot=False,
    )

if __name__ == "__main__":
    main()