import argparse
import pandas as pd
import sys
from pathlib import Path

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

root_path = '/opt/airflow' if is_docker() else 'D:\\Internship\\recsys\\data_pipelines'
sys.path.append(root_path)

# Import data_pipelines shared variables and functions
from dags.shared import *

from movie_recommender.models.gnn_retrain_strategies import GNNRetrainModelHandler


# Define the main function to launch this script as subprocess in the DAG pipeline
def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description="Script for online training of user i GNN model.")

    # Parse the command-line arguments
    parser.add_argument("--user_id", type=str, help="User ID for whom to pre-compute ratings.")
    parser.add_argument("--new_user_ratings_df_filepath", type=str, help="Filepath to the new user-movie ratings DataFrame.")
    parser.add_argument("--old_user_online_model_filepath", type=str, help="Filepath to the old user online GNN model.")
    parser.add_argument("--new_user_online_model_name", type=str, help="Name of the new user online GNN model.")
    args = parser.parse_args()

    # Extract the parameters from the command line arguments
    user_id = args.user_id
    new_user_ratings_df_filepath = args.new_user_ratings_df_filepath
    old_user_online_model_filepath = args.old_user_online_model_filepath
    new_user_online_model_name = args.new_user_online_model_name

    # Load the latest offline GNN model from the previous task
    GraphSAGE_model = GNNRetrainModelHandler.load_pretrained_model(
        pretrained_model_filepath=old_user_online_model_filepath
    )

    # Load new user-movie ratings from the previous task
    new_users_ratings_df = pd.read_csv(new_user_ratings_df_filepath)

    # TODO: Aggiungi gestione nuovi film


    # Set the new train set for the re-training of the model
    GraphSAGE_model.add_new_train_data(new_ratings_df=new_users_ratings_df)

    # Incremental train the GNN model and store the updated model locally for the next task
    GraphSAGE_model.fine_tune(
        num_epochs=15,
        lr=0.001,
        model_name=new_user_online_model_name,
        trained_model_path=TEMP_ONLINE_DIR,
        store_tensorboard_training_plot=False,
    )

if __name__ == "__main__":
    main()
