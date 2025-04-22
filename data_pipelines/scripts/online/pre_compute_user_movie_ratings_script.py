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

from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.models.gnn_retrain_strategies import GNNRetrainModelHandler
from movie_recommender.recommenders.collaborative_filtering import CollaborativeFiltering


# Define the main function to launch this script as subprocess in the DAG pipeline
def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description="Script for online training of user GNN model.")

    # Parse the command-line arguments
    parser.add_argument("--user_id", type=str, help="User ID for whom to pre-compute ratings.")
    parser.add_argument("--model_id", type=str, help="Model ID of the user GNN model.")
    parser.add_argument("--new_user_online_model_filepath", type=str, help="Path to the old user online GNN model.")
    parser.add_argument("--precomputed_ratings_filepath", type=str, help="Filepath to store the pre-computed user-movie ratings.")
    args = parser.parse_args()

    # Extract the parameters from the command line arguments
    user_id = args.user_id
    model_id = int(args.model_id)
    new_user_online_model_filepath = args.new_user_online_model_filepath
    precomputed_ratings_filepath = args.precomputed_ratings_filepath

    # Load the pre-processed movielens datasets
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)

    # Load movies_df from the pre-processed movielens datasets
    movies_df = tdh.get_movies_df_deepcopy()

    # Load the online trained user GNN model from the previous task
    GraphSAGE_model = GNNRetrainModelHandler.load_pretrained_model(
        pretrained_model_filepath=new_user_online_model_filepath
    )

    # Pre-compute user-movie ratings for the user using a collaborative filtering model
    GraphSAGE_recommender = CollaborativeFiltering(model_handler=GraphSAGE_model)
    pred_ratings_train_df = GraphSAGE_recommender.predict_ratings(
        user_id=model_id,
        subset_movies_df=movies_df,
        use_batch=True,
    )

    # Create a new column 'ratings' merging 'ground_truth_rating' in 'predicted_rating' if not NaN 
    # namely, we use the ground truth rating if available, otherwise we use the predicted rating
    pred_ratings_train_df['rating'] = pred_ratings_train_df.apply(
        lambda row: row['ground_truth_rating'] if not pd.isna(row['ground_truth_rating']) else row['predicted_rating'],
        axis=1
    )

    # Keep only the 'movieId' and 'ratings' columns
    pred_ratings_train_df = pred_ratings_train_df[['movieId', 'rating']]

    # Add the userId column to the DataFrame
    pred_ratings_train_df['userId'] = model_id

    # Store the pre-computed user-movie ratings locally for the next task
    pred_ratings_train_df = pred_ratings_train_df[['userId', 'movieId', 'rating']]
    pred_ratings_train_df.to_csv(precomputed_ratings_filepath, index=False)

if __name__ == "__main__":
    main()
