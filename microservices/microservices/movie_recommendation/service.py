import numpy as np
import os
import sys
import time
import tempfile
from typing import Optional, List
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pathlib import Path
from pydantic import BaseModel


def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

# Add the project directories to the system path
if is_docker():
    print("[INFO] Running inside a Docker container")

    # Set the project root
    project_root = "/app"
    sys.path.append(os.path.join(project_root, 'movie_recommendation_system'))
else:
    print("[INFO] Running locally")

    # Dynamically find the project root
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(os.path.join(project_root, 'movie_recommendation_system', 'src'))

# Add the project directories to the system path
sys.path.append(os.path.join(project_root, 'db_handlers'))

# Load environment variables from .env file
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


## Define data and trained models directories
DATA_PATH = os.path.join(project_root, 'data')
# Main directories
DATA_DIR = os.path.join(DATA_PATH, "movielens")
PROCESSED_DATA_DIR = os.path.join(DATA_PATH, "movielens_processed")
# Filepaths
TDH_FILEPATH = os.path.join(PROCESSED_DATA_DIR, "tdh_instance.pkl")


from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.models.gnn_retrain_strategies import GNNRetrainModelHandler
from movie_recommender.recommenders.popularity_ranking import PopularityRanking
from movie_recommender.recommenders.collaborative_filtering import CollaborativeFiltering
from movie_recommender.recommenders.hybrid_filtering import HybridFiltering
from db_handlers.user_postgres_sql_db_handler import (
    get_user_model_id,
)
from db_handlers.movie_cast_and_crew_postgres_sql_db_handler import (
    get_all_movies,
    get_all_cast_and_crew,
    get_joined_movies_and_cast_and_crew,
)
from db_handlers.trained_models_minio_storage_db_handler import (
    download_last_offline_updated_model, 
    upload_online_user_model, 
    download_online_user_model,
)
from db_handlers.user_movie_hbase_table_db_handler import (
    get_all_user_predictions,
)


################################################################
#   Run Instruction (from CMD): 'run service.py'               #
#   Server avaliable at web page: http://localhost:8002/docs   #
################################################################


class FilteringParams(BaseModel):
    """Base model for filtering parameters."""
    genres: Optional[List[str]] = None
    year: Optional[int] = None
    actors: Optional[List[str]] = None 
    director: Optional[str] = None
    # ... add other filtering parameters as needed


# Create a FastAPI app
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    """Redirects the root URL to the '/docs' URL."""
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


def _filter_movies_df(movies_df, filtering_params: FilteringParams):
    """
    Filters movies_df based on filtering_params.
    """
    filtered_df = movies_df.copy()
    if filtering_params.genres:
        filtered_df = filtered_df[filtered_df['genres'].apply(
            lambda g: all(genre in g for genre in filtering_params.genres))]
    if filtering_params.year:
        filtered_df = filtered_df[filtered_df['year'].astype(str) == str(filtering_params.year)]
    if filtering_params.actors:
        for actor in filtering_params.actors:
            filtered_df = filtered_df[filtered_df['cast_list'].apply(
            lambda a: any(x.get('name') == actor for x in a))]
    if filtering_params.director:
        filtered_df = filtered_df[filtered_df['crew_list'].apply(
            lambda d: all(x.get('job') != 'Director' or x.get('name').strip().lower() == filtering_params.director.strip().lower() for x in d))]
    return filtered_df 


# Endpoint for collaborative_filtering.py
@app.get("/users/{user_id}/recommendations/popularity_ranking")
async def get_recommendation_popularity_ranking(
    genres: Optional[List[str]] = Query(default=None),
    year: Optional[int] = None,
    actors: Optional[List[str]] = Query(default=None),
    director: Optional[str] = None,
    top_n: int = 10
):
    filtering_params = FilteringParams(
        genres=genres,
        year=year,
        actors=actors,
        director=director
    )

    if filtering_params.actors is None and filtering_params.director is None:
        # Retrieve movies from the database
        movies_df = get_all_movies()
    else:
        # Retrieve movies and cast/crew from the database
        movies_df = get_joined_movies_and_cast_and_crew()

    # Filter movies based on the filtering parameters
    filtered_movies_df = _filter_movies_df(movies_df, filtering_params)

    # Initialize a tabular data handler instance containing the filtered movies
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)
    tdh.update_datasets(movies_df=filtered_movies_df)

    # Recommend movies based on popularity ranking
    top_movies_for_genre = PopularityRanking.top_movies_IMDB_wr_formula(
        tabular_dataset_handler=tdh,
        top_n=top_n
    )

    return top_movies_for_genre.to_dict(orient='records')


# Endpoint for collaborative_filtering.py
@app.get("/users/{user_id}/recommendations/collaborative_filtering")
async def get_recommendation_collaborative_filtering(
    user_id: str,
    genres: Optional[List[str]] = Query(default=None),
    year: Optional[int] = None,
    actors: Optional[List[str]] = Query(default=None),
    director: Optional[str] = None,
    top_n: int = 10
):
    filtering_params = FilteringParams(
        genres=genres,
        year=year,
        actors=actors,
        director=director
    )

    if filtering_params.actors is None and filtering_params.director is None:
        # Retrieve movies from the database
        movies_df = get_all_movies()
    else:
        # Retrieve movies and cast/crew from the database
        movies_df = get_joined_movies_and_cast_and_crew()

    # Convert user UUID (str) to model ID (int)
    model_id = get_user_model_id(user_id=user_id)

    # Retrieve user's precomputed movie ratings from database
    user_movie_predictions_df = get_all_user_predictions(model_id)

    # Filter movies based on the filtering parameters
    filtered_movies_df = _filter_movies_df(movies_df, filtering_params)

    # Join the user movie predictions with the movies DataFrame
    filtered_movies_df = filtered_movies_df.rename(columns={'id': 'movieId'})
    recommended_movies_df = user_movie_predictions_df.merge(filtered_movies_df, on='movieId')

    # Order the recommended movies by rating
    recommended_movies_df = recommended_movies_df.sort_values(by='rating', ascending=False)

    # Select the top N recommended movies
    recommended_movies_df = recommended_movies_df.head(top_n)

    return recommended_movies_df.to_dict(orient='records')


# Endpoint for hybrid_filtering.py
@app.get("/users/{user_id}/recommendations/hybrid_filtering")
async def get_recommendation_hybrid_filtering(
    user_id: str,
    movie_title: str,
    top_n: int=10
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download the user's GNN model from the database in the temporary directory
        temp_model_name = "temp_model_name"
        download_online_user_model(download_path=tmp_dir, user_id=user_id, custom_name=temp_model_name)

        # Load the user's GNN model from the temporary directory
        model_filepath = os.path.join(tmp_dir, temp_model_name + ".pth")
        GraphSAGE_model = GNNRetrainModelHandler.load_pretrained_model(
            pretrained_model_filepath=model_filepath
        )

    # Initialize the hybrid filtering recommender
    hybrid_GraphSAGE_recommender = HybridFiltering(
        collaborative_filtering=CollaborativeFiltering(model_handler=GraphSAGE_model)
    )

    # Convert user UUID (str) to model ID (int)
    model_id = get_user_model_id(user_id=user_id)

    # Suggest similar movies based on the user's GNN model and the provided movie title
    similar_movies_df = hybrid_GraphSAGE_recommender.suggest_similar_movies(
        user_id=model_id,
        movie_title=movie_title,
        top_n=top_n
    )

    # Substitute ±∞ values with NaN, then replace NaN values with None for JSON serialization
    similar_movies_df = similar_movies_df.replace([float('inf'), float('-inf')], np.nan)
    similar_movies_df = similar_movies_df.astype(object).where(similar_movies_df.notnull(), None)

    return similar_movies_df.to_dict(orient='records')


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server:
    #   - host="0.0.0.0" → Makes the server accessible on the local network
    #   - port=8002 → The API will be available at http://localhost:8002
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MOVIE_RECOMMENDATION_MICROSERVICE_PORT")))
