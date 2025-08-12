import difflib
import os
import pandas as pd
import sys
import time
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pathlib import Path
from pydantic import BaseModel, Field


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


# Import the necessary modules
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from db_handlers.user_postgres_sql_db_handler import (
    store_new_user,
    is_user_registered,
    get_user_info,
    get_user_ids,
)
from db_handlers.user_mongodb_nosql_db_handler import (
    store_user_movie_rating, 
    store_user_preference,
    get_user_movie_ratings, 
    get_user_preferences,
)


################################################################
#   Run Instruction (from CMD): 'run service.py'               #
#   Server avaliable at web page: http://localhost:8003/docs   #
################################################################

# Load the 'movies_df' DataFrame once at startup (for efficiency)
if TabularDatasetHandler:
    try:
        tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)
        movies_df = tdh.get_movies_df_deepcopy()
        print(f"Loaded movies_df with {len(movies_df)} entries.")

        # Ensure the 'title_norm' column exists (as done in service.py)
        if 'title_norm' not in movies_df.columns:
             movies_df['title_norm'] = movies_df['title'].str.lower().str.strip()
    except FileNotFoundError:
        print(f"[ERROR] TDH file not found at {TDH_FILEPATH}")
        movies_df = None
    except Exception as e:
        print(f"[ERROR] Failed to load TDH or movies_df: {e}")
        movies_df = None
else:
    movies_df = None

# Helper function to get movie_id from movie_title (character-based matching)
def _get_movie_id_from_title(movie_title: str) -> str:
    """
    Finds the movie_id for a given movie_title using exact and character-based matching.
    Uses the globally loaded 'movies_df'. Returns movie_id as str.
    """
    if movies_df is None:
        raise RuntimeError("movies_df is not loaded.")

    # Normalize the movie title for comparison
    search_title_norm = movie_title.lower().strip()

    # Try exact match on normalized title
    exact_matches = movies_df[movies_df['title_norm'] == search_title_norm]
    if len(exact_matches) == 1:
        matched_row = exact_matches.iloc[0]
        return str(matched_row['id'])
    elif len(exact_matches) > 1:
        # Multiple exact matches found, ambiguous
        matched_row = exact_matches.iloc[0]
        return str(matched_row['id'])

    # No exact match found, use character-based matching (difflib)
    titles_norm_list = movies_df['title_norm'].tolist()
    close_matches = difflib.get_close_matches(search_title_norm, titles_norm_list, n=1, cutoff=0.8)
    if close_matches:
        matched_norm_title = close_matches[0]
        matched_movies = movies_df[movies_df['title_norm'] == matched_norm_title]
        if len(matched_movies) == 1:
            matched_row = matched_movies.iloc[0]
            return str(matched_row['id'])
        elif len(matched_movies) > 1:
            matched_row = matched_movies.iloc[0]
            return str(matched_row['id'])
    
    # If no match is found, raise ValueError
    raise ValueError(f"No match found for movie title '{movie_title}'")


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    first_name: str
    last_name: str
    email: str

class UserMovieRating(BaseModel):
    user_id: str
    movie_title: str
    rating: float

class UserPreference(BaseModel):
    user_id: str
    preference: str


# Create a FastAPI app
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    """Redirects the root URL to the '/docs' URL."""
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoints for user_postgres_sql_db_handler.py
@app.post("/users")
async def create_user(user: User):
    """Creates a new user and returns the user ID."""
    #user_id = str(uuid.uuid4())
    timestamp = int(time.time())
    store_new_user(user.user_id, user.first_name, user.last_name, user.email, timestamp)
    return {"user_id": user.user_id}

@app.get("/users/{user_id}/exists")
async def user_exists(user_id: str):
    """Checks if a user exists for the given user ID."""
    exists = is_user_registered(user_id)
    return {"exists": exists}
    
@app.get("/users/{user_id}")
async def read_user(user_id: str):
    """Returns the user information for the given user ID."""
    user_info = get_user_info(user_id)
    if user_info is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_info

@app.get("/users")
async def list_users():
    """Returns the list of user IDs."""
    return {"user_ids": get_user_ids()}


# Endpoints for user_mongodb_nosql_db_handler.py
@app.post("/users/{user_id}/ratings")
async def add_movie_rating(rating: UserMovieRating):
    timestamp = int(time.time())

    # Resolve movie_id from movie_title
    movie_id = _get_movie_id_from_title(rating.movie_title)

    store_user_movie_rating(rating.user_id, movie_id, rating.rating, timestamp)
    return {"message": "Movie rating stored successfully", "movie_id": movie_id}

@app.get("/users/{user_id}/ratings")
async def get_movie_ratings(user_id: str, after_timestamp: int=-1):
    ratings = get_user_movie_ratings(user_id, after_timestamp)
    return {"ratings": ratings}

@app.post("/users/{user_id}/preferences")
async def add_preference(preference: UserPreference):
    timestamp = int(time.time())
    store_user_preference(preference.user_id, preference.preference, timestamp)
    return {"message": "User preference stored successfully"}

@app.get("/users/{user_id}/preferences",)
async def get_preferences(user_id: str):
    preferences = get_user_preferences(user_id)
    return {"preferences": preferences}


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server:
    #   - host="0.0.0.0" → Makes the server accessible on the local network
    #   - port=8003 → The API will be available at http://localhost:8003
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("USER_MICROSERVICE_PORT")))