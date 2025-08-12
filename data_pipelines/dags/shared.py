import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Helper functions
def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()

def run_cuda_script(script_path: str, args: Optional[List[str]] = None):
    """ 
    Runs a Python script based on CUDA as a subprocess.
    
    Args:
        script_path (str): The path to the Python script to run.
        args (Optional[List[str]]): A list of arguments to pass to the script. Defaults to None.
    """
    script_name = os.path.basename(script_path)
    print(f"[INFO] Running CUDA script [{script_name}] as a subprocess")

    # Pass the script path and arguments to the subprocess
    command = [sys.executable, script_path]
    if args:
        command.extend(args)

    # Run the script as a subprocess
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Script executed successfully as subprocess: {script_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to execute script as subprocess: {script_path}")
        print(e.stderr)
    except FileNotFoundError:
        print(f"[ERROR] Script not found: {script_path}")


# Add the project directories to the system path
if is_docker():
    print("[INFO] Running inside a Docker container")
    ROOT_PATH = '/app'
    SCRIPT_PATH = '/opt/airflow/scripts'
    sys.path.append(os.path.join(ROOT_PATH, 'movie_recommendation_system'))
else:
    print("[INFO] Running locally")
    ROOT_PATH = 'D:\\Internship\\recsys'
    SCRIPT_PATH = 'D:\\Internship\\recsys\\data_pipelines\\scripts'
    sys.path.append(os.path.join(ROOT_PATH,'movie_recommendation_system', 'src'))

sys.path.append(os.path.join(ROOT_PATH, 'db_handlers'))
sys.path.append(os.path.join(ROOT_PATH, 'data'))


## Define data and trained models directories
DATA_PATH = os.path.join(ROOT_PATH, 'data')
# Main directories
DATA_DIR = os.path.join(DATA_PATH, "movielens")
PROCESSED_DATA_DIR = os.path.join(DATA_PATH, "movielens_processed")
# Temporary directory
TEMP_DIR = os.path.join(DATA_PATH, "temp")
TEMP_INIT_DIR = os.path.join(TEMP_DIR, "init")
TEMP_OFFLINE_DIR = os.path.join(TEMP_DIR, "offline")
TEMP_ONLINE_DIR = os.path.join(TEMP_DIR, "online")

# Create directories if they do not exist
for directory in [TEMP_DIR, TEMP_INIT_DIR, TEMP_OFFLINE_DIR, TEMP_ONLINE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Filepaths
# Init pipeline filepaths
TDH_FILEPATH = os.path.join(PROCESSED_DATA_DIR, "tdh_instance.pkl")
GDH_FILEPATH = os.path.join(PROCESSED_DATA_DIR, "gdh_instance.pkl")

INIT_MODEL_NAME = "init_GNN_model"
INIT_MODEL_FILEPATH = os.path.join(TEMP_INIT_DIR, INIT_MODEL_NAME + ".pth")

# Offline pipeline filepaths
TEMP_OFFLINE_LAST_USERS_RATINGS_FILEPATH = os.path.join(TEMP_OFFLINE_DIR, "last_users_ratings.csv")

OFFLINE_OLD_MODEL_NAME = "old_GNN_model"
TEMP_OFFLINE_OLD_MODEL_FILEPATH = os.path.join(TEMP_OFFLINE_DIR, OFFLINE_OLD_MODEL_NAME + ".pth")
OFFLINE_NEW_MODEL_NAME = "offline_updated_GNN_model"
TEMP_OFFLINE_NEW_MODEL_FILEPATH = os.path.join(TEMP_OFFLINE_DIR, OFFLINE_NEW_MODEL_NAME + ".pth")

# Online pipeline filepaths
#TEMP_ONLINE_LAST_USER_RATINGS_FILEPATH = TEMP_ONLINE_DIR # + user_id + ".csv"

ONLINE_OLD_USER_MODEL_NAME = "old_GNN_model_user_" # + user_id
#TEMP_ONLINE_OLD_USER_MODEL_FILEPATH = os.path.join(TEMP_ONLINE_DIR, ONLINE_OLD_USER_MODEL_NAME + ".pth")
ONLINE_NEW_USER_MODEL_NAME = "online_updated_GNN_user_" # + user_id
#TEMP_ONLINE_NEW_USER_MODEL_FILEPATH = os.path.join(TEMP_ONLINE_DIR, ONLINE_NEW_USER_MODEL_NAME + ".pth")