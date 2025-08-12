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

# Import the necessary modules
from movie_recommender.data.tabular_dataset_handler import TabularDatasetHandler
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler


# Define the main function to launch this script as subprocess in the DAG pipeline
def main():
    # Load the processed data from the previous task
    tdh = TabularDatasetHandler.load_class_instance(filepath=TDH_FILEPATH)

    # Initialize the heterogeneous graph dataset handler
    gdh = HeterogeneousGraphDatasetHandler(preprocessed_tdh=tdh)
    
    # Build the heterogeneous graph dataset
    gdh.build_graph_dataset()

    # Store the current 'HeterogeneousGraphDatasetHandler' instance locally for the next task
    gdh.store_class_instance(filepath=GDH_FILEPATH)

if __name__ == "__main__":
    main()