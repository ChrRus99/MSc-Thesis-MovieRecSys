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
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.models.gnn_model import GCNEncoder, GraphSAGEEncoder, GATEncoder
from movie_recommender.models.gnn_train_eval_pred import GNNModelHandler


# Define the main function to launch this script as subprocess in the DAG pipeline
def main():
    # Load the graph dataset from the previous task
    gdh = HeterogeneousGraphDatasetHandler.load_class_instance(filepath=GDH_FILEPATH)

    # Initialize an autoregressive GNN model with GraphSAGE encoder
    GraphSAGE_model = GNNModelHandler(
        graph_dataset_handler=gdh,
        gnn_encoder=GraphSAGEEncoder(hidden_channels=64, out_channels=64),
    )

    # Train the GNN model and store the trained model for the next task
    GraphSAGE_model.train(
        num_epochs=500, 
        model_name=INIT_MODEL_NAME, 
        trained_model_path=TEMP_INIT_DIR,
        store_tensorboard_training_plot=False,
    )

if __name__ == "__main__":
    main()