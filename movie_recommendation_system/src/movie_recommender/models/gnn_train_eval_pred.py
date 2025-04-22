# Dataset
#import matplotlib
#matplotlib.use('TkAgg')
import pandas as pd
import numpy as np

# Util
import os
import math
#import pickle
import shutil
from overrides import overrides
from typing import Tuple

# Pytorch geometric
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

# Machine learning

# My scripts
from movie_recommender.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from movie_recommender.models import ModelHandlerInterface
from movie_recommender.models.gnn_model import GNNEncoderInterface, GNNModel

"""
References:
    - Link Regression on MovieLens:
        https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2?usp=sharing
    - Link Prediction on MovieLens:
        https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing
    - Link Prediction on Heterogeneous Graphs with PyG:
        https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
    - Heterogeneous Graph Learning (Pytorch-geometric):
        https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
    - GRAPH Link Prediction w/ DGL on Pytorch and PyG Code Example | GraphML | GNN:
        https://www.youtube.com/watch?v=wxJ84sMJfUA&ab_channel=code_your_own_AI
    - PyTorch Geometric tutorial: Graph Autoencoders & Variational Graph Autoencoders:
        https://www.youtube.com/watch?v=qA6U4nIK62E&ab_channel=AntonioLonga
"""


class GNNModelHandler(ModelHandlerInterface):
    """
    This class handles training, evaluation, and prediction for a Graph Neural Network (GNN) model
    in a movie recommendation system.

    Attributes:
        _gdh (HeterogeneousGraphDatasetHandler): An instance of HeterogeneousGraphDatasetHandler managing the dataset.
        _graph_dataset (HeteroData): The heterogeneous graph dataset.
        _gnn (GNNEncoderInterface): The GNN encoder used in the edge regression model.
        _model (GNNModel): The edge regression model.
        _train_data, _val_data, _test_data (HeteroData): Datasets for training, validation, and testing.
        _device (torch.device): The device used for computations (GPU if available, else CPU).
    """

    def __init__(
        self,
        graph_dataset_handler: HeterogeneousGraphDatasetHandler,
        gnn_encoder: GNNEncoderInterface,
    ):
        """
        Initializes the GNN model handler.

        Parameters:
            graph_dataset_handler (HeterogeneousGraphDatasetHandler): An instance of 
                HeterogeneousGraphDatasetHandler managing the dataset.
            gnn_encoder (GNNEncoderInterface): The encoder used in the GNN model.
        """
        super().__init__()

        # Heterogeneous graph dataset
        self._gdh = graph_dataset_handler
        self._graph_dataset = self._gdh.get_graph_dataset()

        # GNN encoder
        self._gnn_encoder = gnn_encoder

        # Initialize the GNN model with the specified encoder
        self._model = GNNModel(dataset=self._graph_dataset, gnn_encoder=gnn_encoder)

        # Split the dataset in training, validation and test sets
        self._train_data, self._val_data, self._test_data = self._edge_level_dataset_split(self._graph_dataset)

        # Enable GPU computation (if available)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @overrides
    def train(
        self,
        num_epochs: int=300,
        lr: float=0.01,
        model_name: str=None,
        trained_model_path: str=None,
        store_tensorboard_training_plot: bool=True,
        plot_train_loop: bool=False,
        enable_early_stopping: bool=True
    ):
        """
        Trains the GNN model on the training set obtained by partitioning the heterogeneous 
        graph dataset.

        This function supports (if enabled):
            - Model storage: The trained model is stored in the specified path. 
            - TensorBoard logging: To store the model training plot.
            - Early stopping: To stop the training when the validation loss does not improve, to 
                avoid overfitting.

        Parameters:
            num_epochs (int, optional): Number of training epochs. Default is 300.
            lr (float, optional): Learning rate for training. Default is 0.01.
            model_name (str, optional): Custom name for saving the trained model.
            trained_model_path (str): Path to store the trained model. If None, the model will not
                be stored. Default is None.
            store_tensorboard_training_plot (bool, optional): Whether to store the TensorBoard plot
                of the model training. Default is True.
            plot_train_loop (bool, optional): Whether to plot the training loop in real-time. 
                Default is False.
            enable_early_stopping (bool, optional): Whether to enable early stopping mechanism.
                Default is True.
        
        Notes on adaptive early stopping:
            - Scales automatically with num_epochs.
            - Prevents overfitting by adjusting patience dynamically.
            - Keeps it bounded to avoid too early or too late stopping
        """
        print(f"Device: '{self._device}'")      

        # Move the model on device
        self._model = self._model.to(self._device)

        # Move data on device
        self._train_data = self._train_data.to(self._device)
        self._val_data = self._val_data.to(self._device)

        # Create a mini-batch loader that generates sub-graphs that can be used as input to the GNN
        # train_loader = self.__training_minibatch_loader()
        # validation_loader = self.__validation_minibatch_loader()

        # Define the optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Define a single training step
        def train_step(train_data):
            # Set the model in training mode
            self._model.train()
            optimizer.zero_grad()

            # Forward pass: estimate the predicted rating
            pred = self._model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data['user', 'movie'].edge_label_index
            )

            # Extract the ground truth rating
            target = train_data['user', 'movie'].edge_label

            # Compute the MSE loss function
            loss = F.mse_loss(pred, target)

            # Backward pass: compute the gradients of the model and update model weights
            loss.backward()
            optimizer.step()

            return float(loss)

        # Define a single evaluation step
        @torch.no_grad()
        def evaluation_step(data):
            # Set the model in evaluation mode
            self._model.eval()

            # Forward pass: estimate the predicted rating
            pred = self._model(
                data.x_dict,
                data.edge_index_dict,
                data['user', 'movie'].edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating
            target = data['user', 'movie'].edge_label.float()

            # Compute the performance measures: RMSE and MAE
            rmse = F.mse_loss(pred, target).sqrt()
            mae = F.l1_loss(pred, target)

            return float(rmse), float(mae)

        # Define the name with which to save the trained model
        if model_name is None:
            model_name = self._gnn_encoder.model_name + "_based_model"  # Default model name

        # Initialize TensorBoard writer
        if store_tensorboard_training_plot:
            # Define the path where to store the tensorboard information
            tensorboard_path = os.path.join(trained_model_path, "runs")

            # Store tensorboard information to get the plot at the end of the training
            tensorboard_foldername = os.path.join(
                tensorboard_path,
                "training_plot_" + model_name + "_" + str(num_epochs) + "_epochs_" + str(lr) + "_lr"
            )

            # If the folder already exists delete it
            if os.path.exists(tensorboard_foldername):  
                shutil.rmtree(tensorboard_foldername)

            # Define the SummaryWriter for tensorboard
            writer = SummaryWriter(tensorboard_foldername)      

        # Define early stopping parameters
        if enable_early_stopping:
            # Early stopping parameters
            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            best_model_epoch = num_epochs

            # Compute adaptive patience based on num_epochs
            patience = min(num_epochs, self._compute_adaptive_patience(num_epochs))
            print(f"Adaptive patience set to {patience} epochs based on num_epochs={num_epochs}.")

        if plot_train_loop:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output

            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(10, 6))
            line_loss, = ax.plot([], [], label='Training Loss')
            line_train_rmse, = ax.plot([], [], label='Training RMSE')
            line_val_rmse, = ax.plot([], [], label='Validation RMSE')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Real-time Training Metrics')
            ax.legend()
            ax.grid(True)
            ax.set_xlim(0, num_epochs)
            ax.set_ylim(0, 4)

            plt.show(block=False) # Ensure the plot is shown but doesn't block execution
            #display(fig) # Force display of the figure widget

            # Initialize lists for metrics
            training_losses = []
            training_rmses = []
            validation_rmses = []
            epochs_list = []

        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Training step
            loss = train_step(self._train_data)

            # Evaluation step
            train_rmse, train_mae = evaluation_step(self._train_data)
            val_rmse, val_mae = evaluation_step(self._val_data)

            # Print training information
            if not plot_train_loop:
                print(f"Epoch: {epoch:03d}, Train loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")

            # TensorBoard logging
            if store_tensorboard_training_plot:
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('RMSE/train', train_rmse, epoch)
                writer.add_scalar('RMSE/val', val_rmse, epoch)
                writer.add_scalar('MAE/train', train_mae, epoch)
                writer.add_scalar('MAE/val', val_mae, epoch)
        
            # If plot real-time training loop is enabled
            if plot_train_loop and epoch % 10 == 0:  # Update every 10 epochs
                # Append new data
                training_losses.append(loss)
                training_rmses.append(train_rmse)
                validation_rmses.append(val_rmse)
                epochs_list.append(epoch)

                # Update the plot lines
                line_loss.set_data(epochs_list, training_losses)
                line_train_rmse.set_data(epochs_list, training_rmses)
                line_val_rmse.set_data(epochs_list, validation_rmses)

                # # Ensure axes are dynamically adjusting
                # ax.set_xlim(0, max(epochs_list) + 1)  # Extend x-axis dynamically
                # ax.set_ylim(0, max(max(training_losses, default=0), max(training_rmses, default=0), max(validation_rmses, default=0)) * 1.1)  # Extend y-axis

                # Update axes and redraw
                ax.relim()
                ax.autoscale_view()
                #plt.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Clear previous output and re-display the figure widget
                clear_output(wait=True)
                display(fig)

                #plt.pause(0.001) # Small pause to allow the plot to update

            # Early stopping
            if enable_early_stopping and epoch > num_epochs/2:  # Prevent early oscillations
                # If validation RMSE does not improve for 'patience' epochs, stop training
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    best_model_state = self._model.state_dict()  # Save best model
                    best_model_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}. Best validation RMSE: {best_val_rmse:.4f} at epoch {best_model_epoch}.")
                        break  # Stop training

        # Restore the best model
        if enable_early_stopping and best_model_state is not None:
            self._model.load_state_dict(best_model_state)
            print(f"Best model with lowest validation RMSE restored from epoch {best_model_epoch}.")

        # Close TensorBoard writer
        if store_tensorboard_training_plot:
            writer.close()
            print(f'\nTensorBoard training information of model {model_name} saved at path: {tensorboard_foldername}')

        # Keep displaying the final plot
        if plot_train_loop:
            plt.ioff() # Turn off interactive mode
            plt.show() # Keep the final plot displayed after training

        # Store the trained model checkpoint in .pth format
        if trained_model_path:
            model_filename = os.path.join(trained_model_path, model_name + ".pth")
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'gnn_encoder': self._gnn_encoder,
                'gnn_encoder_state_dict': self._gnn_encoder.state_dict(),
                'gdh': self._gdh,
                'train_data': self._train_data,
                'val_data': self._val_data,
                'test_data': self._test_data,
            }, model_filename)
            print(f'\nTrained model {model_name} saved at path: {trained_model_path}')

    @overrides
    def evaluate_performance(self):
        """
        Evaluates the model on unseen data from the test set obtained by partioning the 
        heterogeneous graph dataset. The evaluation is based on RMSE and MAE metrics.
        """
        print(f"Device: '{self._device}\n'")

        # Move the model on device
        self._model = self._model.to(self._device)

        with torch.no_grad():
            # Move data on device
            self._test_data = self._test_data.to(self._device)

            # Estimate the predicted rating
            pred = self._model(
                self._test_data.x_dict,
                self._test_data.edge_index_dict,
                self._test_data['user', 'movie'].edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating
            target = self._test_data['user', 'movie'].edge_label.float()

            # Compute the performance measures: RMSE and MAE
            rmse = F.mse_loss(pred, target).sqrt()
            mae = F.l1_loss(pred, target)
            print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}\n')

        # Prints a comparison between ground truth ratings and predicted rating for all data in the test set
        user_id = self._test_data['user', 'movie'].edge_label_index[0].cpu().numpy()
        movie_id = self._test_data['user', 'movie'].edge_label_index[1].cpu().numpy()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        print(pd.DataFrame({'userId': user_id, 'movieId': movie_id, 'pred_rating': pred, 'gt_rating': target}))

    @overrides
    def predict(self, user_id, movie_id, gt_rating=None):
        """
        Predicts the rating for a given user and movie.

        Parameters:
            user_id (int): The ID of the user.
            movie_id (int): The ID of the movie.
            gt_rating (float, optional): Ground truth rating (if available).

        Returns:
            Tuple containing the predicted rating and the ground truth rating.
        """
        # Map user and movie IDs (of the tabular dataset) to their corresponding indices in the graph dataset
        mapped_user_id = self._gdh.mapped_users_ids_df[self._gdh.mapped_users_ids_df['userId'] == user_id]['mappedUserId'].item()
        mapped_movie_id = self._gdh.mapped_movies_ids_df[self._gdh.mapped_movies_ids_df['movieId'] == movie_id]['mappedMovieId'].item()

        # Create edge_label_index for the given user and movie
        edge_label_index = (
            torch.tensor([mapped_user_id], dtype=torch.long),
            torch.tensor([mapped_movie_id], dtype=torch.long)
        )

        # Move the model on device
        self._model = self._model.to(self._device)

        # Move the input data to the device
        data = self._graph_dataset.to(self._device)

        # Use the trained model for prediction
        with torch.no_grad():
            pred = self._model(
                data.x_dict,
                data.edge_index_dict,
                edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating from the edge
            #target = data['user', 'rating', 'movie'].edge_label[edge_label_index[0][0]].item()

            # Extract the ground truth rating directly from the tabular dataset
            target_df = self._gdh.users_ratings_df[
                (self._gdh.users_ratings_df['userId'] == user_id) &
                (self._gdh.users_ratings_df['movieId'] == movie_id)
            ]['rating']

            # Check if there are any matching items
            if not target_df.empty:
                target = target_df.iloc[0]
            else:
                target = np.nan

        return pred.item(), target

    def predict_batch(self, user_id, movie_ids, include_gt_ratings: bool = True):
        """
        Predicts ratings for a given user over a batch of movies.

        Parameters:
            user_id (int): The ID of the user.
            movie_ids (List[int]): A list of movie IDs for which to predict ratings.
            include_gt_ratings (bool, optional): Whether to include ground truth ratings. Defaults 
                to True.

        Returns:
            Tuple: Two lists, one with predicted ratings and one with ground truth ratings (if
                include_gt_ratings is True; otherwise, only predicted ratings).
        """
        # Map the single user ID to its graph index
        mapped_user_id = self._gdh.mapped_users_ids_df[
            self._gdh.mapped_users_ids_df['userId'] == user_id
        ]['mappedUserId'].item()

        # Map each movie ID to its graph index
        mapped_movie_ids = [
            self._gdh.mapped_movies_ids_df[
                self._gdh.mapped_movies_ids_df['movieId'] == movie_id
            ]['mappedMovieId'].item() for movie_id in movie_ids
        ]

        # Create batched edge_label_index: one tensor for user IDs repeated, and one tensor for movie IDs
        edge_label_index = (
            torch.tensor([mapped_user_id] * len(movie_ids), dtype=torch.long),
            torch.tensor(mapped_movie_ids, dtype=torch.long)
        )

        # Move the model and data to the target device (GPU if available)
        self._model = self._model.to(self._device)
        data = self._graph_dataset.to(self._device)

        # Predict ratings in a batched manner
        with torch.no_grad():
            predictions = self._model(
                data.x_dict,
                data.edge_index_dict,
                edge_label_index
            )
            predictions = predictions.clamp(min=0, max=5)

        predicted_ratings = predictions.tolist()

        # Optionally, fetch ground truth ratings for each movie from the tabular dataset
        ground_truth_ratings = []
        if include_gt_ratings:
            for movie_id in movie_ids:
                target_df = self._gdh.users_ratings_df[
                    (self._gdh.users_ratings_df['userId'] == user_id) &
                    (self._gdh.users_ratings_df['movieId'] == movie_id)
                ]['rating']
                if not target_df.empty:
                    ground_truth_ratings.append(target_df.iloc[0])
                else:
                    ground_truth_ratings.append(np.nan)
            return predicted_ratings, ground_truth_ratings

        return predicted_ratings

    @staticmethod
    def load_pretrained_model(
        pretrained_model_filepath: str,
    ) -> "GNNModelHandler":
        """
        Loads a pretrained GNN model handler from the given file path.

        Parameters:
            pretrained_model_filepath (str): Filepath to the saved pretrained model handler 
                checkpoint (.pth format).
        
        Returns:
            GNNModelHandler: The handler instance of the loaded model.
        """
        # Register safe globals to avoid unpickling errors in PyTorch 2.6+
        torch.serialization.add_safe_globals([
            HeterogeneousGraphDatasetHandler, 
            GNNEncoderInterface
        ])

        # Load the pre-trained model handler checkpoint
        #checkpoint = torch.load(pretrained_model_filepath)
        checkpoint = torch.load(
            pretrained_model_filepath,
            map_location=torch.device('cpu'),
            weights_only=False
        )

        # Restore the graph dataset handler and the GNN encoder
        gdh = checkpoint['gdh']
        gnn_encoder = checkpoint['gnn_encoder']

        # Initialize a GNNModelHandler instance to restore the loaded model checkpoint
        instance = GNNModelHandler(
            graph_dataset_handler=gdh, 
            gnn_encoder=gnn_encoder
        )

        # Restore the pre-trained model state
        instance._model.load_state_dict(checkpoint['model_state_dict'])
        instance._gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state_dict'])

        # Restore other objects
        instance._train_data = checkpoint['train_data']
        instance._val_data = checkpoint['val_data']
        instance._test_data = checkpoint['test_data']

        # Enable GPU computation (if available)
        instance._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return instance

    def _edge_level_dataset_split(
        self,
        graph_dataset: HeteroData,
        val_ratio=0.1,
        test_ratio=0.1,
        neg_sampling_ratio=0.0
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """
        Splits the edges of the graph dataset into training, validation, and test sets.

        Parameters:
            graph_dataset (HeteroData): The heterogeneous graph dataset to split.
            val_ratio (float): The portion of the dataset to use for validation.
            test_ratio (float): The portion of the dataset to use for test.
            neg_sampling_ratio (float): The ratio of negative edges generated for the evaluation.

        Returns:
            Tuple of training set, validation set, and test set.
        """
        # Split the set of edges into training (80%), validation (10%), and testing edges (10%).
        # Across the training edges, use 70% of edges for message passing, and 30% of edges for supervision.
        # Generate fixed negative edges for evaluation with a ratio of 2:1.
        # Negative edges will be generated on-the-fly during training.
        transform = T.RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            #disjoint_train_ratio=disjoint_train_ratio,
            neg_sampling_ratio=neg_sampling_ratio,
            #add_negative_train_samples=False,
            edge_types=('user', 'rating', 'movie'),
            rev_edge_types=('movie', 'rev_rating', 'user'),
        )
        train_data, val_data, test_data = transform(graph_dataset)

        return train_data, val_data, test_data

    # Deprecated function: does not work on GPU
    def __training_minibatch_loader(
        self,
        batch_size=128,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0
    ) -> LinkNeighborLoader:
        """
        Creates a mini-batch loader that generates sub-graphs for training.

        Note: while this step is not strictly necessary for small-scale graphs, it is absolutely
        necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise. 
        To do this it samples multiple hops from both ends of an edge and creates a subgraph from it
        (using the loader.LinkNeighborLoader).

        Parameters:
            batch_size (int): The batch size.
            num_neighbors (list of int): The maximum number of neighbors to sample at each hop.
            neg_sampling_ratio (float): The ratio of negative edges to sample during training.

        Returns:
            LinkNeighborLoader: The train loader.
        """
        # Define seed edges
        edge_label_index = self._train_data['user', 'rating', 'movie'].edge_label_index
        edge_label = self._train_data['user', 'rating', 'movie'].edge_label

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        # First hop: sample at most 20 neighbors.
        # Second hop: sample at most 10 neighbors.
        # Negative edges will be sampled on-the-fly during training, with a ratio of 2:1.
        train_loader = LinkNeighborLoader(
            data=self._train_data,
            num_neighbors=num_neighbors,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=(('user', 'rating', 'movie'), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader

    # Deprecated function: does not work on GPU
    def __validation_minibatch_loader(
        self,
        batch_size=3 * 128,
        num_neighbors=[20, 10],
    ) -> LinkNeighborLoader:
        """
        Creates a mini-batch loader that generates sub-graphs for validation.

        Note: while this step is not strictly necessary for small-scale graphs, it is absolutely
        necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise. 
        To do this it samples multiple hops from both ends of an edge and creates a subgraph from it
        (using the loader.LinkNeighborLoader).

        Parameters:
            batch_size (int): The batch size.
            num_neighbors (list of int): The maximum number of neighbors to sample at each hop.

        Returns:
            LinkNeighborLoader: The validation loader.
        """
        # Define seed edges
        edge_label_index = self._val_data["user", "rating", "movie"].edge_label_index
        edge_label = self._val_data["user", "rating", "movie"].edge_label

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        # First hop: sample at most 20 neighbors.
        # Second hop: sample at most 10 neighbors.
        val_loader = LinkNeighborLoader(
            data=self._val_data,
            num_neighbors=num_neighbors,
            edge_label_index=(("user", "rating", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False,
        )

        return val_loader
    
    def _compute_adaptive_patience(
        self,
        num_epochs: int,
        min_patience: int=10,
        max_patience: int=50,
        alpha: float=0.6
    ) -> int:
        """
        Computes an adaptive patience value based on num_epochs using a logistic scaling function.

        Parameters:
            num_epochs (int): Total number of training epochs.
            min_patience (int): Minimum patience value.
            max_patience (int): Maximum patience value.
            alpha (float): Fraction of num_epochs to set as the midpoint M (default 0.5).

        Returns:
            int: Computed patience value.
        """
        k = 0.01                # Steepness of the logistic function
        M = alpha * num_epochs  # Midpoint is proportional to num_epochs

        # Compute patience using logistic function
        patience = (max_patience - min_patience) / (1 + math.exp(-k * (num_epochs - M)))

        # Ensure patience is within [min_patience, max_patience]
        patience = int(min(max(patience, min_patience), max_patience))

        return patience