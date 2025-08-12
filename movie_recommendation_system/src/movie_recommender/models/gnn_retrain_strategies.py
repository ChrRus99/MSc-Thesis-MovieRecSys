# Dataset
import pandas as pd
import numpy as np

# Util
import copy
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
from movie_recommender.data.expandable_graph_dataset_handler import ExpandableHeterogeneousGraphDatasetHandler
from movie_recommender.models import GNNModelHandler
from movie_recommender.models.gnn_model import GNNEncoderInterface, GNNModel
from movie_recommender.data.utils import merge_hetero_data


class GNNRetrainModelHandler(GNNModelHandler):
    """
    This class extends the GNNModelHandler class to support the retraining of a GNN model on new
    data.

    It provides different strategies for retraining a GNN model on new data:

    - FULL RETRAINING: Retrain the model from scratch on the entire updated dataset.
        - Dataset splits:
            - Train set: All new data + old train data.
            - Validation set: Old validation set.
            - Test set: Old test set.
        - Model training:
            - Training: The model is trained from scratch on the new extended training set, using
              the same training procedure and the same hyperparameters as the original model.
            - Evaluation: The model is evaluated on the old validation and test sets.
            - Costs: The full retraining of the model is equally or more computationally expensive
              than the original training.
            - Performance: The full retraining of the model generally achieves the best performance.

    - INCREMENTAL TRAINING: Retrain the model on the new dataset incrementally.
        - Dataset splits:
            - Train set: All new data + uniformly sampled old train data (e.g., 1:1 ratio).
              - To preserve data distribution => avoid overfitting on new data and catastrophic
                forgetting on old data.
            - Validation set: Old validation set.
            - Test set: Old test set.
        - Model training:
            - Training: The pre-trained model is trained incrementally on the new training set,
              using the same training procedure, but with a reduced number of epochs and a lower
              learning rate.
            - Evaluation: The model is evaluated on the old validation and test sets.
            - Costs: The incremental retraining of the model is less computationally expensive
              than the full retraining.
            - Performance: The incremental retraining of the model generally achieves lower
              performance than the full retraining.

    - DISTILLATION TRAINING: Retrain the model using knowledge distillation on the new dataset.
        - Dataset splits:
            - Train set: All new data of a specific i-th user.
              - Uneven distribution of data => distillation cushions the problem of data
                imbalance by transferring knowledge from the old model to the new model.
            - Validation set: Old validation set.
            - Test set: Old test set.
        - Model training:
            - Training: The pre-trained model is trained using knowledge distillation on the
              new training set, using a distillation loss that combines the original loss
              and the distillation loss.
            - Evaluation: The model is evaluated on the old validation and test sets.
            - Costs: The distillation retraining of the model is less computationally expensive
              than the full retraining.
            - Performance: The distillation retraining of the model generally achieves lower
              performance than the full retraining. Notice that this kind of training in the long
              term suffers the risk of overfitting on the new data and of catastrophic forgetting
              on the old data. For this reason, it is recommended to alternate it with incremental
              training or full retraining.

    - FINE-TUNING: Fine-tune the model on the new dataset.
        - Dataset splits:
            - Train set: All new data.
            - Validation set: Old validation set.
            - Test set: Old test set.
        - Model training:
            - Training: The pre-trained model is fine-tuned on the new training set by modifying
              only the last layers of the model, using a reduced number of epochs and a lower
              learning rate.
            - Evaluation: The model is evaluated on the old validation and test sets.
            - Costs: The fine-tuning of the model is less computationally expensive than the
              full retraining.
            - Performance: The fine-tuning of the model generally achieves lower performance
              than the full retraining.
    """

    @classmethod
    def load_pretrained_model(
        cls,
        pretrained_model_filepath: str
    ) -> "GNNRetrainModelHandler":
        """
        Loads a pretrained GNN model handler and wrap it in a GNNRetrainModelHandler instance.

        This method bypasses the standard constructor (__init__) and instead restores the model 
        state from a saved checkpoint. Moreover, it initializes all necessary components to allow
        the retraining of the model on new train data (to be provided through the 
        `add_new_train_data` method).

        Parameters:
            pretrained_model_filepath (str): Filepath to the saved pretrained model handler 
                checkpoint (.pth format).

        Returns:
            GNNRetrainModelHandler: An instance of GNNRetrainModelHandler wrapping the loaded model 
                state.

        Notes:
            - This method avoids calling the __init__ constructor explicitly.
            - The restored instance inherits model parameters, dataset splits, etc. from the saved 
                checkpoint.
            - This is useful for retraining the model with an extended graph dataset.
        """
        # Load pre-trained GNN model handler
        model_handler = GNNModelHandler.load_pretrained_model(pretrained_model_filepath)

        # Extract the graph dataset handler and GNN encoder
        gdh = model_handler._gdh
        gnn_encoder = model_handler._gnn_encoder

        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Initialize instance using parent constructor
        super(GNNRetrainModelHandler, instance).__init__(gdh, gnn_encoder)

        # Wrap the gdh with an expandable graph dataset handler
        instance._gdh = ExpandableHeterogeneousGraphDatasetHandler(model_handler._gdh)
        instance._subgraph_dataset = None

        # Copy necessary attributes from the loaded model handler
        instance._graph_dataset = model_handler._graph_dataset

        instance._gnn_encoder = model_handler._gnn_encoder
        instance._model = model_handler._model

        instance._old_train_data, instance._old_val_data, instance._old_test_data = (
            model_handler._train_data,
            model_handler._val_data,
            model_handler._test_data
        )

        # Remove the edge_label_index from the old train data (which was added by the T.RandomLinkSplit)
        del instance._old_train_data["user", "rating", "movie"].edge_label_index

        # Enable GPU computation (if available)
        instance._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model training, validation, and test sets
        instance._train_data = None
        instance._val_data = instance._old_val_data     # Validation set must ALWAYS remain unchanged!
        instance._test_data = instance._old_test_data   # Validation set must ALWAYS remain unchanged!

        # Wrap the graph dataset handler with an expandable graph dataset handler
        instance._egdh = ExpandableHeterogeneousGraphDatasetHandler(gdh)

        # Make the gdh attribute pointing to the new egdh handler (to allow super class methods to work with the new handler)
        instance._gdh = instance._egdh  

        return instance

    def add_new_train_data(self, new_movies_df: pd.DataFrame= None, new_ratings_df: pd.DataFrame=None):
        """
        Adds new train data to the model handler to be used for retraining the pre-trained model.

        Parameters:
            new_movies_df (pd.DataFrame, optional): New movies DataFrame to be added to the graph 
                dataset. Default is None.
            new_ratings_df (pd.DataFrame, optional): New ratings DataFrame to be added to the graph
                dataset. Default is None.
        """
        # Check whether at least one of the new dataframes is provided
        if new_movies_df is None and new_ratings_df is None:
            raise ValueError("Both new_movies_df and new_ratings_df cannot be None.")

        if new_movies_df is not None:
            # Expand the graph dataset with new movies
            self._egdh.add_new_movies(new_movies_df)

            # Update the graph dataset with the new movies
            self._graph_dataset = self._egdh.get_graph_dataset()
            self._subgraph_dataset = self._egdh.get_subgraph_dataset()           

        if new_ratings_df is not None:
            # Expand the graph dataset with new ratings
            self._egdh.add_new_user_movie_ratings(new_ratings_df)

            # Update the graph dataset with the new ratings
            self._graph_dataset = self._egdh.get_graph_dataset()
            self._subgraph_dataset = self._egdh.get_subgraph_dataset()

        # Complete the subgraph with missing nodes (movie nodes or user nodes), if any
        if new_movies_df is None:
            self._subgraph_dataset['movie'].node_id = self._graph_dataset['movie'].node_id
            self._subgraph_dataset['movie'].x = self._graph_dataset['movie'].x

        if new_ratings_df is None:
            self._subgraph_dataset['user'].node_id = self._graph_dataset['user'].node_id
            self._subgraph_dataset['user'].x = self._graph_dataset['user'].x
    
    def full_retrain(
        self,
        num_epochs: int=300,
        lr: float=0.01,
        model_name: str=None,
        trained_model_path: str=None,
        store_tensorboard_training_plot: bool=True,
        enable_early_stopping: bool=True
    ):
        """
        Retrains the model from scratch on the entire updated dataset.
        
        Parameters:
            num_epochs (int, optional): Number of training epochs. Default is 300.
            lr (float, optional): Learning rate for training. Default is 0.01.
            model_name (str, optional): Custom name for saving the trained model.
            trained_model_path (str): Path to store the trained model. If None, the model will not
                be stored. Default is None.
            store_tensorboard_training_plot (bool, optional): Whether to store the TensorBoard plot. 
                Default is True.
            enable_early_stopping (bool, optional): Whether to enable early stopping mechanism.
                Default is True.
        """
        ## BUILD THE TRAINING SET
        # Check whether the subgraph contains new edges
        if self._subgraph_dataset is None or self._subgraph_dataset["user", "rating", "movie"] == {}:
            print("[WARNING] No new edges to add to the training set. Using old edges for full retraining.")

            # Use the entire graph dataset as training set for full retraining
            self._subgraph_dataset = self._graph_dataset

        # Merge the new train data with the sampled old train data (1:1 ratio)
        self._train_data = self._merge_train_data(self._old_train_data, self._subgraph_dataset) 

        ## FULL RETRAINING
        # Reset the GNN model to its initial state
        self._model = GNNModel(dataset=self._graph_dataset, gnn_encoder=self._gnn_encoder)

        # Perform the full retraining of the model
        super().train(
            num_epochs=num_epochs,
            lr=lr,
            model_name=model_name,
            trained_model_path=trained_model_path,
            store_tensorboard_training_plot=store_tensorboard_training_plot,
            enable_early_stopping=enable_early_stopping
        )

        # Clear the subgraph dataset to avoid reusing it in the next training
        self._subgraph_dataset = None

    def incremental_train(
        self,
        num_epochs: int=10,
        lr: float=0.001,
        model_name: str=None,
        trained_model_path: str=None,
        store_tensorboard_training_plot: bool=True,
        enable_early_stopping: bool=True
    ):
        """
        Retrains the model on the new dataset incrementally, using a 1:1 ratio between old and new
        training data to avoid overfitting on new data and catastrophic forgetting on old data.
        
        Parameters:
            num_epochs (int, optional): Number of training epochs. Default is 300.
            lr (float, optional): Learning rate for training. Default is 0.01.
            model_name (str, optional): Custom name for saving the trained model.
            trained_model_path (str): Path to store the trained model. If None, the model will not
                be stored. Default is None.
            store_tensorboard_training_plot (bool, optional): Whether to store the TensorBoard plot. 
                Default is True.
            enable_early_stopping (bool, optional): Whether to enable early stopping mechanism.
                Default is True.
        """
        ## BUILD THE TRAINING SET
        # Check whether the training set contains new edges
        if self._subgraph_dataset is None or self._subgraph_dataset["user", "rating", "movie"] == {}:
            raise ValueError(
                "Impossible to re-train the model: no new edges to add to the training set. "
                f"Current subgraph structure: {self._subgraph_dataset}"
            )

        # Compute the total size of the new training data
        new_train_data_size = self._subgraph_dataset["user", "rating", "movie"].edge_index.size(1)

        # Compute the validation and test ratios to preserve a 1:1 ratio between old and new training data
        old_train_data_size = self._old_train_data["user", "rating", "movie"].edge_index.size(1)
        val_ratio = test_ratio = (old_train_data_size - new_train_data_size) / (2 * old_train_data_size)

        # Sample old train data to preserve the 1:1 ratio
        sampled_old_train_data, _, _ = self._edge_level_dataset_split(
            graph_dataset=self._old_train_data,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Merge the new train data with the sampled old train data (1:1 ratio)
        self._train_data = self._merge_train_data(self._subgraph_dataset, sampled_old_train_data)

        ## INCREMENTAL TRAINING
        # Perform the incremental training of the model
        super().train(
            num_epochs=num_epochs,
            lr=lr,
            model_name=model_name,
            trained_model_path=trained_model_path,
            store_tensorboard_training_plot=store_tensorboard_training_plot,
            enable_early_stopping=enable_early_stopping
        )

        # Clear the subgraph dataset to avoid reusing it in the next training
        self._subgraph_dataset = None

    def distillation_train(
        self,
        num_epochs: int = 50,
        lr: float = 0.001,
        temperature: float = 1.0,
        alpha: float = 0.5,
        model_name: str = None,
        trained_model_path: str = None,
        store_tensorboard_training_plot: bool = True,
        enable_early_stopping: bool = True
    ):
        """
        Trains the student model using knowledge distillation from the teacher model.

        Knowledge distillation involves training the new model with the distillation loss, where the
        model’s new predictions (on the i-th user data) are aligned with the predictions from the 
        old model. In this case, the old model acts as the teacher, and the new model tries to learn
        from it while still focusing on the i-th user's preferences.

        Pros: The new model retains some of the broader knowledge from the original model (preventing 
        catastrophic forgetting) while specializing on the i-th user's data.

        Cons: This technique is more complex than the others since we need to align the predictions 
        with the old model.

        Parameters: 
            num_epochs (int, optional): Number of training epochs. Default is 300.
            lr (float, optional): Learning rate for training. Default is 0.01.
            temperature (float, optional): Softening factor for distillation. Default is 1.0.
            alpha (float, optional): Weight for distillation loss vs. MSE loss. Default is 0.5.
            model_name (str, optional): Custom name for saving the trained model.
            trained_model_path (str): Path to store the trained model. If None, the model will not
                be stored. Default is None.
            store_tensorboard_training_plot (bool, optional): Whether to store the TensorBoard plot. 
                Default is True.
            enable_early_stopping (bool, optional): Whether to enable early stopping mechanism.
                Default is True.
        """
        ## BUILD THE TRAINING SET
        # Check whether the training set contains new edges
        if self._subgraph_dataset is None or self._subgraph_dataset["user", "rating", "movie"] == {}:
            raise ValueError(
                "Impossible to re-train the model: no new edges to add to the training set. "
                f"Current subgraph structure: {self._subgraph_dataset}"
            )
        
        # Compute the total size of the new training data
        new_train_data_size = self._subgraph_dataset["user", "rating", "movie"].edge_index.size(1)

        # Compute the validation and test ratios to preserve a 1:1 ratio between old and new training data
        old_train_data_size = self._old_train_data["user", "rating", "movie"].edge_index.size(1)
        val_ratio = test_ratio = (old_train_data_size - new_train_data_size) / (2 * old_train_data_size)

        # Sample old train data to preserve the 1:1 ratio
        sampled_old_train_data, _, _ = self._edge_level_dataset_split(
            graph_dataset=self._old_train_data,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Merge the new train data with the sampled old train data (1:1 ratio)
        self._train_data = self._merge_train_data(self._subgraph_dataset, sampled_old_train_data)

        ## DISTILLATION TRAINING
        print(f"Device: '{self._device}'")

        # Initialize the teacher and the student models
        teacher_model = copy.deepcopy(self._model)  # Deep copy
        student_model = self._model                 # Shallow copy

        # Move the student and the teacher model on device
        student_model = student_model.to(self._device)
        teacher_model = teacher_model.to(self._device)

        # Move data on device
        self._train_data = self._train_data.to(self._device)
        self._val_data = self._val_data.to(self._device)
        
        # Define the optimizer
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

        # Define the distillation loss
        def distillation_loss(student_logits, teacher_logits, target):
            """
            Compute the distillation loss as the weighted sum of:
                - KL-divergence loss: computed on softmax distributions over predictions of the 
                    teacher and student models.
                - MSE loss: used to fit the student model to the true labels.
            Formula:
                loss = alpha * KD_loss + (1 - alpha) * MSE_loss
            where alpha is a tunable weight that controls the balance between distillation loss and
            supervised loss
            """
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            mse_loss = F.mse_loss(student_logits, target)
            return alpha * kd_loss + (1 - alpha) * mse_loss

        # Define a single training step
        def train_step(train_data):
            # Set the student model in training mode
            student_model.train()
            optimizer.zero_grad()

            # Forward pass: estimate the predicted rating
            student_pred = student_model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data['user', 'movie'].edge_label_index
            )
            teacher_pred = teacher_model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data['user', 'movie'].edge_label_index
            ).detach()

            # Extract the ground truth rating
            target = train_data['user', 'movie'].edge_label

            # Compute the distillation loss
            loss = distillation_loss(student_pred, teacher_pred, target)

            # Backward pass: compute the gradients of the model and update model weights
            loss.backward()
            optimizer.step()

            return float(loss)

        # Define a single evaluation step
        @torch.no_grad()
        def evaluation_step(data):
            # Set the model in evaluation mode
            student_model.eval()

            # Forward pass: estimate the predicted rating
            pred = student_model(
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
            model_name = self._gnn_encoder.model_name + "_distilled_model"

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
            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None
            best_model_epoch = num_epochs
            patience = self._compute_adaptive_patience(num_epochs)
            print(f"Adaptive patience set to {patience} epochs based on num_epochs={num_epochs}.")

        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Training step
            loss = train_step(self._train_data)

            # Evaluation step
            train_rmse, train_mae = evaluation_step(self._train_data)
            val_rmse, val_mae = evaluation_step(self._val_data)

            # Print training information
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")

            # TensorBoard logging
            if store_tensorboard_training_plot:
                writer.add_scalar('RMSE/train', train_rmse, epoch)
                writer.add_scalar('RMSE/val', val_rmse, epoch)
                writer.add_scalar('MAE/train', train_mae, epoch)
                writer.add_scalar('MAE/val', val_mae, epoch)

            # Early stopping
            if enable_early_stopping and epoch > num_epochs / 2:    # Prevent early oscillations
                # If validation RMSE does not improve for 'patience' epochs, stop training
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    best_model_state = student_model.state_dict()  # Save best model
                    best_model_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}. Best validation RMSE: {best_val_rmse:.4f} at epoch {best_model_epoch}.")
                        break  # Stop training

        # Restore the best model
        if enable_early_stopping and best_model_state is not None:
            student_model.load_state_dict(best_model_state)
            print(f"Best model restored from epoch {best_model_epoch}.")

        # (Redundant operation) Overwrite the pre-trained model with the distilled model
        self._model = student_model

        # Close TensorBoard writer
        if store_tensorboard_training_plot:
            writer.close()
            print(f"TensorBoard training saved at: {tensorboard_foldername}")

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

        # Clear the subgraph dataset to avoid reusing it in the next training
        self._subgraph_dataset = None

    def fine_tune(
        self,
        num_epochs: int=50,
        lr: float=0.001,
        model_name: str=None,
        trained_model_path: str=None,
        store_tensorboard_training_plot: bool=True,
        enable_early_stopping: bool=True
    ):
        """
        Fine-tunes only the last layers of the model, which correspond to the decoder layers
        (EdgeRegressionDecoder), on the new dataset.
        
        Parameters:
            num_epochs (int, optional): Number of training epochs. Default is 50.
            lr (float, optional): Learning rate for training. Default is 0.001.
            model_name (str, optional): Custom name for saving the trained model.
            trained_model_path (str): Path to store the trained model. If None, the model will not
                be stored. Default is None.
            store_tensorboard_training_plot (bool, optional): Whether to store the TensorBoard plot
                of the model training. Default is True.
            enable_early_stopping (bool, optional): Whether to enable early stopping mechanism.
                Default is True.
        """
        ## BUILD THE TRAINING SET
        # Check whether the training set contains new edges
        if self._subgraph_dataset is None or self._subgraph_dataset["user", "rating", "movie"] == {}:
            raise ValueError(
                "Impossible to re-train the model: no new edges to add to the training set. "
                f"Current subgraph structure: {self._subgraph_dataset}"
            )
        
        # Use only the new training data as the training set
        # Note: use _edge_level_dataset_split to add the edge_label_index field to the train data 
        self._train_data, _, _ = self._edge_level_dataset_split(
            graph_dataset=self._subgraph_dataset,
            val_ratio=0.0,
            test_ratio=0.0,
        )

        ## FINE-TUNING
        print(f"Device: '{self._device}'")

        # Move the model on device
        self._model = self._model.to(self._device)

        # Freeze encoder layers and unfreeze decoder layers
        for param in self._model.encoder.parameters():
            param.requires_grad = False

        for param in self._model.decoder.parameters():
            param.requires_grad = True
        
        # Move data on device
        self._train_data = self._train_data.to(self._device)
        self._val_data = self._val_data.to(self._device)

        # Optimizer setup: fine-tuning only decoder parameters
        optimizer = torch.optim.Adam(self._model.decoder.parameters(), lr=lr)

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

        # Initialize TensorBoard writer if required
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
            patience = self._compute_adaptive_patience(num_epochs)
            print(f"Adaptive patience set to {patience} epochs based on num_epochs={num_epochs}.")

        # Fine-tuning loop
        for epoch in range(1, num_epochs + 1):
            # Training step
            loss = train_step(self._train_data)

            # Evaluation step
            train_rmse, train_mae = evaluation_step(self._train_data)
            val_rmse, val_mae = evaluation_step(self._val_data)

            # Print training information
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")

            # TensorBoard logging
            if store_tensorboard_training_plot:
                writer.add_scalar('RMSE/train', train_rmse, epoch)
                writer.add_scalar('RMSE/val', val_rmse, epoch)
                writer.add_scalar('MAE/train', train_mae, epoch)
                writer.add_scalar('MAE/val', val_mae, epoch)

            # Early stopping logic
            if enable_early_stopping and epoch > num_epochs / 2:  # Prevent early oscillations
                # If validation RMSE does not improve for 'patience' epochs, stop training
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    best_model_state = self._model.state_dict()
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

        # Clear the subgraph dataset to avoid reusing it in the next training
        self._subgraph_dataset = None

    def _merge_train_data(self, train_data1: HeteroData, train_data2: HeteroData) -> HeteroData:
        """
        Merges two HeteroData training datasets into a new HeteroData instance.

        Parameters:
            train_data1 (HeteroData): The first training dataset.
            train_data2 (HeteroData): The second training dataset.

        Returns:
            HeteroData: A new HeteroData instance containing merged data from both inputs.
        """
        train_data1.to("cpu")
        train_data2.to("cpu")

        # Merge the two training datasets
        merged_data = merge_hetero_data(train_data1, train_data2)

        # Add the additional field needed for the model training (as done by _edge_level_dataset_split function through T.RandomLinkSplit)
        merged_data['user', 'rating', 'movie'].edge_label_index = merged_data['user', 'rating', 'movie'].edge_index

        return merged_data