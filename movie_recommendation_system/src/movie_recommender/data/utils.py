import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

RANDOM_SEED = 42  # Default seed for reproducibility

def _create_projection_matrix(
    max_users: int,
    feature_dim: int,
    sparse_vector: bool=True,
    sparsity: float=0.1,
    seed: int=42,
) -> torch.Tensor:
    """
    Helper function to create fixed random projection matrix.

    This function generates a random projection matrix that can be used to project user features 
    into a lower-dimensional space.

    Parameters:
        - max_users (int): Maximum number of users to accommodate in the projection matrix.
        - feature_dim (int): Fixed dimensionality of the user feature vectors.
        - sparse_vector (bool): Whether to generate sparse user features. If True, the matrix will 
            have a high proportion of zero entries. Default is True.
        - sparsity (float): Probability of nonzero entries (only applies if sparse=True). Default 
            is 0.1.
        - seed (int): Random seed for reproducibility. Default is 42.
    
    Returns:
        torch.Tensor: The generated projection matrix.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    if sparse_vector:
        # Sparse random projection: only a fraction of values are nonzero
        projection_matrix = torch.zeros(feature_dim, max_users)
        mask = torch.rand(feature_dim, max_users) < sparsity
        projection_matrix[mask] = torch.randn(mask.sum())  # Sparse random values
    else:
        # Dense random projection: standard Gaussian matrix
        projection_matrix = torch.randn(feature_dim, max_users)
    return projection_matrix


def generate_users_external_features(
    users_ratings_df: pd.DataFrame,
    max_users: int=1000000,  # By default, support up to 1 million users
    feature_dim: int=100,
    sparse_vector: bool=True,
    sparsity: float=0.1,
) -> torch.Tensor:
    """
    Constructs fixed-size user feature vectors using a predefined random projection.

    **Mathematical Formulation:**
    Let `U` be the one-hot encoding matrix for users, of size `(num_users × max_users)`,
    where each row represents a user with a single `1` in its corresponding column.
    We define a **fixed random projection matrix** `P` of size `(feature_dim × max_users)`,
    generated once and kept constant.

    The user feature matrix is obtained as:

        U = P ⋅ I_active_users

    Since each user corresponds to a single column in `P`, their feature vector is simply
    the column extracted from `P`, ensuring consistent representations across retraining sessions.

    **Pros & Cons:**
        - ✅ **Fixed Representation**: Ensures the same users always get the same features.
        - ✅ **Compact Representation**: Reduces dimensionality while preserving user uniqueness.
        - ✅ **Efficient**: Does not require retraining or recomputing old users.
        - ❌ **Not Updateable**: If new users exceed `max_users`, they cannot be represented.
        - ❌ **Not Interpretable**: Unlike learned embeddings, this does not capture user behavior.
        - ❌ **Not-unique Representation**: Does not guarantee absolutely unique feature 
            representations for all possible users, especially if the number of potential users
            (max_users) is very large compared to the feature dimension (feature_dim). However, it
            makes it highly probable that different users will have unique feature representations.

    Parameters:
        - users_ratings_df (pd.DataFrame): DataFrame containing user interactions.
        - max_users (int): Maximum number of users to accommodate in the projection matrix.
        - feature_dim (int): Fixed dimensionality of the user feature vectors.
        - sparse_vector (bool): Whether to generate sparse user features. If True, the matrix will 
            have a high proportion of zero entries. Default is True.
        - sparsity (float): Probability of nonzero entries (only applies if sparse=True). Default 
            is 0.1.

    Returns:
        torch.Tensor: Tensor of shape `(num_users, feature_dim)`, containing user features.
    """
    # Create an identity matrix representation for the users (one-hot encoding)
    user_ids = users_ratings_df['userId'].unique()
    user_ids = torch.tensor(user_ids)
    assert user_ids.max() < max_users, "User ID exceeds predefined limit"

    # Create fixed random projection matrix
    projection_matrix = _create_projection_matrix(
        max_users=max_users,
        feature_dim=feature_dim, 
        sparse_vector=sparse_vector,
        sparsity=sparsity, 
        seed=RANDOM_SEED
    )

    # Generate user features by selecting the corresponding columns from the projection matrix
    return projection_matrix[:, user_ids].T  # Shape: (num_users, feature_dim)


def merge_hetero_data(data1: HeteroData, data2: HeteroData) -> HeteroData:
    """
    Merges two HeteroData objects while preserving original node and edge IDs.

    Parameters:
        data1 (HeteroData): First data instance with 'user' and 'movie' nodes and their associated edges.
        data2 (HeteroData): Second data instance structured identically to data1.

    Returns:
        HeteroData: A merged data instance where:
            - 'user' nodes are merged by selecting the one with the largest number of unique user IDs.
            - 'movie' nodes are concatenated then deduplicated.
            - Edges (edge_index, edge_label, and y) are concatenated and then deduplicated, keeping 
                the first occurrence.
            - The graph is converted to an undirected version.

    Note:
        Deduplication is performed by scanning the concatenated tensors, retaining only the first 
        occurrence of each unique edge (based on its (source, target) tuple).
    """
    ## INITIALIZATION
    # Decouple the data to avoid modifying the original data
    data1 = data1.clone()
    data2 = data2.clone()

    # Initialize a new HeteroData object for the merged data
    merged_data = HeteroData()

    ## MERGE USER NODES
    # NOTE: This code is wrong in our case because the user nodes features are created as an 
    # identity matrix of size num_user_id, hence the two matrices have sizes:
    #   x = [num_user_id, num_user_id] 
    #   x' = [num_user_id', num_user_id'] != x
    # this means that we cannot merge two matrices whose feature sizes are different (unlike what we
    # have with the movie features).
    # However, since, by construction, one the two matrices is an extension of the other, we can 
    # just keep the one with the largest number of unique user nodes and discard the other one. 
    """ 
    # Merge Users by concatenation and deduplication
    merged_data['user'].node_id = torch.cat([data1['user'].node_id, data2['user'].node_id], dim=0)
    merged_data['user'].x = torch.cat([data1['user'].x, data2['user'].x], dim=0)
    orig_ids = merged_data['user'].node_id.tolist()
    seen = set()
    unique_ids = []
    unique_indices = []
    for i, uid in enumerate(orig_ids):
        if uid not in seen:
            seen.add(uid)
            unique_ids.append(uid)
            unique_indices.append(i)
    merged_data['user'].node_id = torch.tensor(unique_ids)
    merged_data['user'].x = merged_data['user'].x[unique_indices]
    user_mapping = {uid: uid for uid in unique_ids}  # identity mapping"
    """

    # Select the dataset with more unique user nodes
    unique_users_1 = set(data1['user'].node_id.tolist())
    unique_users_2 = set(data2['user'].node_id.tolist())
    if len(unique_users_1) >= len(unique_users_2):
        merged_data['user'].node_id = data1['user'].node_id.clone()
        merged_data['user'].x = data1['user'].x.clone()
    else:
        merged_data['user'].node_id = data2['user'].node_id.clone()
        merged_data['user'].x = data2['user'].x.clone()

    ## MERGE MOVIE NODES
    # NOTE: For movie nodes the features have the same size in both datasets, hence we can merge them:
    #   x = [num_movie_id, FIXED_SIZE]
    #   x' = [num_movie_id', FIXED_SIZE]

    # Merge Movies by concatenation and deduplication
    merged_data['movie'].node_id = torch.cat([data1['movie'].node_id, data2['movie'].node_id], dim=0)
    merged_data['movie'].x = torch.cat([data1['movie'].x, data2['movie'].x], dim=0)
    orig_ids = merged_data['movie'].node_id.tolist()
    seen = set()
    unique_ids = []
    unique_indices = []
    for i, nid in enumerate(orig_ids):
        if nid not in seen:
            seen.add(nid)
            unique_ids.append(nid)
            unique_indices.append(i)
    merged_data['movie'].node_id = torch.tensor(unique_ids)
    merged_data['movie'].x = merged_data['movie'].x[unique_indices]
    movie_mapping = {nid: nid for nid in unique_ids}  # identity mapping

    ## MERGE EDGES
    # Merge Edges without remapping (using identity mapping)
    merged_data['user', 'rating', 'movie'].edge_index = torch.cat([
        data1['user', 'rating', 'movie'].edge_index, 
        data2['user', 'rating', 'movie'].edge_index
    ], dim=1)

    merged_data['user', 'rating', 'movie'].edge_label = torch.cat([
        data1['user', 'rating', 'movie'].edge_label,
        data2['user', 'rating', 'movie'].edge_label
    ], dim=0)

    merged_data['user', 'rating', 'movie'].y = torch.cat([
        data1['user', 'rating', 'movie'].y,
        data2['user', 'rating', 'movie'].y
    ], dim=0)

    # Deduplicate edges while preserving original edge IDs
    edge_index = merged_data['user', 'rating', 'movie'].edge_index
    edge_label = merged_data['user', 'rating', 'movie'].edge_label
    y = merged_data['user', 'rating', 'movie'].y
    unique_indices = []
    seen_edges = set()
    for i in range(edge_index.shape[1]):
        edge = (edge_index[0, i].item(), edge_index[1, i].item())
        if edge not in seen_edges:
            seen_edges.add(edge)
            unique_indices.append(i)
    merged_data['user', 'rating', 'movie'].edge_index = edge_index[:, unique_indices]
    merged_data['user', 'rating', 'movie'].edge_label = edge_label[unique_indices]
    merged_data['user', 'rating', 'movie'].y = y[unique_indices]

    # Convert the merged graph to an undirected graph
    merged_data = T.ToUndirected()(merged_data)

    return merged_data
