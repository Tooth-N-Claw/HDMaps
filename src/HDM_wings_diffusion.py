import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def load_point_cloud(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(','))
            points.append((x, y))
    #return np.array(points)
    return np.array(points[:6])

def compute_distance_matrix(file_paths):
    data_list = [load_point_cloud(fp) for fp in file_paths]
    flattened_data = [d.flatten() for d in data_list]  # Ensure uniform shape
    D = squareform(pdist(flattened_data, metric='euclidean'))
    return D

def parse_metadata(file_name):
    meta = {}
    parts = file_name.split("_")
    meta['group'] = parts[0]
    meta['species'] = parts[1]
    meta['temp'] = parts[2]
    meta['sex'] = parts[3]
    meta['id'] = parts[4]
    meta['side'] = parts[5].split(".")[0]  # Remove file extension
    return meta

def summarize_distance_matrix(D, metadata):
    plt.figure(figsize=(8, 6))
    plt.imshow(D, cmap='viridis', aspect='auto')
    plt.colorbar(label='Distance')
    plt.title('Original Distance Matrix')
    # plt.show()
    
    # Sort indices by species
    species_order = sorted(set(meta['species'] for meta in metadata))
    species_map = {species: i for i, species in enumerate(species_order)}
    sorted_indices = sorted(range(len(metadata)), key=lambda i: species_map[metadata[i]['species']])
    
    D_permuted = D[np.ix_(sorted_indices, sorted_indices)]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(D_permuted, cmap='viridis', aspect='auto')
    plt.colorbar(label='Distance')
    plt.title('Permuted Distance Matrix by Species')
    # plt.show()
    
    median_distance = np.median(D)
    print(f'Median Distance: {median_distance}')
    
    return median_distance

def run_diffusion_maps(D, n_components=3, alpha=0.5):
    dmap = dm.DiffusionMap(alpha=alpha, n_evecs=n_components)
    embedding = dmap.fit_transform(D)
    return embedding

def plot_embedding(fig_folder, embedding, metadata, color_key):
    # Get unique labels and create a mapping from label to index
    unique_labels = list(set(meta[color_key] for meta in metadata))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)  # 'tab20' for up to 20 distinct colors
    
    # Map each metadata entry to its corresponding index in the color map
    colors = [label_map[meta[color_key]] for meta in metadata]
    
    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=0.5)
    
    # Create legend with species names
    handles = []
    for label in unique_labels:
        color = cmap(label_map[label] / (num_colors - 1))  # Get the color from the colormap
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    plt.legend(handles=handles, title=color_key)
    
    # Formatting
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Diffusion Map Embedding Colored by {color_key}')

    # Save plot to the specified folder
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)  # Create the folder if it doesn't exist
    
    plt.savefig(os.path.join(fig_folder, f"{color_key}.png"))

def prepare_distance_matrix(data_dir):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
    metadata = [parse_metadata(os.path.basename(fp)) for fp in file_paths]
    D = compute_distance_matrix(file_paths)
    return D, metadata


def diffusion_maps(D, n_components=3, epsilon=1.0):
    # Step 1: Compute affinity matrix
    K = np.exp(-D**2 / epsilon)

    # Step 2: Efficiently set the top 20 values to non-zero
    num_top_values = 20
    for i in range(K.shape[0]):
        # Get the indices of the top 20 values in each row (sorted by descending order)
        top_indices = np.argsort(K[i, :])[-num_top_values:]
        
        # Create a mask for the top 20 values (all others will be zero)
        mask = np.zeros(K.shape[1], dtype=bool)
        mask[top_indices] = True
        
        # Apply the mask to keep only the top 20 values in K[i, :]
        K[i, ~mask] = 0

    # Step 3: Convert K to a sparse matrix
    K_sparse = csr_matrix(K)
    K = (K + K.T)/2
    P = K / K.sum(axis=1, keepdims=True)  # Row normalization
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigsh(P, k=10, which='LM')
    
    # Step 4: Sort by largest eigenvalues
    idx = np.argsort(-eigvals)[:n_components]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    return eigvals, eigvecs





# You can reuse your existing functions for loading data and computing the distance matrix
# Here I'll add the MDS implementation

def perform_mds(D, n_components=3, random_state=42):
    """
    Perform Multidimensional Scaling on a distance matrix.
    
    Parameters:
    -----------
    D : numpy.ndarray
        Distance matrix (shape: n_samples x n_samples)
    n_components : int, optional (default=3)
        Number of dimensions to reduce to
    random_state : int, optional (default=42)
        Random state for reproducibility
    
    Returns:
    --------
    embedding : numpy.ndarray
        The MDS embedding (shape: n_samples x n_components)
    """
    # Initialize MDS
    mds = MDS(n_components=n_components, 
              dissimilarity='precomputed',
              random_state=random_state)
    
    # Fit MDS to distance matrix
    embedding = mds.fit_transform(D)
    
    return embedding

def plot_mds_embedding(embedding, metadata, color_key, fig_folder=None):
    """
    Plot the MDS embedding, coloring points by a metadata field.
    
    Parameters:
    -----------
    embedding : numpy.ndarray
        MDS embedding (n_samples x n_components)
    metadata : list of dict
        Metadata for each point
    color_key : str
        The metadata key to use for coloring (e.g., 'species')
    fig_folder : str, optional
        Folder to save the plot (if None, the plot is just displayed)
    """
    # Get unique labels and create a mapping from label to index
    unique_labels = list(set(meta[color_key] for meta in metadata))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)  # 'tab20' for up to 20 distinct colors
    
    # Map each metadata entry to its corresponding index in the color map
    colors = [label_map[meta[color_key]] for meta in metadata]
    
    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=0.8)
    
    # Create legend with labels
    handles = []
    for label in unique_labels:
        color = cmap(label_map[label] / (num_colors - 1))
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    plt.legend(handles=handles, title=color_key)
    
    # Formatting
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title(f'MDS Embedding Colored by {color_key}')
    
    # Save plot if folder is specified
    if fig_folder:
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        plt.savefig(os.path.join(fig_folder, f"mds_{color_key}.png"))
        
    plt.show()

def plot_mds_embedding_3d(embedding, metadata, color_key, fig_folder=None):
    """
    Plot 3D MDS embedding, coloring points by a metadata field.
    
    Parameters:
    -----------
    embedding : numpy.ndarray
        MDS embedding (n_samples x n_components)
    metadata : list of dict
        Metadata for each point
    color_key : str
        The metadata key to use for coloring (e.g., 'species')
    fig_folder : str, optional
        Folder to save the plot (if None, the plot is just displayed)
    """
    # Get unique labels and create a mapping from label to index
    unique_labels = list(set(meta[color_key] for meta in metadata))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)  # 'tab20' for up to 20 distinct colors
    
    # Map each metadata entry to its corresponding index in the color map
    colors = [label_map[meta[color_key]] for meta in metadata]
    
    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                         c=colors, cmap=cmap, alpha=0.8, s=80)
    
    # Create legend with labels
    handles = []
    for label in unique_labels:
        color = cmap(label_map[label] / (num_colors - 1))
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    ax.legend(handles=handles, title=color_key)
    
    # Set multiple viewing angles for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Formatting
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_zlabel('MDS Dimension 3')
    ax.set_title(f'3D MDS Embedding Colored by {color_key}')
    
    # Save plot if folder is specified
    if fig_folder:
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        plt.savefig(os.path.join(fig_folder, f"mds3d_{color_key}.png"))
        
    plt.show()


def compute_diffusion_distance(D, epsilon=1.0, t=1, k=10):

    # Step 1: Compute affinity matrix (Gaussian kernel)
    K = np.exp(-D**2 / epsilon)
    
    # Optional: Sparsify the kernel (keep only top connections)
    num_top_values = 20
    for i in range(K.shape[0]):
        top_indices = np.argsort(K[i, :])[-num_top_values:]
        mask = np.zeros(K.shape[1], dtype=bool)
        mask[top_indices] = True
        K[i, ~mask] = 0
    
    # Ensure symmetry
    K = (K + K.T) / 2
    
    # Step 2: Row normalization to create Markov transition matrix
    d = np.sum(K, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    P = np.diag(d_inv_sqrt) @ K @ np.diag(d_inv_sqrt)  # Symmetric normalization
    
    # Step 3: Compute eigendecomposition
    eigvals, eigvecs = eigsh(P, k=k, which='LM')
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(-eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    # Step 4: Compute diffusion map embedding (scale eigenvectors by eigenvalues)
    diffusion_coords = eigvecs @ np.diag(eigvals**t)
    
    # Step 5: Compute pairwise diffusion distances
    # L2 distances in the diffusion space, weighted by eigenvalues
    diffusion_distances = np.zeros((diffusion_coords.shape[0], diffusion_coords.shape[0]))
    
    for i in range(diffusion_coords.shape[0]):
        for j in range(diffusion_coords.shape[0]):
            diffusion_distances[i, j] = np.sqrt(np.sum((diffusion_coords[i, :] - diffusion_coords[j, :])**2))
    
    return diffusion_distances, eigvals, eigvecs

def compare_mds_approaches(D, metadata, fig_folder=None):
    # Create subfolder for comparison plots if needed
    if fig_folder:
        comparison_folder = os.path.join(fig_folder, "comparison")
        if not os.path.exists(comparison_folder):
            os.makedirs(comparison_folder)
    else:
        comparison_folder = None
    
    # 1. MDS on original distances
    mds_original = perform_mds(D, n_components=2)
    
    # 2. Compute diffusion distances
    md = np.median(D)  # Use median distance as epsilon
    diffusion_distances, eigvals, eigvecs = compute_diffusion_distance(D, epsilon=md, t=1, k=10)
    
    # 3. MDS on diffusion distances
    mds_diffusion = perform_mds(diffusion_distances, n_components=2)
    
    # 4. Direct use of top diffusion eigenvectors
    diffusion_embedding = eigvecs[:, 1:3]  # Skip first eigenvector (constant)
    
    # 5. Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get unique labels and create color mapping
    color_key = "species"  # You can change this to any metadata field
    unique_labels = list(set(meta[color_key] for meta in metadata))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)
    colors = [label_map[meta[color_key]] for meta in metadata]
    
    # Plot original MDS
    axes[0].scatter(mds_original[:, 0], mds_original[:, 1], c=colors, cmap=cmap, alpha=0.8)
    axes[0].set_title("MDS on Original Distances")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    
    # Plot diffusion MDS
    axes[1].scatter(mds_diffusion[:, 0], mds_diffusion[:, 1], c=colors, cmap=cmap, alpha=0.8)
    axes[1].set_title("MDS on Diffusion Distances")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    
    # Plot direct diffusion map embedding
    axes[2].scatter(diffusion_embedding[:, 0], diffusion_embedding[:, 1], c=colors, cmap=cmap, alpha=0.8)
    axes[2].set_title("Direct Diffusion Map Embedding")
    axes[2].set_xlabel("Eigenvector 1")
    axes[2].set_ylabel("Eigenvector 2")
    
    # Add legend
    handles = []
    for label in unique_labels:
        color = cmap(label_map[label] / (num_colors - 1))
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    # Place legend outside the plots
    fig.legend(handles=handles, title=color_key, loc='center right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    
    # Save comparison plot
    if comparison_folder:
        plt.savefig(os.path.join(comparison_folder, f"mds_comparison_{color_key}.png"), dpi=300)
    
    plt.show()
    
    return mds_original, mds_diffusion, diffusion_embedding

# Example usage (to replace your current diffusion maps code):
D, metadata = prepare_distance_matrix("data/v3 Landmarks_and_centroids and intersection_1500/Landmarks")
md = summarize_distance_matrix(D, metadata)

# # 1. Standard MDS on original distance matrix
fig_folder = "data/figures/normal_distance"
mds_embedding = perform_mds(D, n_components=3)
plot_mds_embedding_3d(mds_embedding, metadata, color_key="species", fig_folder=fig_folder)

# # 2. Compute diffusion distances
diffusion_distances, eigvals, eigvecs = compute_diffusion_distance(D, epsilon=md, t=1, k=10)

fig_folder = "data/figures/diffusion_distance"
mds_diffusion = perform_mds(diffusion_distances, n_components=3)
plot_mds_embedding_3d(mds_diffusion[:, :3], metadata, color_key="species", 
                   fig_folder=fig_folder)

