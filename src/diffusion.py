import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mds_embed import embed_plot

folder_path = 'data/v3 Landmarks_and_centroids and intersection_1500/Landmarks'


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
    # plt.figure(figsize=(8, 6))
    # plt.imshow(D, cmap='viridis', aspect='auto')
    # plt.colorbar(label='Distance')
    # plt.title('Original Distance Matrix')
    # plt.show()
    
    # Sort indices by species
    species_order = sorted(set(meta['species'] for meta in metadata))
    species_map = {species: i for i, species in enumerate(species_order)}
    sorted_indices = sorted(range(len(metadata)), key=lambda i: species_map[metadata[i]['species']])
    
    D_permuted = D[np.ix_(sorted_indices, sorted_indices)]
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(D_permuted, cmap='viridis', aspect='auto')
    # plt.colorbar(label='Distance')
    # plt.title('Permuted Distance Matrix by Species')
    # plt.show()
    
    median_distance = np.median(D)
    print(f'Median Distance: {median_distance}')
    
    return median_distance


def plot_embedding(fig_folder, embedding, metadata, color_key, filename):
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
    
    plt.savefig(os.path.join(fig_folder, filename))

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

def plot_embedding_3d(embedding, metadata, color_key):
    # Get unique labels and create a mapping from label to index
    unique_labels = list(set(meta[color_key] for meta in metadata))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', num_colors)  # 'tab20' for up to 20 distinct colors
    
    # Map each metadata entry to its corresponding index in the color map
    colors = [label_map[meta[color_key]] for meta in metadata]
    
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, cmap=cmap, alpha=0.75)
    
    # Create legend with species names
    handles = []
    for label in unique_labels:
        color = cmap(label_map[label] / (num_colors - 1))  # Get the color from the colormap
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        handles.append(handle)
    
    ax.legend(handles=handles, title=color_key)
    ax.view_init(elev=30, azim=45)
    
    # Formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Diffusion Map Embedding Colored by {color_key}')
    
    plt.show()



D, metadata = prepare_distance_matrix(folder_path)
md = summarize_distance_matrix(D, metadata)

eigvals, eigvecs = diffusion_maps(D, n_components=4, epsilon=md)
fig_folder = "data/wings/figures"


plot_embedding(fig_folder, eigvecs[:, 1:3], metadata, color_key="species", filename="diffusion_species_2d.png")
# plot_embedding_3d( eigvecs[:, 1:4], metadata, color_key="species")

# points = embed_plot(eigvecs[:, 1:3])
# plot_embedding(points, metadata, color_key="species", filename="mds_diffusion_species_2d.png")
