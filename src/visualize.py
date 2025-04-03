import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os


def shire_visualize_3d(fig_folder, embedding, species_labels, title_suffix="Species"):
    """
    Plot 3D embedding colored by species
    
    Parameters:
    -----------
    fig_folder : str
        Folder to save the figure
    embedding : numpy.ndarray
        The coordinates (n_samples, n_dimensions), should have at least 3 columns
    species_labels : list
        List of species labels (strings)
    title_suffix : str
        Text to add to the plot title
    """
    # Ensure we have 3D data
    if embedding.shape[1] < 3:
        raise ValueError("Embedding must have at least 3 dimensions for 3D plotting")
    
    # Get unique species and create a mapping
    unique_species = sorted(set(species_labels))
    species_map = {species: i for i, species in enumerate(unique_species)}
    
    # Map each species to its corresponding index for coloring
    colors = [species_map[species] for species in species_labels]
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_species)
    cmap = cm.get_cmap('tab20', max(num_colors, 20))
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color by species
    for species in unique_species:
        indices = [i for i, s in enumerate(species_labels) if s == species]
        color = cmap(species_map[species] / (max(num_colors - 1, 1)))
        ax.scatter(
            embedding[indices, 0], 
            embedding[indices, 1], 
            embedding[indices, 2],
            c=[color], 
            label=species,
            alpha=0.7,
            s=50
        )
    
    # Add labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f'3D MDS Embedding Colored by {title_suffix}')
    
    # Add legend
    ax.legend(title="Species", loc="best")
    
    # Add stress value if available in a corner of the plot
    if hasattr(embedding, 'stress_'):
        fig.text(0.02, 0.02, f"Stress: {embedding.stress_:.4f}", ha="left")
    
    # Enable interactive rotation
    plt.tight_layout()
    
    # Save plot
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    
    plt.savefig(os.path.join(fig_folder, f"mds_3d_embedding_{title_suffix.lower()}.png"), dpi=300)
    
    # This makes the plot interactive - user can rotate to see different angles
    plt.show()
    
    return fig, ax

def shira_visualize(fig_folder, embedding, species_labels, title_suffix="Species"):
    """
    Plot 2D embedding colored by species
    
    Parameters:
    -----------
    fig_folder : str
        Folder to save the figure
    embedding : numpy.ndarray
        The coordinates (n_samples, n_dimensions)
    species_labels : list
        List of species labels (strings)
    title_suffix : str
        Text to add to the plot title
    """
    # Get unique species and create a mapping
    unique_species = sorted(set(species_labels))
    species_map = {species: i for i, species in enumerate(unique_species)}
    
    # Map each species to its corresponding index for coloring
    colors = [species_map[species] for species in species_labels]
    
    # Use a colormap that provides enough distinct colors
    num_colors = len(unique_species)
    cmap = cm.get_cmap('tab20', max(num_colors, 20))  # 'tab20' for up to 20 distinct colors
    
    # Plotting - use only first two dimensions for 2D plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=0.7, s=50)
    
    # Create legend with species names
    handles = []
    for species in unique_species:
        color = cmap(species_map[species] / (max(num_colors - 1, 1)))  # Get color from colormap
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=color, markersize=10, label=species)
        handles.append(handle)
    
    plt.legend(handles=handles, title="Species", loc="best")
    
    # Formatting
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f'MDS Embedding Colored by {title_suffix}')
    
    # Add stress value if available
    if hasattr(embedding, 'stress_'):
        plt.figtext(0.02, 0.02, f"Stress: {embedding.stress_:.4f}", ha="left")
    
    # Save plot
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    
    plt.savefig(os.path.join(fig_folder, f"mds_embedding_{title_suffix.lower()}.png"), dpi=300)
    plt.show()

def visualize(points: np.ndarray, data_samples_types: list[str] = None) -> None:
    print(points.shape)
    print(len(data_samples_types) if data_samples_types else "No labels provided")
    
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    
    if data_samples_types is not None:
        if len(data_samples_types) != len(points):
            raise ValueError("data_samples_types must have the same length as points")
        
        unique_species = sorted(set(data_samples_types))
        num_species = len(unique_species)
        
        species_to_index = {species: i for i, species in enumerate(unique_species)}
        indices = np.array([species_to_index[species] for species in data_samples_types])
        
        point_cloud["species"] = indices
        cmap = cm.get_cmap("rainbow", num_species)
        norm = Normalize(vmin=0, vmax=num_species-1)
        
        plotter.add_mesh(point_cloud, scalars="species", point_size=15, 
                         render_points_as_spheres=True, cmap="rainbow", 
                         clim=[0, num_species-1], show_scalar_bar=False)
        
        # Create legend with color swatches
        legend_entries = []
        for species, index in species_to_index.items():
            color = cmap(norm(index))[:3]  # Extract RGB
            legend_entries.append([species, color])
        
        plotter.add_legend(legend_entries, bcolor=(0, 0, 0))  # White background
    else:
        plotter.add_mesh(point_cloud, color="red", point_size=5, render_points_as_spheres=True)
    # plotter.save_graphic(os.path.join(fig_folder, "3d_plot.pdf"))

    plotter.show()
    # save the plot
