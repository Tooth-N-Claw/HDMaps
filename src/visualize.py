import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os


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
        
        plotter.add_mesh(point_cloud, scalars="species", point_size=10, 
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
