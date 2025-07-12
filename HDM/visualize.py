import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os


def visualize(points: np.ndarray) -> None:
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    
    plotter.add_mesh(point_cloud, color="red", point_size=5, render_points_as_spheres=True)

    plotter.show()


def visualize_corresponding_points(points: np.ndarray, num_samples) -> None:
    
    # NOTE!! only works for wings!
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    print(np.arange(6))
    scalars = np.tile(np.arange(6), num_samples)
    print(scalars)
    cmap = cm.get_cmap("rainbow", 6)
    norm = Normalize(vmin=0, vmax=6-1)
    
    plotter.add_mesh(point_cloud, scalars=scalars, point_size=10, 
                     render_points_as_spheres=True, cmap="rainbow", 
                     clim=[0, 5], show_scalar_bar=False)

    plotter.show()


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
