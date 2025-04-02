import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize

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
    
    plotter.show()
