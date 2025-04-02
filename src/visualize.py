import numpy as np
import pyvista as pv


# def visualize(points: np.ndarray, data_samples_types: list[str] = None) -> None:
    
#     point_cloud = pv.PolyData(points)

#     plotter = pv.Plotter()
#     plotter.add_mesh(point_cloud, color="red", point_size=5, render_points_as_spheres=True)
#     plotter.show()


def visualize(points: np.ndarray, data_samples_types: list[str] = None) -> None:
    print(points.shape)
    print(len(data_samples_types))
    point_cloud = pv.PolyData(points)
    
    plotter = pv.Plotter()
    
    if data_samples_types is not None:
        if len(data_samples_types) != len(points):
            raise ValueError("data_samples_types must have the same length as points")
        
        unique_species = sorted(list(set(data_samples_types)))
        num_species = len(unique_species)
        
        species_to_index = {species: i for i, species in enumerate(unique_species)}
        indices = np.array([species_to_index[species] for species in data_samples_types])
        
        point_cloud["species"] = indices
        
        plotter.add_mesh(point_cloud, scalars="species", point_size=15, 
                         render_points_as_spheres=True, cmap="rainbow", 
                         clim=[0, num_species-1])
        
        plotter.add_scalar_bar(title="Species Index")
        
        legend = "Species Legend:\n" + "\n".join([f"{i}: {s}" for i, s in enumerate(unique_species)])
        plotter.add_text(legend, position=(0.02, 0.85), font_size=10, viewport=True)
    else:
        plotter.add_mesh(point_cloud, color="red", point_size=5, render_points_as_spheres=True)
    
    plotter.show()