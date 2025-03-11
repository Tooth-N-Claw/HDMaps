import numpy as np
import pyvista as pv


def visualize(points: np.ndarray):
    point_cloud = pv.PolyData(points)

    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="red", point_size=5, render_points_as_spheres=True)
    plotter.show()
