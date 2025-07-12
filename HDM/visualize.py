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
