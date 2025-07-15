import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from HDM import hdm_embed, HDMConfig


fiber_kernel = load_npz("fiber_kernel.npz")
base_distances = load_npz("base_distances.npz")
block_indices = np.load("block_indices.npy")


config = HDMConfig(
    base_sparsity = 0.4,
)

points = hdm_embed(
    config = config,
    block_indices = block_indices,
    fiber_kernel = fiber_kernel,
    base_distances = base_distances
)

points = pv.PolyData(points)
plotter = pv.Plotter()   
plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
plotter.show()

