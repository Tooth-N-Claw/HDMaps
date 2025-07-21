import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from scipy.io import loadmat

from HDM import hdm_embed, HDMConfig, compute_fiber_kernel_from_maps


maps = loadmat("platyrrhine/softMapMatrix.mat")["softMapMatrix"]
fiber_kernel = compute_fiber_kernel_from_maps(maps)
base_distances = load_npz("example-data/teeth/base_distances.npz")
block_indices = np.load("example-data/teeth/block_indices.npy")


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

