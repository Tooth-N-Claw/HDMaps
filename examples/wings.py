import os
import numpy as np
import pyvista as pv
from HDM import hdm_embed, HDMConfig

directory_path = "example-data/wing"
files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
files = files[:120]
data_samples = [
    np.loadtxt(os.path.join(directory_path, file), delimiter=",") for file in files
]


config = HDMConfig(
    base_epsilon=0.004,
    fiber_epsilon=0.0006,
    base_sparsity=0.08,
    base_knn = None,
    fiber_sparsity=0.08,
    fiber_knn = None,
    device="gpu"
)

diffusion_coords = hdm_embed(data_samples=data_samples, config=config, fiber_kernel=None)

# embed_vs_actual(diffusion_coords, data_samples, num_samples=4)


points = pv.PolyData(diffusion_coords)
plotter = pv.Plotter()
plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
plotter.show()

# point_cloud = pv.PolyData(diffusion_coords[:, :3])
# plotter = pv.Plotter()

# scalars = np.tile(np.arange(sample_length), num_samples)
# cmap = plt.get_cmap("rainbow", sample_length)
# norm = Normalize(vmin=0, vmax=sample_length-1)

# plotter.add_mesh(point_cloud, scalars=scalars, point_size=10,
#                  render_points_as_spheres=True, cmap="tab20",
#                  clim=[0, sample_length-1], show_scalar_bar=False)

# plotter.show()
