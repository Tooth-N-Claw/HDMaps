import os
import numpy as np
import pyvista as pv
from HDM import hdm_embed, HDMConfig
from scipy.sparse import csr_matrix, block_array
from scipy.sparse import eye as speye
from joblib import Parallel, delayed
directory_path = "example-data/wing"
files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
print(len(files))
files = files[:1000]  # Limit to 5 samples for initial testing
data_samples = [
    np.loadtxt(os.path.join(directory_path, file), delimiter=",") for file in files
]

n = len(files)
maps = np.empty((n, n), dtype=object)

def create_identity(i, j, size):
    return (i, j, speye(size, format='csr'))

results = Parallel(n_jobs=-1)(
    delayed(create_identity)(i, j, data_samples[0].shape[0])
    for i in range(n) for j in range(n)
)

for i, j, mat in results:
    maps[i, j] = mat
        

if __name__ == '__main__':
    config = HDMConfig(
        base_epsilon=0.03,
        fiber_epsilon=0.002,
        base_sparsity=1,
        base_knn = None,
        fiber_sparsity=1,
        fiber_knn=None,
        device="cpu"
    )

    diffusion_coords = hdm_embed(data_samples=data_samples, config=config, fiber_kernel=None, maps=maps)

    # embed_vs_actual(diffusion_coords, data_samples, num_samples=4)


    # points = pv.PolyData(diffusion_coords)
    # plotter = pv.Plotter()
    # plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
    # plotter.show()

    # point_cloud = pv.PolyData(diffusion_coords[:, :3])
    # plotter = pv.Plotter()

    # scalars = np.tile(np.arange(sample_length), num_samples)
    # cmap = plt.get_cmap("rainbow", sample_length)
    # norm = Normalize(vmin=0, vmax=sample_length-1)

    # plotter.add_mesh(point_cloud, scalars=scalars, point_size=10,
    #                  render_points_as_spheres=True, cmap="tab20",
    #                  clim=[0, sample_length-1], show_scalar_bar=False)

    # plotter.show()
