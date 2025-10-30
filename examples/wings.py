import os
import numpy as np
from HDM import hdm_embed, HDMConfig
from scipy.sparse import eye as speye
from joblib import Parallel, delayed
import pyvista as pv

directory_path = "example-data/wing"
files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]

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
        fiber_sparsity=1,
        device="cpu"
    )

    diffusion_coords = hdm_embed(data_samples=data_samples, config=config, fiber_kernel=None, maps=maps)

    # embed_vs_actual(diffusion_coords, data_samples, num_samples=4)


    points = pv.PolyData(diffusion_coords)
    plotter = pv.Plotter()
    plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
    plotter.show()
