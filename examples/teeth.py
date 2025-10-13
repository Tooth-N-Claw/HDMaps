import numpy as np
import pyvista as pv

from scipy.sparse import load_npz, block_array, bsr_matrix
from scipy.io import loadmat

from HDM import (
    hdm_embed,
    HDMConfig,
    compute_fiber_kernel_from_maps,
)


from pathlib import Path


def load_off_vertices(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    n_vertices, _, _ = map(int, lines[1].split())

    return np.array(
        [list(map(float, lines[i].split())) for i in range(2, 2 + n_vertices)],
        dtype=float,
    )


def load_all_off_vertices(folder):
    folder = Path(folder)
    arrays = [load_off_vertices(p) for p in sorted(folder.glob("*.off"))]
    return arrays


data_samples = load_all_off_vertices("platyrrhine/ReparametrizedOFF")

# points = pv.PolyData(data_samples[0])
# plotter = pv.Plotter()
# plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
# plotter.show()


maps = loadmat("platyrrhine/softMapMatrix.mat")["softMapMatrix"]
# fiber_kernel = compute_fiber_kernel_from_maps(maps)
base_distances = load_npz(
    "example-data/teeth/base_distances.npz"
)  # .tocoo().eliminate_zeros()
block_indices = np.load("example-data/teeth/block_indices.npy")


# config = HDMConfig(base_sparsity=0.4, base_knn=4, device="cpu")
# config = HDMConfig(
#     base_epsilon = 0.04,
#     fiber_epsilon = 0.08,
#     device="cpu")

config = HDMConfig(
    base_epsilon = 0.04,
    fiber_epsilon = 0.08,
    base_knn=4,
    fiber_knn=4,
    # base_sparsity=0.1,
    # fiber_sparsity=0.1,
    device="gpu")
# data_samples = [np.random.uniform(-1, 1, size=(4463, 3)) for _ in range(50)]

# base_distances.data[base_distances.data >= config.base_sparsity] = 0
# base_distances.eliminate_zeros()
# fiber_kernel.eliminate_zeros()
# maps = block_array(maps, format="csr")

# maps = np.array([[bsr_matrix(maps[i,j]) for i in range(maps.shape[0])] for j in range(maps.shape[1])])

print("start")
points = hdm_embed(
    config=config,
    block_indices = block_indices,
    maps=maps,
    base_distances = base_distances,
    data_samples = data_samples
)


points = pv.PolyData(points)
plotter = pv.Plotter()
plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
plotter.show()
