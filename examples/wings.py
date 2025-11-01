import os
import numpy as np
import matplotlib.pyplot as plt
from HDM import hdm_embed, HDMConfig
from scipy.sparse import eye as speye
from joblib import Parallel, delayed
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import pyvista as pv

from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import numpy as np


directory_path = "wing-data"
files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
files = files[:500]

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
        base_epsilon=0.003,
        fiber_epsilon=0.0001,
        base_sparsity=1,
        fiber_sparsity=1,
        device="cpu"
    )

    hdm_coords, hbdm_coords, D = hdm_embed(data_samples=data_samples, config=config, fiber_kernel=None, maps=maps)

    # 3D MDS
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    X_3d = mds.fit_transform(D)  # shape (n_points, 3)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c='blue', s=50, alpha=0.7)
    ax.set_xlabel('MDS 1')
    ax.set_ylabel('MDS 2')
    ax.set_zlabel('MDS 3')
    ax.set_title('3D MDS of Horizontal Base Diffusion Map')

    plt.show()

    # Labels and title
    # ax.set_title("3D MDS embedding of HBDM coordinates", fontsize=14)
    # ax.set_xlabel("MDS Dimension 1")
    # ax.set_ylabel("MDS Dimension 2")
    # ax.set_zlabel("MDS Dimension 3")

    # # Optional: grid and viewing angle
    # ax.grid(True, linestyle='--', alpha=0.5)
    # ax.view_init(elev=30, azim=45)  # adjust elevation and azimuth if needed

    # plt.show()    
    # embed_vs_actual(diffusion_coords, data_samples, num_samples=4)


    points = pv.PolyData(hdm_coords)
    plotter = pv.Plotter()
    plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
    plotter.show()
