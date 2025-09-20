import os
from HDM import hdm_embed, HDMConfig, compute_fiber_kernel_from_maps
from scipy.io import loadmat
import numpy as np
from scipy.sparse import random 
from joblib import Parallel, delayed
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize

from HDM.visualization_tools import embed_vs_actual

def generate_samples(data_dim, data_points, num_samples, seed):
    np.random.seed(seed)  
    data = [np.random.uniform(-1, 1, size=(data_points, data_dim)) for _ in range(num_samples)]
    return data


def generate_fiber_bundle_data(data_dim, data_points, num_samples, seed, base_dim=2, fiber_dim=1):
    """Generate data on a fiber bundle for horizontal diffusion maps."""
    np.random.seed(seed)
    
    # Ensure base_dim + fiber_dim <= data_dim
    base_dim = min(base_dim, data_dim - 1)
    fiber_dim = min(fiber_dim, data_dim - base_dim)
    
    data = []
    
    for sample_idx in range(num_samples):
        sample_data = []
        
        for _ in range(data_points):
            # Generate point on base manifold (e.g., on a 2D surface)
            if base_dim == 2:
                # Example: points on a 2D torus or sphere embedded in higher dimensions
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, 2*np.pi)
                base_point = np.array([
                    (2 + 0.5*np.cos(phi)) * np.cos(theta),  # Torus parametrization
                    (2 + 0.5*np.cos(phi)) * np.sin(theta),
                ])
            else:
                # General case: points on base manifold
                base_point = np.random.uniform(-2, 2, base_dim)
            
            # Generate fiber coordinates (vertical directions)
            # These vary between datasets but are "vertical" to the base
            fiber_offset = 0.2 * sample_idx  # Different datasets sample different fiber positions
            fiber_point = fiber_offset + 0.1 * np.random.randn(fiber_dim)
            
            # Embed in higher dimensional space
            full_point = np.zeros(data_dim)
            full_point[:base_dim] = base_point
            full_point[base_dim:base_dim+fiber_dim] = fiber_point
            
            # Add some mixing to make it more realistic
            if data_dim > base_dim + fiber_dim:
                # Add small random components in remaining dimensions
                full_point[base_dim+fiber_dim:] = 0.05 * np.random.randn(data_dim - base_dim - fiber_dim)
            
            sample_data.append(full_point)
        
        data.append(np.array(sample_data))
    
    return data


def generate_single_sparse_map(num_datapoints, density, seed):
    rng = np.random.RandomState(seed)
    return random(num_datapoints, num_datapoints, density=density, format='coo', random_state=rng)


def generate_sparse_maps(num_samples, num_datapoints, density, seed, n_jobs=-1):
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31, size=num_samples * num_samples)
    
    sparse_matrices = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_sparse_map)(num_datapoints, density, s) for s in seeds
    )
    
    map_matrix = np.empty((num_samples, num_samples), dtype=object)
    for idx, sparse_map in enumerate(sparse_matrices):
        map_matrix[idx // num_samples, idx % num_samples] = sparse_map
    
    return map_matrix


def visualize_corresponding_points(points: np.ndarray, num_samples) -> None:  
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    scalars = np.tile(np.arange(6), num_samples)
    cmap = cm.get_cmap("rainbow", 6)
    norm = Normalize(vmin=0, vmax=6-1)
    
    plotter.add_mesh(point_cloud, scalars=scalars, point_size=10, 
                     render_points_as_spheres=True, cmap="rainbow", 
                     clim=[0, 5], show_scalar_bar=False)

    plotter.show()


# benchmark parameters
seed = 42
data_dim = 2
data_points = 100
num_samples = 10
density = 0.01

print(f"Generating {num_samples} samples with {data_points} points each in {data_dim}D space.")
data_samples = generate_samples(data_dim, data_points, num_samples, seed)
# data_samples = generate_fiber_bundle_data(data_dim, data_points, num_samples, seed, base_dim=2, fiber_dim=1)
print(f"Generating sparse maps with density {density} for {num_samples} samples.")
maps = generate_sparse_maps(num_samples, data_points, density, seed)
fiber_kernel = compute_fiber_kernel_from_maps(maps)
print("Data and maps generated successfully.")

# # print(data[:10])
# maps = loadmat("platyrrhine/softMapMatrix.mat")["softMapMatrix"]
# fiber_kernel = compute_fiber_kernel_from_maps(maps)
# base_distances = load_npz("example-data/teeth/base_distances.npz")
# block_indices = np.load("example-data/teeth/block_indices.npy")
directory_path = 'examples/example-data/wing/'
files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
files = files[:50]
data_samples = [np.loadtxt(os.path.join(directory_path, file), delimiter=',') for file in files]



# config = HDMConfig(
#     base_epsilon = 0.004,
#     fiber_epsilon = 0.0006,
#     base_sparsity = 0.08,
#     base_knn = None,
#     fiber_sparsity = 0.08,
#     fiber_knn = None,
#     device="gpu",

    
# )

config = HDMConfig(
    base_epsilon = 0.004,
    fiber_epsilon = 0.0006,
    base_sparsity = 0.08,
    base_knn = None,
    fiber_sparsity = 0.08,
    fiber_knn = None,

)


points = hdm_embed(
    data_samples=data_samples,
    config=config
    # # fiber_kernel=fiber_kernel
)


# embed_vs_actual(points, data_samples, num_samples=4)


# points = pv.PolyData(points)
# plotter = pv.Plotter()   
# plotter.add_mesh(points, color="red", point_size=5, render_points_as_spheres=True)
# plotter.show()




# plotter = pv.Plotter()


# for points_3d in points:
#     # points_3d = np.column_stack([sample, np.zeros(len(sample))])
#     plotter.add_mesh(pv.PolyData(points_3d), point_size=5, render_points_as_spheres=True)

# plotter.show()


