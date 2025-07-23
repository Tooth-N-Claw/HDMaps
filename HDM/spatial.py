from sklearn.neighbors import NearestNeighbors
from scipy.sparse import bmat, csr_matrix, coo_matrix
import numpy as np


from .utils import HDMConfig


def compute_fiber_kernel_from_maps(maps):
    num_rows, num_cols = maps.shape

    blocks = [
        [csr_matrix(maps[i, j]) for j in range(num_cols)]
        for i in range(num_rows)
    ]

    fiber_kernel = bmat(blocks, format='csr').tocoo()

    return fiber_kernel


def compute_base_distances(config: HDMConfig, data_samples: list[np.ndarray]) -> csr_matrix:
    print("Assumes all data samples has same shape")
    data = np.array([sample.flatten() for sample in data_samples])

    
    if config.base_knn != None:
        nn = NearestNeighbors(n_neighbors=config.base_knn, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nn.fit(data)
        sparse_dist_matrix = nn.kneighbors_graph(data, mode='distance')
    elif config.base_sparsity != None:
        nn = NearestNeighbors(radius=config.base_sparsity, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nn.fit(data)
        sparse_dist_matrix = nn.radius_neighbors_graph(data, mode='distance')
        
    
    return sparse_dist_matrix


def compute_fiber_distances(config: HDMConfig, data_samples: list[np.ndarray]) -> csr_matrix:
    data = np.vstack(data_samples)

    if config.fiber_knn != None:
        nn = NearestNeighbors(n_neighbors=config.fiber_knn, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nn.fit(data)
        sparse_dist_matrix = nn.kneighbors_graph(data, mode='distance')
    elif config.fiber_sparsity != None:
        nn = NearestNeighbors(radius=config.fiber_sparsity, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nn.fit(data)
        sparse_dist_matrix = nn.radius_neighbors_graph(data, mode='distance')

    return sparse_dist_matrix
   

def compute_kernel(distances, eps):
    kernel = distances.copy()
    kernel.data = np.exp(- kernel.data ** 2 / eps)
    kernel.setdiag(1.0)

    return kernel


def compute_base_spatial(config: HDMConfig, data_samples, base_distances, base_kernel) -> csr_matrix:
    """"""

    if base_distances is None and base_kernel is None:
        base_distances = compute_base_distances(config, data_samples)
    elif base_distances is not None and base_kernel is None:
        base_distances.data[base_distances.data >= config.base_sparsity] = 0
        base_distances.eliminate_zeros()

    if base_kernel is None:
        base_kernel = compute_kernel(base_distances, config.base_epsilon)

    return base_kernel


def compute_fiber_spatial(config: HDMConfig, data_samples, fiber_distances, fiber_kernel)-> coo_matrix:
    """"""

    if fiber_distances is None and fiber_kernel is None:
        fiber_distances = compute_fiber_distances(config, data_samples)
    elif fiber_distances is not None and fiber_kernel is None:
        fiber_distances.data[fiber_distances.data >= config.fiber_sparsity] = 0
        fiber_distances.eliminate_zeros()

    if fiber_kernel is None:
        fiber_kernel = compute_kernel(fiber_distances, config.fiber_epsilon)

    return fiber_kernel.tocoo()


def compute_joint_kernel(
    base_kernel: csr_matrix,
    fiber_kernel: coo_matrix,
    block_indices: np.ndarray
) -> coo_matrix:
    fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side='right') - 1
    fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side='right') - 1

    block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)

    joint_data = fiber_kernel.data * block_vals
    joint_kernel = coo_matrix((joint_data, (fiber_kernel.row, fiber_kernel.col)), shape=fiber_kernel.shape)

    joint_kernel.eliminate_zeros()

    return joint_kernel


def symmetrize(mat):
    return (mat + mat.T) / 2


def normalize_kernel(diffusion_matrix: coo_matrix) -> csr_matrix:
    row_sums = np.array(diffusion_matrix.sum(axis = 1)).flatten()
    inv_sqrt_diag = 1 / np.sqrt(row_sums)

    new_data = diffusion_matrix.data * inv_sqrt_diag[diffusion_matrix.row] * inv_sqrt_diag[diffusion_matrix.col]
    
    normalized_kernel = csr_matrix((new_data, (diffusion_matrix.row, diffusion_matrix.col)), shape=diffusion_matrix.shape)    

    normalized_kernel = symmetrize(normalized_kernel)
    
    return normalized_kernel, inv_sqrt_diag
