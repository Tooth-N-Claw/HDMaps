import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

from .utils import HDMConfig, compute_block_indices, get_backend
from itertools import permutations


def hdm_embed(
    config: HDMConfig = HDMConfig(),
    data_samples: Optional[list[np.ndarray]] = None,
    block_indices: Optional[np.ndarray] = None,
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
    maps = None,
) -> np.ndarray:
    """
    Compute the Horizontal Diffusion Maps (HDM) embedding from input data.

    This function constructs and processes base and fiber kernels from the input data or
    precomputed distances/kernels, normalizes the resulting joint kernel, and computes
    a HDM embedding.

    Parameters:
        config (HDMConfig): Configuration object specifying HDM parameters.
        data_samples (list[np.ndarray], optional): List of data arrays (e.g., sampled fibers).
        block_indices (np.ndarray, optional): Block indices specifying data partitioning.
        base_kernel (coo_matrix, optional): Precomputed base kernel (spatial proximity).
        fiber_kernel (coo_matrix, optional): Precomputed fiber kernel (fiber similarity).
        base_distances (coo_matrix, optional): Precomputed base distances.
        fiber_distances (coo_matrix, optional): Precomputed fiber distances.

    Returns:
        np.ndarray: Diffusion coordinates from the joint HDM embedding.
    """

    base_kernel = compute_base_spatial(
        config, data_samples, base_distances, base_kernel
    )
    print("Compute base kernel: Done.")

    fiber_kernels = compute_fiber_spatial(
        config, data_samples, fiber_distances, fiber_kernel
    )
    print("Compute fiber kernel: Done.")

    if block_indices is None and data_samples is not None:
        block_indices = compute_block_indices(data_samples)

    ## From here on we can use gpu to speed up the computation, if the user wants to
    backend = get_backend(config)
    
    
    # joint_kernel = backend.compute_joint_kernel(
    #     base_kernel, fiber_kernels, block_indices, maps
    # )

    # print("Compute joint kernel: Done.")

    # normalized_kernel, inv_sqrt_diag = backend.normalize_kernel(joint_kernel)

    # print("Normalize kernel: Done.")



    normalized_kernel, inv_sqrt_diag = backend.compute_joint_kernel_linear_operator(
        base_kernel, fiber_kernels, block_indices, maps
    )
    print("Construct Linear Operator: Done.")
    
    diffusion_coordinates = backend.spectral_embedding(
        config, normalized_kernel, inv_sqrt_diag
    )
    
    print("Spectral embedding: Done.")

    return diffusion_coordinates


def compute_base_distances(
    config: HDMConfig, data_samples: list[np.ndarray]
) -> csr_matrix:
    print("Assumes all data samples has same shape")
    data = np.array([sample.flatten() for sample in data_samples])
    # print(data.shape)
    n = len(data_samples)
    dist = np.zeros((n, n))

    # for i, j in permutations(range(50), 2):
    #     dist[i, j] = np.linalg.norm(data_samples[i] - data_samples[j])
    # return csr_matrix(dist)

    if config.base_knn != None:
        nn = NearestNeighbors(
            n_neighbors=config.base_knn, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        nn.fit(data)
        sparse_dist_matrix = nn.kneighbors_graph(data, mode="distance")
    elif config.base_sparsity != None:
        nn = NearestNeighbors(
            radius=config.base_sparsity, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        nn.fit(data)
        sparse_dist_matrix = nn.radius_neighbors_graph(data, mode="distance")

    return sparse_dist_matrix


# def compute_fiber_distances(
#     config: HDMConfig, data_samples: list[np.ndarray]
# ) -> csr_matrix:
#     data = np.vstack(data_samples)

#     if config.fiber_knn != None:
#         nn = NearestNeighbors(
#             n_neighbors=config.fiber_knn,
#             algorithm="auto",
#             metric="euclidean",
#             n_jobs=-1,
#         )
#         nn.fit(data)
#         sparse_dist_matrix = nn.kneighbors_graph(data, mode="distance")
#     elif config.fiber_sparsity != None:
#         nn = NearestNeighbors(
#             radius=config.fiber_sparsity,
#             algorithm="auto",
#             metric="euclidean",
#             n_jobs=-1,
#         )
#         nn.fit(data)
#         sparse_dist_matrix = nn.radius_neighbors_graph(data, mode="distance")

#     return sparse_dist_matrix

def compute_fiber_distances(
    config: HDMConfig, data_samples: list[np.ndarray]
) -> csr_matrix:
    # data = np.vstack(data_samples)
    fiber_distances = []
    if config.fiber_knn != None:
        for data in data_samples:
            nn = NearestNeighbors(
                n_neighbors=config.fiber_knn,
                algorithm="auto",
                metric="euclidean",
                n_jobs=-1,
            )
            nn.fit(data)
            sparse_dist_matrix = nn.kneighbors_graph(data, mode="distance")
            fiber_distances.append(sparse_dist_matrix)
    elif config.fiber_sparsity != None:
        for data in data_samples:

            nn = NearestNeighbors(
                radius=config.fiber_sparsity,
                algorithm="auto",
                metric="euclidean",
                n_jobs=-1,
            )
            nn.fit(data)
            sparse_dist_matrix = nn.radius_neighbors_graph(data, mode="distance")
            fiber_distances.append(sparse_dist_matrix)

    return fiber_distances



def compute_kernel(distances, eps):
    kernel = distances.copy()
    kernel.data = np.exp(-(kernel.data**2) / eps)
    kernel.setdiag(1.0)

    return kernel


def compute_base_spatial(
    config: HDMConfig, data_samples, base_distances, base_kernel
) -> csr_matrix:
    """"""

    if base_distances is None and base_kernel is None:
        base_distances = compute_base_distances(config, data_samples)
    elif base_distances is not None and base_kernel is None:
        if config.base_sparsity != None:
            base_distances.data[base_distances.data >= config.base_sparsity] = 0
        base_distances.eliminate_zeros()


    if base_kernel is None:
        base_kernel = compute_kernel(base_distances, config.base_epsilon)

    return base_kernel


# def compute_fiber_spatial(
#     config: HDMConfig, data_samples, fiber_distances, fiber_kernel
# ) -> coo_matrix:
#     """"""
#     if fiber_distances is None and fiber_kernel is None:
#         fiber_distances = compute_fiber_distances(config, data_samples)
#     elif fiber_distances is not None and fiber_kernel is None:
#         fiber_distances.data[fiber_distances.data >= config.fiber_sparsity] = 0
#         fiber_distances.eliminate_zeros()
#     if fiber_kernel is None:
#         fiber_kernel = compute_kernel(fiber_distances, config.fiber_epsilon)
#     return fiber_kernel.tocoo()


def compute_fiber_spatial(
    config: HDMConfig, data_samples, fiber_distances, fiber_kernels
) -> coo_matrix:
    """"""
    if fiber_kernels is None:
        if fiber_distances is None:
            fiber_distances = compute_fiber_distances(config, data_samples)
        else:
            fiber_distances.data[fiber_distances.data >= config.fiber_sparsity] = 0
            fiber_distances.eliminate_zeros()
        fiber_kernels = []
        for fiber in fiber_distances:
            fiber_kernel = compute_kernel(fiber, config.fiber_epsilon)
            fiber_kernels.append(fiber_kernel)
    return fiber_kernels
