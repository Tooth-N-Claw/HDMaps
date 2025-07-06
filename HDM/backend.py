import jax
import jax.numpy as jnp
import scipy.sparse as sparse
import numpy as np
from typing import NamedTuple

from .utils.containers import HDMConfig, HDMData, JaxCoo
from .utils.spatial import compute_distances


def eigendecomposition(matrix, num_eigenvectors):
    """Perform eigendecomposition on a sparse matrix."""
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, 
        k=num_eigenvectors, 
        which="LM", 
        maxiter=5000,
        tol=1e-10
    )
    
    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    return eigvals, eigvecs


def compute_fiber_distances(hdm_config, hdm_data):
    pass


def compute_fiber_kernel(hdm_config, hdm_data):
    if hdm_data.fiber_distances is None:
        hdm_data = compute_fiber_distances(hdm_config, hdm_data)
    

    fiber_kernel = hdm_data.fiber_distances.with_values(
        jnp.exp(- hdm_data.fiber_distances.values**2 / hdm_config.fiber_epsilon)
    )

    hdm_data = hdm_data.with_fiber_kernel(fiber_kernel)

    return hdm_data


def compute_base_distances(hdm_config, hdm_data):
    
    base_distances = None
    
    return hdm_data


def compute_base_kernel(hdm_config, hdm_data):
    if hdm_data.base_distances is None:
        hdm_data = compute_base_distances(hdm_config, hdm_data)
    
    base_kernel = hdm_data.base_distances.with_data(
        jnp.exp(- hdm_data.base_distances.data**2 / hdm_config.base_epsilon)
    )
    
    hdm_data = hdm_data.with_base_kernel(base_kernel)

    return hdm_data



def compute_joint_kernel(hdm_config, hdm_data):   
    row_block_idx = jnp.searchsorted(
        hdm_data.cumulative_block_indices,
        hdm_data.fiber_kernel.row,
        side='right'
    ) - 1
    col_block_idx = jnp.searchsorted(
        hdm_data.cumulative_block_indices,
        hdm_data.fiber_kernel.col,
        side='right'
    ) - 1

    ncols = hdm_data.base_kernel.shape[1]
    fiber_keys = row_block_idx * ncols + col_block_idx
    block_keys = hdm_data.base_kernel.row * ncols + hdm_data.base_kernel.col

    sort_idx = jnp.argsort(block_keys)
    sorted_block_keys = block_keys[sort_idx]
    sorted_block_data = hdm_data.base_kernel.data[sort_idx]

    idx = jnp.searchsorted(sorted_block_keys, fiber_keys)

    # Handle out-of-bounds safely
    in_bounds = idx < sorted_block_keys.shape[0]
    matches = in_bounds & (sorted_block_keys[idx] == fiber_keys)

    scalars = jnp.where(matches, sorted_block_data[idx], 0.0)
    joint_kernel_data = hdm_data.fiber_kernel.data * scalars

    joint_kernel = hdm_data.fiber_kernel.with_data(joint_kernel_data).purge_zeros()

    return joint_kernel



def normalize_kernel(diffusion_matrix):
    """Compute the horizontal diffusion Laplacian."""
    # Compute row sums and check for zeros
    row_sums = np.sum(diffusion_matrix, axis=1).A1
    # if np.any(row_sums == 0):
    #     print("Warning: Zero row sums detected in diffusion matrix")
    #     row_sums[row_sums == 0] = 1e-10
    
    # Create diagonal matrix of inverse sqrt of row sums
    sqrt_diag = sparse.diags(1.0 / np.sqrt(row_sums), 0)
    
    # Compute normalized Laplacian

    # horizontal_diffusion_laplacian = sparse.eye(diffusion_matrix.shape[0]) - sqrt_diag @ diffusion_matrix @ sqrt_diag
    normalized_kernel = sqrt_diag @ diffusion_matrix @ sqrt_diag
    
    # Ensure symmetry
    # normalized_kernel = symmetrize(normalized_kernel)
    # print((horizontal_diffusion_laplacian.data < 0).any())
    
    return normalized_kernel, sqrt_diag


def run_hdm(hdm_config, hdm_data):

    if hdm_data.base_kernel is None:
        hdm_data = compute_base_distances(hdm_config, hdm_data)
        hdm_data = compute_base_kernel(hdm_config, hdm_data)

    if hdm_data.fiber_kernel is None:
        hdm_data = compute_fiber_distances(hdm_config, hdm_data)
        hdm_data = compute_fiber_kernel(hdm_config, hdm_data)

    print("separate kernels")

    joint_kernel = compute_joint_kernel(hdm_config, hdm_data)

    print("joint kernel")
    
    normalized_kernel, sqrt_diag = normalize_kernel(joint_kernel.toscipy())
    
    eigvals, eigvecs = eigendecomposition(normalized_kernel, hdm_config.num_eigenvectors)
    
    print("decomp")

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    # eigvals = np.maximum(eigvals, 1e-12)  # Clip negative values to near-zero
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
