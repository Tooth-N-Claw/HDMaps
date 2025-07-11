from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import jax
import numpy as np

from jax.experimental.sparse import bcoo_reduce_sum
from .distances import compute_base_distances, compute_fiber_distances
from .utils import HDMConfig


def compute_kernel(distances: BCOO, eps: float) -> BCOO:
    kernel_data = jnp.exp(-distances.data ** 2 / eps)   
    
    n = distances.shape[0]
    
    # Add 1's in the diagonal
    diag_indices = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)
    diag_data = jnp.ones(n)
    
    combined_indices = jnp.concatenate([diag_indices, distances.indices], axis=0)
    combined_data = jnp.concatenate([diag_data, kernel_data], axis=0)
    
    sort_keys = combined_indices[:, 0] * n + combined_indices[:, 1]
    sort_order = jnp.argsort(sort_keys)
    
    sorted_indices = combined_indices[sort_order]
    sorted_data = combined_data[sort_order]
    
    kernel = BCOO((sorted_data, sorted_indices), shape=distances.shape)

    return kernel


def ensure_sparse(matrix):
    """Converts a sparse scipy matrix to a jax BCOO"""
    return BCOO.from_scipy_sparse(matrix) if matrix is not None else None


def compute_base_kernel(
    config: HDMConfig,
    data_samples: list[np.ndarray],
    base_distances: BCOO,
    base_kernel: BCOO
):
    """"""
    base_distances = ensure_sparse(base_distances)
    base_kernel = ensure_sparse(base_kernel)

    if base_distances is None and base_kernel is None:
        base_distances = compute_base_distances(config, data_samples)

    if base_kernel is None:
        base_kernel = compute_kernel(base_distances, config.base_epsilon)

    return base_kernel


def compute_fiber_kernel(
    config: HDMConfig,
    data_samples: list[np.ndarray],
    fiber_distances: BCOO,
    fiber_kernel: BCOO
):
    """"""
    fiber_distances  = ensure_sparse(fiber_distances)
    fiber_kernel = ensure_sparse(fiber_kernel)

    if fiber_distances is None and fiber_kernel is None:
        fiber_distances = compute_base_distances(config, data_samples)

    if fiber_kernel is None:
        fiber_kernel = compute_kernel(fiber_distances, config.fiber_epsilon)

    return fiber_kernel


def compute_joint_kernel(
    base_kernel: BCOO,
    fiber_kernel: BCOO,
    block_indices: jnp.ndarray
):
    fiber_base_row = jnp.searchsorted(block_indices, fiber_kernel.indices[:, 0], side='right') - 1
    fiber_base_col = jnp.searchsorted(block_indices, fiber_kernel.indices[:, 1], side='right') - 1
    fiber_base_ids = fiber_base_row * base_kernel.shape[0] + fiber_base_col
      
    base_ids = base_kernel.indices[:, 0] * base_kernel.shape[0] + base_kernel.indices[:, 1]


    positions = jnp.searchsorted(base_ids, fiber_base_ids, side='left')

    valid = (positions < base_ids.size) & (base_ids[positions] == fiber_base_ids)


    matching_indices = positions[valid]

    new_data = fiber_kernel.data[valid] * base_kernel.data[matching_indices]
    new_indices = fiber_kernel.indices[valid, :]

    joint_kernel = BCOO((new_data, new_indices), shape = fiber_kernel.shape)

    return joint_kernel


def normalize_kernel(diffusion_matrix: BCOO):
    row_sums = diffusion_matrix.sum(axis = 1).todense()
    inv_sqrt_diag = 1 / jnp.sqrt(row_sums)  

    row = diffusion_matrix.indices[:, 0]
    col = diffusion_matrix.indices[:, 1]

    new_data = diffusion_matrix.data * inv_sqrt_diag[row] * inv_sqrt_diag[col]
    
    normalized_kernel = BCOO((new_data, diffusion_matrix.indices), shape=diffusion_matrix.shape)    
    
    return normalized_kernel, inv_sqrt_diag
