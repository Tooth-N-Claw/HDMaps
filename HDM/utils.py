import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

from jax.experimental.sparse import BCOO
from scipy.sparse import csr_matrix


class HDMConfig(NamedTuple):
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    device: str | None = "CPU"
    base_metric: str = "frobenius"
    fiber_metric: str = "euclidean"
    base_sparsity: float = 2
    fiber_sparsity: float = 2


def ensure_sparse(matrix):
    """Converts a sparse scipy matrix to a jax BCOO"""
    return BCOO.from_scipy_sparse(matrix) if matrix is not None else None


def threshold_sparsify(matrix, threshold):
    mask = matrix.data <= threshold
    new_data = matrix.data[mask]
    new_indices = matrix.indices[mask, :]
    new_matrix = BCOO((new_data, new_indices), shape = matrix.shape)
    return new_matrix

    
def compute_block_indices(data_samples: list) -> jnp.ndarray:
    """Compute cumulative start indices for a list of data samples."""
    lengths = jnp.array([len(s) for s in data_samples], dtype=jnp.int32)
    return jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(lengths)])


def bcoo_to_csr(matrix: BCOO) -> csr_matrix:
    data = np.array(matrix.data)
    row = np.array(matrix.indices[:, 0])
    col = np.array(matrix.indices[:, 1])

    matrix = csr_matrix((data, (row, col)), shape=matrix.shape)

    return matrix
