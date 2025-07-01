import jax
import jax.numpy as jnp
from typing import NamedTuple

from HDM.utils.containers import HDMConfig, HDMData, JaxCoo
from HDM.utils.spatial import compute_dist_mat


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
    
    base_kernel = hdm_data.base_distances.with_values(
        jnp.exp(- hdm_data.base_distances.values**2 / hdm_config.base_epsilon)
    )
    
    hdm_data = hdm_data.with_base_kernel(base_kernel)

    return hdm_data


def compute_joint_kernel():
    pass


def cumulative_indices(data_samples: list) -> np.ndarray:
    """Calculate cumulative indices for data samples."""
    return np.insert(
        np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 
        0, 0
    )


def run_hdm(hdm_config, hdm_data):

    if hdm_data.base_kernel is None:
        hdm_data = compute_base_distances(hdm_config, hdm_data)
        hdm_data = compute_base_kernel(hdm_config, hdm_data)

    if hdm_data.fiber_kernel is None:
        hdm_data = compute_fiber_distances(hdm_config, hdm_data)
        hdm_data = compute_fiber_kernel(hdm_config, hdm_data)
