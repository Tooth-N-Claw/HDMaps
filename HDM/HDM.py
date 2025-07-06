import jax.numpy as jnp
import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix

from .backend import run_hdm
from .utils.containers import HDMConfig, HDMData, JaxCoo, jax_coo
    

def cumulative_indices(data_samples: list) -> np.ndarray:
    """Calculate cumulative indices for data samples."""
    return np.insert(
        np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 
        0, 0
    )


METRICS = {"frobenius", "euclidean"}


def HDM(
    data_samples: list[np.ndarray],
    base_epsilon: float = 0.04,
    fiber_epsilon: float = 0.08,
    num_eigenvectors: int = 4,
    device: str | None = "CPU",
    base_metric: str = "frobenius",
    fiber_metric: str = "euclidean",
    base_sparsity: float = 0.08,
    fiber_sparsity: float = 0.08,
    base_kernel: Optional[jnp.ndarray] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[jnp.ndarray] = None,
    fiber_distances: Optional[coo_matrix] = None,
    ):
        
    if base_kernel is not None:
        base_kernel = jax_coo(base_kernel)
    else:
        print("Base sparsity parameter is ignored due to precomputed base_kernel")

    if fiber_kernel is not None:
        fiber_kernel = jax_coo(fiber_kernel)
    else:
        print("Fiber sparsity parameter is ignored due to precomputed fiber_kernel")
        
    if base_distances is not None:
        base_distances = jax_coo(base_distances)

    if fiber_distances is not None:
        fiber_distances = jax_coo(fiber_distances)

    if not base_metric in METRICS:
        raise f"Error: {base_metric} is not a valid base metric."

    if not fiber_metric in METRICS:
        raise f"Error: {fiber_metric} is not a valid fiber metric."

    hdm_config = HDMConfig(
        base_epsilon,
        fiber_epsilon,
        num_eigenvectors,
        device,
    )

    cumulative_block_indices = cumulative_indices(data_samples)

    hdm_data = HDMData(
        data_samples = data_samples,
        base_distances = base_distances,
        fiber_kernel = fiber_kernel,
        cumulative_block_indices = cumulative_block_indices
    )
    
    diffusion_coords = run_hdm(hdm_config, hdm_data)

    return diffusion_coords
