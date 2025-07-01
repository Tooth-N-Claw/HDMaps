import jax.numpy as jnp
import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix

from backend import run_hdm
from HDM.utils.containers import HDMConfig, HDMData, JaxCoo

    
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
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
    ):
        
    cumulative_block_indices = cumulative_indices(data_samples)

    if base_kernel is not None:
        base_kernel = JaxCoo.from_scipy(base_kernel)

    if fiber_kernel is not None:
        fiber_kernel = JaxCoo.from_scipy(fiber_kernel)
        
    if base_distances is not None:
        base_distances = JaxCoo.from_scipy(base_distances)

    if fiber_distances not None:
        fiber_distances = JaxCoo.from_scipy(fiber_distances)

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

    hdm_data = HDMData(
        data_samples = data_samples,
        base_distances = base_dist,
        cumulative_block_indices = cumulative_block_indices
    )
    
    diffusion_coords = run_hdm(backend, hdm_config, hdm_data)
