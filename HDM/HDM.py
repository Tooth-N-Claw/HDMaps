import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix


from .spatial import (
    compute_base_spatial,
    compute_fiber_spatial,
    compute_joint_kernel,
    normalize_kernel
)
from .utils import compute_block_indices, HDMConfig
from .spectral import spectral_embedding
 


def hdm_embed(
    config: HDMConfig = HDMConfig(),
    data_samples: Optional[list[np.ndarray]] = None,
    block_indices: Optional[np.ndarray] = None,
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
) -> np.ndarray:
    """Compute HDM embedding with partial input handling."""

    if block_indices is None and data_samples is not None:
        block_indices = compute_block_indices(data_samples)       


    base_kernel = compute_base_spatial(config, data_samples, base_distances, base_kernel)
    print("Compute base kernel: Done.")

    fiber_kernel = compute_fiber_spatial(config, data_samples, fiber_distances, fiber_kernel)
    print("Compute fiber kernel: Done.")

       
    joint_kernel = compute_joint_kernel(base_kernel, fiber_kernel, block_indices)   

    print("Compute joint kernel: Done.")
    
    normalized_kernel, inv_sqrt_diag = normalize_kernel(joint_kernel)


    print("Normalize kernel: Done.")

    diffusion_coordinates = spectral_embedding(config, normalized_kernel, inv_sqrt_diag)

    print("Spectral embedding: Done.")

    return diffusion_coordinates
