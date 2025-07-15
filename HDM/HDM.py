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
