import jax.numpy as jnp
import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix
from jax.experimental.sparse import BCOO
import jax


from .kernels import (
    compute_base_kernel,
    compute_fiber_kernel,
    compute_joint_kernel,
    normalize_kernel
)
from .spectral import spectral_embedding
from .validate import validate_config, validate_data
from .utils import compute_block_indices, HDMConfig
 


def hdm_embed(
    data_samples: list[np.ndarray],
    config: HDMConfig = HDMConfig(),
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
) -> jnp.ndarray:
    """"""    
    validate_config(config)
    validate_data(data_samples)
    print("Validated user input.")

    base_kernel = compute_base_kernel(
        config,
        data_samples,
        base_distances,
        base_kernel
    )
    print("Computed base kernel.")

    fiber_kernel = compute_fiber_kernel(
        config,
        data_samples,
        fiber_distances,
        fiber_kernel
    )
    print("Computed fiber base kernel.")

    block_indices = compute_block_indices(data_samples)

    joint_kernel = compute_joint_kernel(base_kernel, fiber_kernel, block_indices)   
    
    normalized_kernel, inv_sqrt_diag = normalize_kernel(joint_kernel)

    diffusion_coordinates = spectral_embedding(config, normalized_kernel, inv_sqrt_diag)

    return diffusion_coordinates
