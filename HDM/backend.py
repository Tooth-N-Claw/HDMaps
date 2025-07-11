import jax
from functools import partial
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




def compute_base_distances(hdm_config, hdm_data):
    
    base_distances = None
    
    return hdm_data




def run_hdm(hdm_config, kernel):

    print("separate kernels")

    joint_kernel = compute_joint_kernel(hdm_config, hdm_data)

    print("joint kernel")
    
    normalized_kernel, sqrt_diag = normalize_kernel(joint_kernel.toscipy())
    
    eigvals, eigvecs = eigendecomposition(normalized_kernel, hdm_config.num_eigenvectors)
    
    print("decomp")

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
