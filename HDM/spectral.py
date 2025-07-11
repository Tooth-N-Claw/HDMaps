import numpy as np
import jax.numpy as jnp
import scipy.sparse as sparse

from jax.experimental.sparse import BCOO

from .utils import HDMConfig, bcoo_to_csr


def eigendecomposition(
    matrix: sparse.csr_matrix,
    num_eigenvectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, 
        k=num_eigenvectors, 
        which="LM", 
        maxiter=1000, 
        tol=1e-10
    )
    
    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    return eigvals, eigvecs


def spectral_embedding(
    config: HDMConfig,
    kernel: BCOO,
    inv_sqrt_diag: jnp.ndarray,
) -> np.ndarray:
    sqrt_diag = sparse.diags(np.array(inv_sqrt_diag), 0)

    kernel = bcoo_to_csr(kernel)
    
    eigvals, eigvecs = eigendecomposition(kernel, config.num_eigenvectors)
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    print("Computed spectral embedding.")

    return bundle_HDM_full
