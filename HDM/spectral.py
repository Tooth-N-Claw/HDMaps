import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sparse

from .utils import HDMConfig


def eigendecomposition(
    matrix: sparse.csr_matrix,
    num_eigenvectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, 
        k=num_eigenvectors, 
        which="LM", 
        maxiter=10000, 
        tol=1e-10
    )

    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    return eigvals, eigvecs


def spectral_embedding(
    config: HDMConfig,
    kernel: csr_matrix,
    inv_sqrt_diag: np.ndarray,
) -> np.ndarray:
    sqrt_diag = sparse.diags(inv_sqrt_diag, 0)

    eigvals, eigvecs = eigendecomposition(kernel, config.num_eigenvectors)
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
