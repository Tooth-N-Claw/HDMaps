import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sparse

from .utils import HDMConfig


def eigendecomposition(
    config,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    device = config.device
    tol = 1e-10
    maxiter = 10000
    k = config.num_eigenvectors
    which = "LM"

    if device == "cpu":
        eigvals, eigvecs = sparse.linalg.eigsh(
            matrix, 
            k=k, 
            which=which, 
            maxiter=maxiter, 
            tol=tol
        )
    elif device == "cuda":
        with cupy.cuda.Device(int(device.split(":")[-1])):
            cupy_matrix = cupyx.scipy.sparse.csr_matrix(matrix)
            eigvals, eigvecs = cupyx.scipy.sparse.linalg.eigsh(
                cupy_matrix,
                k=k,
                which=which,
                maxiter=maxiter,
                tol=tol
            )
            eigvals, eigvecs = cupy.asnumpy(eigvals), cupy.asnumpy(eigvecs)
    else:
        raise ValueError(f"Unsupported device: {device}")    

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

    eigvals, eigvecs = eigendecomposition(config, kernel)
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
