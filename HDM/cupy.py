from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np

import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csr_matrix
from cupyx.scipy import sparse

from .utils import HDMConfig


def compute_joint_kernel(
    base_kernel: scipy.sparse.csr_matrix,
    fiber_kernel: scipy.sparse.coo_matrix,
    block_indices: np.ndarray
) -> coo_matrix:
    fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side='right') - 1
    fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side='right') - 1

    block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)
    
    joint_data = fiber_kernel.data * block_vals
    
    # joint_kernel = coo_matrix((joint_data, (fiber_kernel.row, fiber_kernel.col)), shape=fiber_kernel.shape)
    joint_kernel = coo_matrix(
    (joint_data, (fiber_kernel.row, fiber_kernel.col)), 
    shape=fiber_kernel.shape
)
    joint_kernel.eliminate_zeros()
    
    return joint_kernel


def symmetrize(mat):
    return (mat + mat.T) / 2


def normalize_kernel(diffusion_matrix: coo_matrix) -> csr_matrix:
    row_sums = cp.array(diffusion_matrix.sum(axis = 1)).flatten()
    inv_sqrt_diag = 1 / cp.sqrt(row_sums)

    new_data = diffusion_matrix.data * inv_sqrt_diag[diffusion_matrix.row] * inv_sqrt_diag[diffusion_matrix.col]
    
    normalized_kernel = csr_matrix((new_data, (diffusion_matrix.row, diffusion_matrix.col)), shape=diffusion_matrix.shape)    

    normalized_kernel = symmetrize(normalized_kernel)
    
    return normalized_kernel, inv_sqrt_diag


def eigendecomposition(
    config,
    matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    device = config.device
    tol = 1e-10
    maxiter = 10000
    k = config.num_eigenvectors
    which = "LM"

    # if device == "cpu":
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, 
        k=k, 
        which=which, 
        maxiter=maxiter, 
        tol=tol
    )
    # elif device == "cuda":
    #     with cupy.cuda.Device(int(device.split(":")[-1])):
    #         cupy_matrix = cupyx.scipy.sparse.csr_matrix(matrix)
    #         eigvals, eigvecs = cupyx.scipy.sparse.linalg.eigsh(
    #             cupy_matrix,
    #             k=k,
    #             which=which,
    #             maxiter=maxiter,
    #             tol=tol
    #         )
    #         eigvals, eigvecs = cupy.asnumpy(eigvals), cupy.asnumpy(eigvecs)
    # else:
    #     raise ValueError(f"Unsupported device: {device}")    

    # Sort in descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    return eigvals, eigvecs


def spectral_embedding(
    config: HDMConfig,
    kernel: csr_matrix,
    inv_sqrt_diag: cp.ndarray,
):
    sqrt_diag = sparse.diags(inv_sqrt_diag, 0)

    eigvals, eigvecs = eigendecomposition(config, kernel)
    
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(cp.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full





