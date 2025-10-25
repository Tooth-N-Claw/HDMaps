from scipy.sparse import csr_matrix, coo_matrix, block_array, block_diag
import numpy as np
import scipy.sparse as sparse
from .utils import HDMConfig
from scipy.sparse.linalg import LinearOperator

def compute_joint_kernel_linear_operator(
    base_kernel: csr_matrix, sample_dists: list[np.ndarray], block_indices: np.ndarray, maps
) -> coo_matrix:
    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m
    
    base_kernel = base_kernel.toarray()
    maps = maps * base_kernel
    maps = block_array(maps)
    sample_dists = block_diag(sample_dists)
    
    def diffusion_matvec(v):
        return 0.5 * ( maps @ (sample_dists @ v) + sample_dists.T @ (maps.T @ v) )


    row_sums = diffusion_matvec(np.ones(total_size))
    inv_sqrt_diag = 1 / np.sqrt(row_sums)
    
    def normalized_matvec(v):
        return inv_sqrt_diag * diffusion_matvec(inv_sqrt_diag * v)
    
    normalized_kernel = LinearOperator(
        shape=(total_size, total_size),
        matvec=normalized_matvec,
        dtype=np.float32
    )

    return normalized_kernel, inv_sqrt_diag


def eigendecomposition(
    config,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    tol = 1e-10
    maxiter = 10000
    k = config.num_eigenvectors
    which = "LM"

    rng = np.random.default_rng(42)
    v0 = rng.random(size=matrix.shape[0]).astype(np.float64)
    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, k=k, which=which, maxiter=maxiter, tol=tol, v0=v0
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


    eigvals, eigvecs = eigendecomposition(config, kernel)
    sqrt_diag = sparse.diags(inv_sqrt_diag, 0)

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
