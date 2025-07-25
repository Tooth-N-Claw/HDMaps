from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import scipy.sparse as sparse
from .utils import HDMConfig
from HDM import utils


def compute_joint_kernel(
    base_kernel: csr_matrix,
    fiber_kernel: coo_matrix,
    block_indices: np.ndarray
) -> coo_matrix:
    fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side='right') - 1
    fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side='right') - 1

    block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)

    joint_data = fiber_kernel.data * block_vals
    joint_kernel = coo_matrix((joint_data, (fiber_kernel.row, fiber_kernel.col)), shape=fiber_kernel.shape)

    joint_kernel.eliminate_zeros()
    return joint_kernel


def symmetrize(mat):
    return (mat + mat.T) / 2


def normalize_kernel(diffusion_matrix: coo_matrix) -> csr_matrix:
    row_sums = np.array(diffusion_matrix.sum(axis = 1)).flatten()
    inv_sqrt_diag = 1 / np.sqrt(row_sums)

    new_data = diffusion_matrix.data * inv_sqrt_diag[diffusion_matrix.row] * inv_sqrt_diag[diffusion_matrix.col]
    
    normalized_kernel = csr_matrix((new_data, (diffusion_matrix.row, diffusion_matrix.col)), shape=diffusion_matrix.shape)    

    normalized_kernel = symmetrize(normalized_kernel)
    
    return normalized_kernel, inv_sqrt_diag


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

    eigvals, eigvecs = sparse.linalg.eigsh(
        matrix, 
        k=k, 
        which=which, 
        maxiter=maxiter, 
        tol=tol
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

    import matplotlib.pyplot as plt

    coo = kernel.tocoo()
    rows, cols = coo.row, coo.col
    values = coo.data

    # Create scatter plot to mimic heatmap
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(cols, rows, c=values, cmap='viridis', s=1, marker='s')
    plt.colorbar(sc, label='Value')
    plt.title('Sparse Matrix Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.gca().invert_yaxis()  # Optional: match matrix orientation
    plt.tight_layout()
    plt.show()

    eigvals, eigvecs = eigendecomposition(config, kernel)   
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda

    return bundle_HDM_full
