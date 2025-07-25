import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import cupyx.scipy.sparse.linalg as cp_linalg
import scipy.sparse as scipy_sparse


# def compute_joint_kernel(
#     base_kernel: scipy_sparse.csr_matrix,
#     fiber_kernel: scipy_sparse.coo_matrix,
#     block_indices: np.ndarray
# ) -> cp_sparse.coo_matrix:
#     # Keep CPU operations until COO matrix construction
#     fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side='right') - 1
#     fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side='right') - 1

#     block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)
    
#     joint_data = fiber_kernel.data * block_vals
    
#     # Convert to CuPy from here
#     joint_data_gpu = cp.asarray(joint_data)
#     fiber_row_gpu = cp.asarray(fiber_kernel.row)
#     fiber_col_gpu = cp.asarray(fiber_kernel.col)
    
#     joint_kernel = cp_sparse.coo_matrix(
#         (joint_data_gpu, (fiber_row_gpu, fiber_col_gpu)), 
#         shape=fiber_kernel.shape
#     )
#     joint_kernel.eliminate_zeros()
    
#     return joint_kernel


def compute_joint_kernel(
    base_kernel,
    fiber_kernel,
    block_indices: np.ndarray
) -> cp_sparse.coo_matrix:
    """GPU-accelerated version using CuPy"""
    
    # Transfer data to GPU
    fiber_row_gpu = cp.asarray(fiber_kernel.row)
    fiber_col_gpu = cp.asarray(fiber_kernel.col)
    fiber_data_gpu = cp.asarray(fiber_kernel.data)
    block_indices_gpu = cp.asarray(block_indices)
    
    # Convert base kernel to GPU
    base_kernel_gpu = cp_sparse.csr_matrix(base_kernel)
    
    fiber_base_row = cp.searchsorted(block_indices_gpu, fiber_row_gpu, side='right') - 1
    fiber_base_col = cp.searchsorted(block_indices_gpu, fiber_col_gpu, side='right') - 1
    
    block_vals = base_kernel_gpu[fiber_base_row, fiber_base_col].toarray().flatten()
    
    joint_data = fiber_data_gpu * block_vals
    
    # Create result and transfer back to CPU
    joint_kernel_gpu = cp_sparse.coo_matrix(
        (joint_data, (fiber_row_gpu, fiber_col_gpu)), 
        shape=fiber_kernel.shape
    )
    joint_kernel_gpu.eliminate_zeros()
    
    return joint_kernel_gpu



def symmetrize(mat):
    return (mat + mat.T) / 2


def normalize_kernel(diffusion_matrix: cp_sparse.coo_matrix) -> tuple[cp_sparse.csr_matrix, cp.ndarray]:
    row_sums = cp.asarray(diffusion_matrix.sum(axis=1)).flatten()
    inv_sqrt_diag = 1 / cp.sqrt(row_sums)

    new_data = diffusion_matrix.data * inv_sqrt_diag[diffusion_matrix.row] * inv_sqrt_diag[diffusion_matrix.col]
    
    normalized_kernel = cp_sparse.csr_matrix(
        (new_data, (diffusion_matrix.row, diffusion_matrix.col)), 
        shape=diffusion_matrix.shape
    )    

    normalized_kernel = symmetrize(normalized_kernel)
    
    return normalized_kernel, inv_sqrt_diag


def eigendecomposition(
    config,
    matrix: cp_sparse.csr_matrix,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Perform eigendecomposition on a sparse matrix using CuPy."""
    tol = 1e-10
    maxiter = 10000
    k = config.num_eigenvectors
    which = "LM"

    # Use CuPy's sparse eigenvalue solver
    eigvals, eigvecs = cp_linalg.eigsh(
        matrix, 
        k=k, 
        which=which, 
        maxiter=maxiter, 
        tol=tol
    )
 
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    return eigvals, eigvecs


def spectral_embedding(
    config,  # Removed HDMConfig type hint to be more general
    kernel: cp_sparse.csr_matrix,
    inv_sqrt_diag: cp.ndarray,
) -> cp.ndarray:
    sqrt_diag = cp_sparse.diags(inv_sqrt_diag, 0)

    eigvals, eigvecs = eigendecomposition(config, kernel)
    
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = cp_sparse.diags(cp.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda
    bundle_HDM_full = cp.asnumpy(bundle_HDM_full)  # Convert to NumPy array for consistency

    return bundle_HDM_full