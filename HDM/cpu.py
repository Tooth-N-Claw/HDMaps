from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, diags, block_array, issparse
import numpy as np
import scipy.sparse as sparse
from .utils import HDMConfig
from scipy.sparse.linalg import LinearOperator
from numba import jit, prange
import multiprocessing
# def compute_joint_kernel(
#     base_kernel: csr_matrix, fiber_kernel: coo_matrix, block_indices: np.ndarray
# ) -> coo_matrix:
#     fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side="right") - 1
#     fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side="right") - 1
#     block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)

#     joint_data = fiber_kernel.data * block_vals
#     joint_kernel = coo_matrix(
#         (joint_data, (fiber_kernel.row, fiber_kernel.col)), shape=fiber_kernel.shape
#     )

#     joint_kernel.eliminate_zeros()
#     return joint_kernel

def compute_joint_kernel(
    base_kernel: csr_matrix, sample_dists: list[np.ndarray], block_indices: np.ndarray, maps
) -> coo_matrix:

    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    
    # Create n×n grid to store blocks
    blocks = [[None for _ in range(n)] for _ in range(n)]
    
    # Compute each block
    for i in range(n):
        for j in range(i+1):
            prob = base_kernel[i, j]
            map_ij = maps[i, j]
 
            block = prob * (map_ij @ sample_dists[i])
            
            # Store as CSR (or keep dense if preferred)
            blocks[i][j] = csr_matrix(block)
            if i != j:
                blocks[j][i] = csr_matrix(block.T)


    # Combine all blocks into one big matrix
    fiber_kernel = block_array(blocks, format='csr')
    
    return fiber_kernel


def compute_joint_kernel_linear_operator(
    base_kernel: csr_matrix, sample_dists: list[np.ndarray], block_indices: np.ndarray, maps
) -> coo_matrix:

    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m
 
    for i in range(n):
        for j in range(i+1):
            # print(0)
            # print(maps[i,j])            
            maps[i,j] *= base_kernel[i,j]
            # print(1)
            # print(maps[i,j])
            # print(1.1)
            # print(sample_dists[i].shape)
            # print(sample_dists[i].nnz)
            maps[i,j] = sample_dists[i] @ maps[i,j]
            # print(2)
            # print(maps[i,j])

    def diffusion_matvec(v):
        v_blocks = v.reshape(n, m)
        result = np.zeros((n, m))
        
        pairs = [(i, j) for i in range(n) for j in range(i+1)]

        for i, j in pairs:
            result[i] += maps[i,j] @ v_blocks[j]
            # temp = sample_dists[i] @ v_blocks[j]
            # result[i] += maps[i, j] @ temp

            if i != j:
                result[j] += maps[i,j].T @ v_blocks[i]
                # temp_transpose = maps[i, j].T @ v_blocks[i]
                # result[j] += sample_dists[i].T @ temp_transpose
        return result.ravel()

    print("Computing normalization...")
    row_sums = diffusion_matvec(np.ones(total_size))
    inv_sqrt_diag = 1 / np.sqrt(row_sums)

    def normalized_matvec(v):
        """Use precomputed inv_sqrt_diag"""
        return inv_sqrt_diag * diffusion_matvec(inv_sqrt_diag * v)

    normalized_kernel = LinearOperator(
        shape=(total_size, total_size),
        matvec=normalized_matvec,
        dtype=np.float64
    )
    
    return normalized_kernel, inv_sqrt_diag

def symmetrize(mat):
    return (mat + mat.T) / 2


# def normalize_kernel(diffusion_matrix: coo_matrix) -> csr_matrix:
#     row_sums = np.array(diffusion_matrix.sum(axis=1)).flatten()

#     inv_sqrt_diag = 1 / np.sqrt(row_sums)
#     new_data = (
#         diffusion_matrix.data
#         * inv_sqrt_diag[diffusion_matrix.row]
#         * inv_sqrt_diag[diffusion_matrix.col]
#     )

#     normalized_kernel = csr_matrix(
#         (new_data, (diffusion_matrix.row, diffusion_matrix.col)),
#         shape=diffusion_matrix.shape,
#     )

#     normalized_kernel = symmetrize(normalized_kernel)

#     return normalized_kernel, inv_sqrt_diag

def normalize_kernel(diffusion_matrix: csr_matrix) -> bsr_matrix:
    row_sums = np.array(diffusion_matrix.sum(axis=1)).flatten()
    inv_sqrt_diag = 1 / np.sqrt(row_sums)
    
    D_inv_sqrt = diags(inv_sqrt_diag, format='csr')
    
    # Normalize: D^(-1/2) @ K @ D^(-1/2)
    normalized_kernel = D_inv_sqrt @ diffusion_matrix @ D_inv_sqrt
    
    # Convert to BSR with same blocksize
    normalized_kernel = normalized_kernel.tobsr()

    normalized_kernel = symmetrize(normalized_kernel)
    
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
