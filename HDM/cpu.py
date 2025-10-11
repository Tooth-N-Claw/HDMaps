from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, diags, block_array
import numpy as np
import scipy.sparse as sparse
from .utils import HDMConfig
from scipy.sparse.linalg import LinearOperator

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

    # fiber_base_row = np.searchsorted(block_indices, fiber_kernel.row, side="right") - 1
    # fiber_base_col = np.searchsorted(block_indices, fiber_kernel.col, side="right") - 1
    # block_vals = np.array(base_kernel[fiber_base_row, fiber_base_col]).reshape(-1)

    # joint_data = fiber_kernel.data * block_vals
    # joint_kernel = coo_matrix(
    #     (joint_data, (fiber_kernel.row, fiber_kernel.col)), shape=fiber_kernel.shape
    # )

    # joint_kernel.eliminate_zeros()
    # # return joint_kernel
    # n = base_kernel.shape[0]
    # m = sample_dists[0].shape[0]
    
    # fiber_kernel = np.empty((block_indices[-1], block_indices[-1]))

    # # Create ALL (i,j) pairs (full n×n grid)
    # i_indices = np.repeat(np.arange(n), n)  # [0,0,0,..., 1,1,1,..., n-1,n-1,n-1]
    # j_indices = np.tile(np.arange(n), n)    # [0,1,2,...,n-1, 0,1,2,...,n-1, ...]

    # # Stack and compute (fully vectorized)
    # all_maps = np.stack([maps[i,j] for i,j in zip(i_indices, j_indices)])
    # all_probs = np.array([base_kernel[i,j] for i,j in zip(i_indices, j_indices)])
    # all_dists = np.stack([sample_dists[i].toarray() for i in i_indices])
    
    # results = all_probs[:, None, None] * (all_maps @ all_dists)

    # for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
    #     fiber_kernel[block_indices[i]:block_indices[i+1], block_indices[j]:block_indices[j+1]] = results[idx]
    #     if i != j:  # Symmetry
    #         fiber_kernel[block_indices[j]:block_indices[j+1], block_indices[i]:block_indices[i+1]] = results[idx].T
            
    # return csr_matrix(fiber_kernel)
    

    # n = base_kernel.shape[0]
    # m = sample_dists[0].shape[0]
    
    # # Create n×n grid to store blocks
    # blocks = [[None for _ in range(n)] for _ in range(n)]
    
    # # Compute each block
    # for i in range(n):
    #     for j in range(n):
    #         prob = base_kernel[i, j]
    #         map_ij = maps[i, j]
 
    #         block = prob * (map_ij @ sample_dists[i])
            
    #         # Store as CSR (or keep dense if preferred)
    #         blocks[i][j] = csr_matrix(block)


    # # Combine all blocks into one big matrix
    # fiber_kernel = block_array(blocks, format='csr')
    
    # return fiber_kernel

    # n = base_kernel.shape[0]
    
    # blocks = [[None for _ in range(n)] for _ in range(n)]
    
    # batch_size = 100 
    # all_pairs = [(i, j) for i in range(n) for j in range(n)]
    
    # for batch_start in range(0, len(all_pairs), batch_size):
    #     batch_pairs = all_pairs[batch_start:batch_start + batch_size]
        
    #     batch_maps = np.stack([maps[i, j] for i, j in batch_pairs])
    #     batch_probs = np.array([base_kernel[i, j] for i, j in batch_pairs])
    #     batch_dists = np.stack([sample_dists[i].toarray() for i, j in batch_pairs])
        
    #     batch_results = batch_probs[:, None, None] * (batch_maps @ batch_dists)
        
    #     for idx, (i, j) in enumerate(batch_pairs):
    #         blocks[i][j] = csr_matrix(batch_results[idx])
    
    # return block_array(blocks, format='csr')
    
    
    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m


    def diffusion_matvec(v):
        v_blocks = v.reshape(n, m)
        result = np.zeros((n, m))
        
        for i in range(n):
            # Vectorize inner loop
            temp_all = sample_dists[i] @ v_blocks.T  # (m, n)
            
            # Apply all maps[i, j] at once: (n, m, m) @ (m, n) -> (n, m)
            mapped = np.einsum('jkl,lj->jk', maps[i], temp_all)
            # Or: mapped = (maps_4d[i] @ temp_all.T).T
            
            # Weighted sum
            result[i] = base_kernel[i, :] @ mapped
        
        return result.ravel()
    # sample_dists_dense = np.array([sample_dists[i].toarray() for i in range(n)])  # (n, m, m)
    # print(f"base_kernel.shape {base_kernel.shape}")
    # def diffusion_matvec(v):
    #     v_blocks = v.reshape(n, m)
        
    #     # Fully vectorized - no loops!
    #     temp_all = np.einsum('ikl,lj->ikj', sample_dists_dense, v_blocks.T)  # (n, m, n)
    #     mapped = np.einsum('ijkl,ilj->ijk', maps, temp_all)  # (n, n, m)
    #     result = np.einsum('ij,ijk->ik', base_kernel.toarray(), mapped)  # (n, m)
        
    #     return result.ravel()

    # Compute row sums ONCE upfront
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
