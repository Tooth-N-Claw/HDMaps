import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import cupyx.scipy.sparse.linalg as cp_linalg
cp.random.seed(0)

def compute_joint_kernel(
    base_kernel, fiber_kernel, block_indices: np.ndarray
) -> cp_sparse.coo_matrix:
    """GPU-accelerated version using CuPy"""

    fiber_row_gpu = cp.asarray(fiber_kernel.row)
    fiber_col_gpu = cp.asarray(fiber_kernel.col)
    fiber_data_gpu = cp.asarray(fiber_kernel.data)
    block_indices_gpu = cp.asarray(block_indices)

    fiber_base_row = cp.searchsorted(block_indices_gpu, fiber_row_gpu, side="right") - 1
    fiber_base_col = cp.searchsorted(block_indices_gpu, fiber_col_gpu, side="right") - 1
    base_kernel_gpu = cp_sparse.csr_matrix(base_kernel)

    block_vals = base_kernel_gpu[fiber_base_row, fiber_base_col].flatten()

    joint_data = fiber_data_gpu * block_vals

    joint_kernel_gpu = cp_sparse.coo_matrix(
        (joint_data, (fiber_row_gpu, fiber_col_gpu)), shape=fiber_kernel.shape
    )
    joint_kernel_gpu.eliminate_zeros()
    return joint_kernel_gpu


def symmetrize(mat):
    return (mat + mat.T) / 2


def compute_joint_kernel_linear_operator(
    base_kernel, sample_dists: list, block_indices, maps
):

    # Convert sample_dists to GPU dense matrices for faster batched operations
    print("Converting sample_dists to dense...")
    sample_dists_dense = cp.stack([cp.asarray(sd.toarray()) for sd in sample_dists])
    # sample_dists_dense_T = cp.transpose(sample_dists_dense, (0, 2, 1))

    n = base_kernel.shape[0]
    m = sample_dists_dense.shape[1]
    total_size = n * m

    # Keep maps sparse to save memory
    maps_gpu = {}
    for i in range(n):
        for j in range(i+1):
            maps_gpu[(i, j)] = cp_sparse.csr_matrix(maps[i, j] * base_kernel[i, j])

    # Precompute transposed maps
    pairs = [(i, j) for i in range(n) for j in range(i+1)]
    # maps_T = {(i, j): maps_gpu[(i, j)].T for i, j in pairs}

    def diffusion_matvec(v):
        v_blocks = v.reshape(n, m)
        result = cp.zeros((n, m))

        # Batch compute all sample_dists[i] @ v_blocks[j] operations at once
        temp_all = cp.einsum('imk,jk->ijm', sample_dists_dense, v_blocks)  # (n, n, m)

        # Now apply sparse maps operations in loop (unavoidable since maps is sparse)
        for i, j in pairs:
            result[i] += maps_gpu[(i, j)] @ temp_all[i, j]
            # if i != j:
            #     # For transpose path, compute maps_T @ v_blocks, then apply sample_dists_T
            #     temp_transpose = maps_T[(i, j)] @ v_blocks[i]
            #     result[j] += sample_dists_dense_T[i] @ temp_transpose

        return result.ravel()

    print("Computing normalization...")
    row_sums = diffusion_matvec(cp.ones(total_size))
    inv_sqrt_diag = 1 / cp.sqrt(row_sums)

    def normalized_matvec(v):
        return symmetrize(inv_sqrt_diag * diffusion_matvec(inv_sqrt_diag * v))

    normalized_kernel = cp_linalg.LinearOperator(
        shape=(total_size, total_size),
        matvec=normalized_matvec,
        dtype=cp.float64
    )

    return normalized_kernel, inv_sqrt_diag


def normalize_kernel(
    diffusion_matrix: cp_sparse.coo_matrix,
) -> tuple[cp_sparse.csr_matrix, cp.ndarray]:
    row_sums = cp.asarray(diffusion_matrix.sum(axis=1)).flatten()
    inv_sqrt_diag = 1 / cp.sqrt(row_sums)

    new_data = (
        diffusion_matrix.data
        * inv_sqrt_diag[diffusion_matrix.row]
        * inv_sqrt_diag[diffusion_matrix.col]
    )

    normalized_kernel = cp_sparse.csr_matrix(
        (new_data, (diffusion_matrix.row, diffusion_matrix.col)),
        shape=diffusion_matrix.shape,
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
    
    rng = np.random.default_rng(42)
    v0_cpu = rng.random(size=matrix.shape[0]).astype(np.float64)
    v0_gpu = cp.asarray(v0_cpu)
    eigvals, eigvecs = cp_linalg.eigsh(
        matrix, k=k, which=which, maxiter=maxiter, tol=tol, v0=v0_gpu
    )

    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    return eigvals, eigvecs


def spectral_embedding(
    config,
    kernel: cp_sparse.csr_matrix,
    inv_sqrt_diag: cp.ndarray,
) -> cp.ndarray:
    sqrt_diag = cp_sparse.diags(inv_sqrt_diag, 0)

    eigvals, eigvecs = eigendecomposition(config, kernel)

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = cp_sparse.diags(cp.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda
    bundle_HDM_full = cp.asnumpy(bundle_HDM_full)

    return bundle_HDM_full
