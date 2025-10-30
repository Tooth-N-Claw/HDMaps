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
    base_kernel, sample_dists: list, maps
):

    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m

    # Convert to GPU CSR matrix and multiply by 0.5 for symmetry
    base_kernel_gpu = cp_sparse.csr_matrix(base_kernel)
    base_kernel_gpu.data *= 0.5  # we multiply by 0.5 here for symmetry when we later in diffusion_matvec do 0.5 * (M D + D^T M^T), we moved the 0.5 for efficiency

    # Iterate over CSR structure without materializing rows/cols arrays
    for i in range(base_kernel_gpu.shape[0]):
        start = base_kernel_gpu.indptr[i]
        end = base_kernel_gpu.indptr[i + 1]
        for idx in range(start, end):
            j = base_kernel_gpu.indices[idx]
            value = base_kernel_gpu.data[idx]
            maps[i][j].data *= value  # In-place multiplication

    maps = cp_sparse.bmat([[cp_sparse.csr_matrix(maps[i, j]) for j in range(n)] for i in range(n)], format='csr')

    # Build block diagonal matrix using bmat
    sample_dists_gpu = [cp_sparse.csr_matrix(sd) for sd in sample_dists]
    sample_dists_blocks = [[sample_dists_gpu[i] if i == j else None for j in range(n)] for i in range(n)]
    sample_dists = cp_sparse.bmat(sample_dists_blocks, format='csr')

    def diffusion_matvec(v):
        return maps @ (sample_dists @ v) + sample_dists.T @ (maps.T @ v)

    row_sums = diffusion_matvec(cp.ones(total_size))
    inv_sqrt_diag = 1 / cp.sqrt(row_sums)
    D_inv_sqrt = cp_sparse.diags(inv_sqrt_diag, format='csr')
    # precomputes normalized components to offload normalized_diffusion_matvec function since it will be called many times
    sample_dists = sample_dists @ D_inv_sqrt
    maps = D_inv_sqrt @ maps

    # def normalized_matvec(v):
    #     return inv_sqrt_diag * diffusion_matvec(inv_sqrt_diag * v)

    def normalized_diffusion_matvec(v):
        return maps @ (sample_dists @ v) + sample_dists.T @ (maps.T @ v)

    normalized_kernel = cp_linalg.LinearOperator(
        shape=(total_size, total_size),
        matvec=normalized_diffusion_matvec,
        dtype=cp.float32
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


    eigvals, eigvecs = eigendecomposition(config, kernel)
    sqrt_diag = cp_sparse.diags(inv_sqrt_diag, 0)

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = cp_sparse.diags(cp.sqrt(eigvals[1:]), 0)
    bundle_HDM_full = bundle_HDM @ sqrt_lambda
    bundle_HDM_full = cp.asnumpy(bundle_HDM_full)

    return bundle_HDM_full
