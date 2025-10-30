import time
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax.experimental.sparse.linalg import lobpcg_standard
from scipy.sparse import csr_matrix, block_array, block_diag, kron, eye, diags
import numpy as np
from .utils import HDMConfig


def compute_joint_kernel_linear_operator(
    base_kernel: csr_matrix,
    sample_dists: list,
    maps
):
    """Creates the normalized joint kernel as a JAX function. Jax does not however good support for sparse matrices, so is first done in scipy, then converted to jax sparse matrices."""
    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m


    base_kernel.multiply(0.5) # we multiply by 0.5 here for symmetry when we later in diffison_matecv do 0.5 * (M D + D^T M^T), we moved the 0.5 for efficiency
    
    # Iterate over CSR structure without materializing rows/cols arrays
    for i in range(base_kernel.shape[0]):
        start = base_kernel.indptr[i]
        end = base_kernel.indptr[i + 1]
        for idx in range(start, end):
            j = base_kernel.indices[idx]
            value = base_kernel.data[idx]
            maps[i][j].data *= value  # In-place multiplication

    maps = block_array(maps, format='csr')
    
    sample_dists_scipy = block_diag(sample_dists) 

    def diffusion_matvec(v):
        return maps @ (sample_dists_scipy @ v) + sample_dists_scipy.T @ (maps.T @ v)

    row_sums = diffusion_matvec(jnp.ones(total_size))
    inv_sqrt_diag = 1.0 / jnp.sqrt(row_sums)
    inv_sqrt_diag_np = np.array(inv_sqrt_diag) 
    D_inv_sqrt = diags(inv_sqrt_diag_np, format='csr')

    # precomputes normalized components to offload normalized_diffusion_matvec function since it will be called many times
    sample_dists = sample_dists_scipy @ D_inv_sqrt
    maps = D_inv_sqrt @ maps

    # At the moment of writing this code (jax 0.8.0) jax BCSR do not support tranpsose operation, so we precompute maps_T and sample_dists_T
    maps_T = jsparse.BCSR.from_scipy_sparse(maps.T)    
    maps = jsparse.BCSR.from_scipy_sparse(maps)
    sample_dists_T = jsparse.BCSR.from_scipy_sparse(sample_dists.T)
    sample_dists = jsparse.BCSR.from_scipy_sparse(sample_dists)

    @jax.jit
    def normalized_diffusion_matvec(v):
        return maps @ (sample_dists @ v) + sample_dists_T @ (maps_T @ v)

    return normalized_diffusion_matvec, inv_sqrt_diag


def eigendecomposition(config, matvec_fn, matrix_size):
    k = config.num_eigenvectors
    tol = 1e-10
    maxiter = 10000

    key = jax.random.PRNGKey(42)
    X0 = jax.random.normal(key, (matrix_size, k), dtype=jnp.float32)

    eigvals, eigvecs, n_iter = lobpcg_standard(
        A=matvec_fn,
        X=X0,
        m=maxiter,
        tol=tol
    )

    sort_indices = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]

    return eigvals, eigvecs


def spectral_embedding(
    config: HDMConfig,
    normalized_kernel,
    inv_sqrt_diag: jnp.ndarray
) -> np.ndarray:
    matrix_size = len(inv_sqrt_diag)
    eigvals, eigvecs = eigendecomposition(config, normalized_kernel, matrix_size)

    bundle_HDM = inv_sqrt_diag[:, None] * eigvecs[:, 1:]
    sqrt_lambda = jnp.sqrt(eigvals[1:])
    bundle_HDM_full = bundle_HDM * sqrt_lambda[None, :]

    return np.array(bundle_HDM_full)
