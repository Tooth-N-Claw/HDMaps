import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.experimental import sparse as jsparse
from jax.experimental.sparse.linalg import lobpcg_standard
from scipy.sparse import csr_matrix, block_array, block_diag
import numpy as np
from .utils import HDMConfig


def compute_joint_kernel_linear_operator(
    base_kernel: csr_matrix,
    sample_dists: list,
    block_indices: jnp.ndarray,
    maps
):
    n = base_kernel.shape[0]
    m = sample_dists[0].shape[0]
    total_size = n * m

    print(1)
    # Apply base_kernel weights to maps (scipy operation)
    base_kernel_dense = base_kernel.toarray()
    maps = maps * base_kernel_dense
    print(2)

    # Create single large sparse matrices using scipy
    maps_big = block_array(maps)
    sample_dists_big = block_diag(sample_dists)
    print(3)

    # Convert to JAX BCOO
    maps_bcoo = jsparse.BCOO.from_scipy_sparse(maps_big)
    sample_dists_bcoo = jsparse.BCOO.from_scipy_sparse(sample_dists_big)
    print(4)

    def diffusion_matvec(v):
        # Simple matrix operations, same as cpu.py
        return 0.5 * (maps_bcoo @ (sample_dists_bcoo @ v) + sample_dists_bcoo.T @ (maps_bcoo.T @ v))

    row_sums = diffusion_matvec(jnp.ones(total_size))
    inv_sqrt_diag = 1.0 / jnp.sqrt(row_sums)

    def normalized_matvec(v):
        if v.ndim == 2:
            return inv_sqrt_diag[:, None] * diffusion_matvec(inv_sqrt_diag[:, None] * v)
        return inv_sqrt_diag * diffusion_matvec(inv_sqrt_diag * v)

    return normalized_matvec, inv_sqrt_diag


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
