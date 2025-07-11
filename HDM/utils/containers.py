from typing import NamedTuple, Optional
import scipy.sparse as sparse
import jax.numpy as jnp
from jax.experimental import sparse


class JaxCoo(NamedTuple):
    row: jnp.ndarray
    col: jnp.ndarray
    data: jnp.ndarray
    shape: tuple[int, int]

    def with_data(self, new_data):
        return self._replace(data = new_data)

    def purge_zeros(self):
        mask = self.data != 0
        return JaxCoo(
            row = self.row[mask],
            col = self.col[mask],
            data = self.data[mask],
            shape = self.shape
        )

    def toscipy(self):
        return sparse.coo_matrix((self.data, (self.row, self.col)), self.shape)

  
def jax_coo(arr):
    return JaxCoo(
        row = jnp.array(arr.row),
        col = jnp.array(arr.col),
        data = jnp.array(arr.data),
        shape = arr.shape
    )
        


class HDMData(NamedTuple):
    data_samples: list[jnp.ndarray]
    cumulative_block_indices: jnp.ndarray 
    base_kernel: sparse.BCOO | None = None
    fiber_kernel: sparse.BCOO | None = None
    base_distances: sparse.BCOO | None = None
    fiber_distances: sparse.BCOO | None = None

    # def with_base_kernel(self, base_kernel):
    #     return self._replace(base_kernel = base_kernel)

    # def with_fiber_kernel(self, fiber_kernel):
    #     return self._replace(fiber_kernel = fiber_kernel)


class HDMConfig(NamedTuple):
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    device: str | None = "CPU"
    base_dist_func: str = "Frobenius"
    fiber_dist_func: str = "Euclidean"
    base_sparsity: float = 0.08
    fiber_sparsity: float = 0.08
