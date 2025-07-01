from typing import NamedTuple, Optional
import jax.numpy as jnp


class Coo(NamedTuple):
    row_idx: jnp.ndarray
    col_idx: jnp.ndarray
    values: jnp.ndarray
    shape: tuple[int, int]

    def with_values(self, values):
        return self._replace(values = values)


class HDMData(NamedTuple):
    data_samples: jnp.ndarray
    base_kernel: Optional[Coo] = None
    fiber_kernel: Optional[Coo] = None
    base_distances: Optional[Coo] = None
    fiber_distances: Optional[Coo] = None

    def with_base_kernel(self, base_kernel):
        return self._replace(base_kernel = base_kernel)

    def with_fiber_kernel(self, fiber_kernel):
        return self._replace(fiber_kernel = fiber_kernel)


class HDMConfig(NamedTuple):
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    device: str | None = "CPU"
    base_dist_func: str = "Frobenius"
    fiber_dist_func: str = "Euclidean"
    base_sparsity: float = 0.08
    fiber_sparsity: float = 0.08
