


import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from jax.experimental import sparse
import numpy as np


base_kernel = jnp.array(
    [[1, 0],
     [1, 1]]
)
base_kernel = sparse.BCOO.fromdense(base_kernel)

fiber_kernel = jnp.array(
    [[1, 0, 0, 0],
     [0, 1, 1, 0],
     [0, 0, 0, 0],
     [0, 1, 1, 1]]
)
fiber_kernel = sparse.BCOO.fromdense(fiber_kernel)

block_indices = jnp.array([0, 2])

from HDM.kernels import compute_joint_kernel

joint_kernel = compute_joint_kernel(base_kernel, fiber_kernel, block_indices)

print("what")
print(block_indices)
print(base_kernel.todense())
print(fiber_kernel.todense())
print(joint_kernel.todense())
