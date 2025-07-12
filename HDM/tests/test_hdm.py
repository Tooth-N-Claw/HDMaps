import pytest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.experimental import sparse


from HDM.kernels import compute_joint_kernel


def test_compute_joint_kernel():

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

    joint_kernel = compute_joint_kernel(base_kernel, fiber_kernel, block_indices)
    joint_kernel = joint_kernel.todense()

    expected_joint_kernel = jnp.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 1]]
    )

    assert jnp.array_equal(joint_kernel, expected_joint_kernel), (  
        "Joint kernel output doesn't match expected output"
    )
