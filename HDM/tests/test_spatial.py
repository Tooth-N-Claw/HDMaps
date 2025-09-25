import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from HDM import cpu


def test_compute_joint_kernel():
    # Define base kernel as CSR (Compressed Sparse Row)
    base_kernel_data = np.array([[1, 0], [1, 1]])
    base_kernel = csr_matrix(base_kernel_data)

    # Define fiber kernel as COO (Coordinate format)
    fiber_kernel_data = np.array(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 1, 1]]
    )
    fiber_kernel = coo_matrix(fiber_kernel_data)

    # Block indices
    block_indices = np.array([0, 2])

    # Compute joint kernel
    backend = cpu  # Use the CPU backend
    joint_kernel = backend.compute_joint_kernel(
        base_kernel, fiber_kernel, block_indices
    )
    joint_kernel_dense = joint_kernel.toarray()

    # Define expected output
    expected_joint_kernel = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1]]
    )

    # Assertion
    assert np.array_equal(joint_kernel_dense, expected_joint_kernel), (
        "Joint kernel output doesn't match expected output"
    )
