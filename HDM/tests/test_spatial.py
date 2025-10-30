import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from HDM import hdm_embed, HDMConfig

def test_cpu_hdm_runs():
    samples = [np.random.rand(10, 3) for _ in range(5)]
    config = HDMConfig(
        base_epsilon=0.5,
        fiber_epsilon=0.5,
        base_knn=3,
        fiber_knn=3,
        device="cpu",
        verbose=False
    )
    maps = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            if i == j:
                maps[i, j] = csr_matrix(np.eye(10))
            else:
                maps[i, j] = csr_matrix((10, 10))       
    diffusion_coords = hdm_embed(
        config=config,
        data_samples=samples,
        maps=maps
    )
    
    
def test_jax_hdm_runs():
    samples = [np.random.rand(10, 3) for _ in range(5)]
    config = HDMConfig(
        base_epsilon=0.5,
        fiber_epsilon=0.5,
        base_knn=3,
        fiber_knn=3,
        device="jax",
        verbose=False
    )
    maps = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            if i == j:
                maps[i, j] = csr_matrix(np.eye(10))
            else:
                maps[i, j] = csr_matrix((10, 10))       
    diffusion_coords = hdm_embed(
        config=config,
        data_samples=samples,
        maps=maps
    )

# def test_compute_joint_kernel():
#     # Define base kernel as CSR (Compressed Sparse Row)
#     base_kernel_data = np.array([[1, 0], [1, 1]])
#     base_kernel = csr_matrix(base_kernel_data)

#     # Define fiber kernel as COO (Coordinate format)
#     fiber_kernel_data = np.array(
#         [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 1, 1]]
#     )
#     fiber_kernel = coo_matrix(fiber_kernel_data)

#     # Block indices
#     block_indices = np.array([0, 2])

#     # Compute joint kernel
#     backend = cpu  # Use the CPU backend
#     joint_kernel = backend.compute_joint_kernel(
#         base_kernel, fiber_kernel, block_indices
#     )
#     joint_kernel_dense = joint_kernel.toarray()

#     # Define expected output
#     expected_joint_kernel = np.array(
#         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1]]
#     )

#     # Assertion
#     assert np.array_equal(joint_kernel_dense, expected_joint_kernel), (
#         "Joint kernel output doesn't match expected output"
#     )
