import pickle
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
# from src.HDM import *




# def test_compute_base_dist():
#     data_samples = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 12]])]
#     base_norm_function = lambda x, y: np.linalg.norm(x - y)
#     result = compute_base_dist(data_samples, base_norm_function, 4)
#     expected_result = np.array([[0,3],[0,0]])
#     assert (result == expected_result).all()



# def test_compute_fiber_dist():
#     data_samples = [np.array([[2, 4, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [4, 3, 6]])]
#     base_dist_mat = lil_matrix([[0, 3], [0, 0]])
#     sparsity_param_fiber = 1
#     data_object_maps = {(0, 1): csr_matrix([[1, 0], [0, 1], [0, 1]]), (1, 0): csr_matrix([[1, 0, 0], [0, 1, 1]])}
#     fiber_norm_function = lambda x, y: np.linalg.norm(x - y)
#     block_indices = np.insert(np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 0, 0)
#     result = compute_fiber_dist(base_dist_mat, sparsity_param_fiber, data_object_maps, fiber_norm_function, data_samples, block_indices)
#     expected_result = np.array([[0, 0, 0, np.linalg.norm(data_samples[0][0] - data_samples[1][0]), 0], 
#                                 [0, 0, 0, 0, np.linalg.norm(data_samples[0][1] - data_samples[1][1])],
#                                 [0, 0, 0, 0, np.linalg.norm(data_samples[0][2] - data_samples[1][1])],
#                                 [0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0]])
#     # convert lil_matrix to numpy
#     result = result.toarray()
#     np.testing.assert_allclose(result, expected_result, rtol=1e-5, atol=1e-8)
         
    
# # def test_compute_horizontal_diffusion_laplacianI():
# #     data_samples = [np.array([[2, 4, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [4, 3, 6]])]
# #     base_dist_mat = lil_matrix([[0, 3], [0, 0]])
# #     fiber = np.array([[0, 0, 0, np.linalg.norm(data_samples[0][0] - data_samples[1][0]), 0], 
# #                                 [0, 0, 0, 0, np.linalg.norm(data_samples[0][1] - data_samples[1][1])],
# #                                 [0, 0, 0, 0, np.linalg.norm(data_samples[0][2] - data_samples[1][1])],
# #                                 [0, 0, 0, 0, 0],
# #                                 [0, 0, 0, 0, 0]])


# def test_eigendecomposition():
#     A = np.array([[1, 0], [0, 2]])
#     eigvals, eigvecs = eigendecomposition(A, 100)
#     expected_eigvals = np.array([2, 1])
#     expected_eigvecs = np.array([[0, 1], [1, 0]])
    
#     assert np.allclose(eigvals, expected_eigvals)
#     assert np.allclose(np.abs(eigvecs), np.abs(expected_eigvecs))