import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.io import loadmat
import trimesh
from itertools import combinations
from typing import Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import scipy
import pickle
import glob
import os
from visualize import visualize


def sparse_distance_matrix(data_samples1, data_samples2, norm_function, sparsity_param):
    m, n = len(data_samples1), len(data_samples2)
    dist_matrix = lil_matrix((m, n), dtype=np.float32)
    for i, j in combinations(range(max(m, n)), 2):
        dist = norm_function(data_samples1[i], data_samples2[j])
        if dist <= sparsity_param:
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def subsample(data_sample: np.ndarray, supsample_mapping: float) -> np.ndarray: # type: ignore
    pass


def compute_base_dist(
    data_samples: list[np.ndarray],
    base_norm_function: Callable[[np.ndarray, np.ndarray], np.float32],
    sparsity_param_base: float = 10,
) -> csr_matrix:  # type: ignore
    """
    NegativeDistError - if the value returned from the base norm function name is non-positive.
    InputNumOfDistParam - the number of parameters in base norm function name doesn’t match.
    NonBoundedSubSample - subsample mapping is out of the range [0, 1].
    NegativeSparsity - the sparsity param base parameter is negative.

    >>> data_samples = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 12]])]
    >>> base_norm_function = lambda x, y: np.linalg.norm(x - y)
    >>> result = compute_base_dist(data_samples, base_norm_function, 4)
    >>> result[0,1]
    3.0
    >>> result[1,0]
    3.0
    >>> len(result.nonzero()[0])
    2
    """
    # TODO: Make this function calculate distance by the data samples instead of loading from a file
    # k = len(data_samples)
    # base_dist_mat = lil_matrix((k, k), dtype=np.float32)
    # for i in range(k): # TODO: could be optimized
    #     for j in range(i+1, k):
    #         dist = base_norm_function(data_samples[i], data_samples[j])
    #         if dist <= sparsity_param_base and i != j:
    #             base_dist_mat[i, j] = dist

    base_dist_mat = loadmat("../data/cPMSTDistMatrix.mat")["ImprDistMatrix"]
    
    return base_dist_mat


def compute_block(data_sample1, data_sample2, fiber_dist_block_mat, correspondance_matrix: csr_matrix, fiber_norm_func_name, sparsity_param_fiber, i, j) -> None:
    #print(correspondance_matrix.shape)
    rows, cols = correspondance_matrix.nonzero()
    # print(rows, cols)
    #print(sorted(rows))
    #print(data_sample1.shape, data_sample2.shape)


    for t, k in zip(rows, cols):
        # print(k, t)
        dist = fiber_norm_func_name(data_sample1[t], data_sample2[k])
        scaled_dist = dist * correspondance_matrix[t, k]
        if scaled_dist >= sparsity_param_fiber:
            fiber_dist_block_mat[i+t, j+k] = scaled_dist
    

def compute_fiber_dist(base_dist_mat: lil_matrix, sparsity_param_fiber: float, data_object_maps, fiber_norm_func_name, data_samples: list[np.ndarray], block_indices) -> lil_matrix:  # type: ignore
    k = block_indices[-1]
    fiber_dist_block_mat = lil_matrix((k, k), dtype=np.float32)
        
    for i in range(len(data_samples)):
        for j in range(i+1, len(data_samples)):
            print(i, j)
            if base_dist_mat[i, j] != 0:
                compute_block(data_samples[i], data_samples[j], fiber_dist_block_mat, data_object_maps[(i, j)], fiber_norm_func_name, sparsity_param_fiber, block_indices[i], block_indices[j])
                
    return fiber_dist_block_mat.tocsr()



def compute_fiber_epsilon(base_dist_map: csr_matrix) -> float:  # type: ignore
    pass


def compute_base_epsilon(base_dist_mat: csr_matrix) -> float:  # type: ignore
    pass


def _construct_weight_matrix() -> csr_matrix:
    pass 


def construct_block_matrix():
    pass


def symmetrize(matrix: csr_matrix) -> csr_matrix:
    return 1/2*(matrix + matrix.T)


def compute_horizontal_diffusion_laplacian(diffusion_param_base, base_dist_mat: csr_matrix, fiber_dist_mat, kernel_func_base, kernel_func_fiber, diffusion_param_fiber, block_indices) -> csr_matrix:  # type: ignore
    #np.exp(base_dist_mat)
    #base_dist_mat = base_dist_mat.toarray()
    #fiber_dist_mat = fiber_dist_mat.toarray()
    #print(fiber_dist_mat.shape)
    #base_diffusion_matrix = np.exp((-base_dist_mat ** 2 / diffusion_param_base ** 2)) #kernel_func_base(base_dist_mat, diffusion_param_base)
    base_diffusion_matrix = np.exp((-base_dist_mat ** 2 / diffusion_param_base ** 2)) #kernel_func_base(base_dist_mat, diffusion_param_base)

    fiber_diffusion_matrix = fiber_dist_mat.copy()
    fiber_diffusion_matrix.data = np.exp((-fiber_dist_mat ** 2 / diffusion_param_fiber ** 2).data)
    #fiber_diffusion_matrix = np.exp((-fiber_dist_mat ** 2 / diffusion_param_fiber ** 2))
    #print(fiber_diffusion_matrix.shape)

    #print(fiber_diffusion_matrix.shape)
    #kernel_func_fiber(fiber_dist_mat, diffusion_param_fiber)
    for i in range(len(block_indices)-1):
        for j in range(i+1, len(block_indices)-1):
            fiber_diffusion_matrix[block_indices[i]:block_indices[i+1], block_indices[j]:block_indices[j+1]] *= base_diffusion_matrix[i, j]
            #if base_diffusion_matrix[i, j] == 0.0:
            #   var = 1.0 
            #else: 
            #    var *= base_diffusion_matrix[i, j]

    symmetric_fiber_diffusion_matrix = symmetrize(fiber_diffusion_matrix)
    print(symmetric_fiber_diffusion_matrix.shape)
    
    # normalize it so that each row/ column will sum up to 1.
    horizontal_diffusion_laplacian = symmetric_fiber_diffusion_matrix / np.sum(symmetric_fiber_diffusion_matrix, axis=1)
    print(horizontal_diffusion_laplacian.shape)


    return horizontal_diffusion_laplacian


def eigendecomposition(horizontal_diffusion_laplacian: csr_matrix, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    n = horizontal_diffusion_laplacian.shape[0]
    #eigen_range = [max(0, n-num_return_eigen_vec), n-1]
    
    #eigvals, eigvecs = scipy.linalg.eig(horizontal_diffusion_laplacian, subset_by_index=[n-num_return_eigen_vec, n-1])
    #eigvals, eigvecs = scipy.sparse.linalg.eigsh(horizontal_diffusion_laplacian)
    #print(eigvals, eigvecs)

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(horizontal_diffusion_laplacian, k=num_return_eigen_vec, which="LM")
    print(eigvals)
    # Flip to descending order
    #eigvals = eigvals[::-1]
    #eigvecs = eigvecs[:, ::-1]
        
    return (eigvals, eigvecs)


def cumulative_indices(data_samples: list[np.ndarray]) -> np.ndarray:
    return np.insert(np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 0, 0)

# TODO: maybe add path argument, depending on how it is gonna be used
def load_maps(map_path: str) -> dict:
    with open("../data/mappings.pkl", "rb") as f:
        loaded_mappings = pickle.load(f)
    return loaded_mappings


def load_data_samples(data_samples_path: str, subsample_mapping: float) -> list[np.ndarray]:
    names = loadmat("../data/names.mat")["taxa_code"]
    #print(names)
    #ply_paths = glob.glob(os.path.join(data_samples_path, "*.ply"))
    ply_paths = ["../data/ply/" + name[0] + ".ply" for name in names[0]]
    #print(ply_paths)
    data_samples = [trimesh.load(path).vertices for path in ply_paths[:2]]
    return data_samples


def HDM(
    data_samples_path: str,
    map_path: str,
    sparsity_param_base: float,
    sparsity_param_fiber: float,
    kernel_func_base: Callable[[np.ndarray, np.ndarray], np.float32],
    kernel_func_fiber: Callable[[np.ndarray, np.ndarray], np.float32],
    num_return_eigen_vec: int,
    subsample_mapping: float,
) -> tuple:
    """
    Calculates HDM

    Attributes
    ----------
    data_samples_path : str
        Data samples folder path. Each data sample must be a n x 3 matrix.
        File must be .npy file.
    map_path : str
        Map location path. File must be .pkl file.
    sparsity_param_base : float
        Base kernel sparsity parameter.
    sparsity_param_fiber : float
        Fiber kernel sparsity parameter.

    Notes
    -----

    # TODO: add notes

    Examples
    --------

    # TODO: add examples

    """
    data_samples = load_data_samples(data_samples_path, subsample_mapping)
    maps = load_maps(map_path)
    base_dist_mat = compute_base_dist(data_samples, kernel_func_base, sparsity_param_base)
    block_indices = cumulative_indices(data_samples)
    fiber_dist_mat = compute_fiber_dist(base_dist_mat, sparsity_param_fiber, maps, kernel_func_fiber, data_samples, block_indices)
    diffusion_param_base = 1e-4#compute_base_epsilon(base_dist_mat)
    diffusion_param_fiber = 1e-4 #compute_fiber_epsilon(fiber_dist_mat)
    horizontal_diffusion_laplacian = compute_horizontal_diffusion_laplacian(
        diffusion_param_base,
        base_dist_mat,
        fiber_dist_mat,
        kernel_func_base,
        kernel_func_fiber,
        diffusion_param_fiber,
        block_indices,
    )
    eigenvalues, eigenvectors = eigendecomposition(horizontal_diffusion_laplacian, num_return_eigen_vec)
    return eigenvalues, eigenvectors


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    eigvals, eigvecs = HDM(data_samples_path="../data/ply/", 
        map_path="../data/mappings.pkl", 
        sparsity_param_base=0.4, 
        sparsity_param_fiber=4,
        kernel_func_base=lambda x, y: np.linalg.norm(x - y),
        kernel_func_fiber=lambda x, y: np.linalg.norm(x - y),
        num_return_eigen_vec=10,
        subsample_mapping=1.0)
    #print(eigvals)
    #print(eigvecs)
    coords = eigvecs[:, 1:4] * np.sqrt(eigvals[1:4])
    #print(coords)
    visualize(coords)
