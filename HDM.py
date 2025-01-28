import numpy as np
from scipy.sparse import csr_matrix
import trimesh
from itertools import combinations
from typing import Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist


# @dataclass
# class Block_tree:
#     self.data: list['block_tree'] | np.ndarray
#     self.shape: Tuple[int, int]
    # self.blocks: Optional[list['block_tree']]
    # self.matrix: Optional[np.ndarray]

# @dataclass
# class Data_tree:
#     data: list['Data_tree'] | np.ndarray


# def collapse_block():
DataTree = list['DataTree'] | np.ndarray


def construct_block_distance_matrix(data_tree1: DataTree, data_tree2: DataTree, funcs: list[Callable[DataTree, np.ndarray]]) -> np.ndarray:
    """
        Both data trees must be of equal depth
        and funcs must be the same length as the depth
    """
    if isinstance(data_tree1, np.ndarray) and isinstance(data_tree2, np.ndarray):
        return funcs[0](data_tree1, data_tree2)

    n = len(data_tree1)
    m = len(data_tree2)

    blocks = [
        [
            funcs[0](data_tree1[i], data_tree2[j]) * construct_block_distance_matrix(data_tree1[i], data_tree2[j], funcs[1:])
            for j in range(m)
        ]
        for i in range(n)
    ]    

    return np.block(blocks)


data_tree = [np.array([[1, 0], [2, 3]]), np.array([[1, 2], [3, 3], [3, 3]])]

def func1(A, B):
    return cdist(A, B)

print(construct_block_distance_matrix(data_tree, data_tree, [func1, func1]))

    
# def func2(A, B):
#     return pdi
    


# def collapse_data_tree(data_tree: Data_tree, funcs: list[Callable]) -> np.ndarray:
#     match data_tree.data:
#         case list() as data_samples if all(isinstance(b, np.ndarray) for b in data_samples):
#             n = len(self.data)

#             dist_matrix = [[distance_matrix(self.data[i], self.data[j]) for j in range(n)] for i in range(n)]
#             # for i in combinations(range(len(self.data))):
                
#                 # , j in combinations(range(len(self.data)):
#                 # distance_matrix(self.data[i], self.data[j])
#         case list() as sub_trees:
#             pass
#         # self.blocks = [collapse_block_tree(b) for b in self.blocks]
#         # self.matrix = np.array(self.blocks)
#     pass





def sparse_distance_matrix(data_samples1, data_samples2, norm_function, sparsity_param):
    m, n = len(data_samples1), len(data_samples2)
    dist_matrix = csr_matrix((m, n), dtype=np.float32)
    for i, j in combinations(range(max(m, n))):
        dist = base_norm_function(data_samples1[i], data_samples2[j])
        if dist <= sparsity_param_base:
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat




def subsample(data_sample: np.ndarray, supsample_mapping: float) -> np.ndarray: # type: ignore
    pass


def compute_base_dist(
    data_samples: list[np.ndarray],
    base_norm_function: Callable[[np.ndarray, np.ndarray], np.float32],
    sparsity_param_base: float = 10,
    subsample_mapping: float = 1.0,
) -> csr_matrix:  # type: ignore
    """
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
    """
    NegativeDistError - if the value returned from the base norm function name is non-positive.
    InputNumOfDistParam - the number of parameters in base norm function name doesn’t match.
    NonBoundedSubSample - subsample mapping is out of the range [0, 1].
    NegativeSparsity - the sparsity param base parameter is negative."""
    if subsample_mapping != 1:
        data_samples = [subsample(data_sample, subsample_mapping) for data_sample in data_samples]
    k = len(data_samples)
    base_dist_mat = sparse_distance_matrix(data_samples, data_samples, base_norm_function, sparsity_param_base)
    return base_dist_mat


def compute_fiber_dist(base_dist_map: csr_matrix, maps, sparsity_param_fiber: float, fiber_norm_func_name, data_samples: list[np.ndarray]) -> csr_matrix:  # type: ignore
    fiber_dist_block_mat = csr_matrix(())
    # blocks = [for ]
    # should return a list of matrices
    # The block matrix should be constructed in the very end
    pass


def compute_fiber_epsilon(base_dist_map: csr_matrix) -> float:  # type: ignore
    pass


def compute_base_epsilon(fiber_dist_mat: csr_matrix) -> float:  # type: ignore
    pass

def _construct_weight_matrix() -> csr_matrix:
    pass 


def compute_horizontal_diffusion_laplacian(diffusion_param_base, base_dist_mat: csr_matrix, kernel_func_base, kernel_func_fiber, fiber_dist_mat, diffusion_param_fiber) -> csr_matrix:  # type: ignore
    diag = np.sum(weight_matrix, axis=1)
    diag_ = np.diag(diag**(1/alpha))

    weight_matrix_alpha = diag_@weight_matrix@diag_
    diag_alpha = np.sum(weight_matrix_alpha, axis=1)
    diag_alpha_ = np.diag(diag_alpha**(1/2))

    graph_laplacian = np.identity(weight_matrix.shape) - diag_alpha_@weight_matrix_alpha@diag_alpha_

    return graph_laplacian


def eigendecomposition(horizontal_diffusion_laplacian: csr_matrix, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    eigvals, eigvecs = np.linalg.eigh(laplacian)

    # Flip order and take first k
    eigvals = eigvals[-embedding_dimension:][::-1]
    eigvecs = eigvecs[:, -embedding_dimension:][:, ::-1]
    
    embedding = eigvecs*(eigvals**t)
    
    return embedding



def load_maps():  # ignore: type
    pass


def load_data_samples():  # ignore: type
    pass


def HDM(
    data_samples_path: str,
    map_path: str,
    sparsity_param_base: float,
    sparsity_param_fiber: float,
    kernel_func_base,
    kernel_func_fiber,
    num_return_eigen_vec,
):
    """
    Calculates HDM

    Attributes
    ----------
    data_samples_path : str
        Data samples folder path. Each data sample must be a n x 3 matrix.
        File must be .npy file.
    map_path : str
        Paths for location of the maps.
        Files must be .npy file.
    sparsity_param_base : float
        Threshold for when a distance should be zero
    sparsity_param_fiber : float
        # TODO: add description

    Notes
    -----

    # TODO: add notes

    Examples
    --------

    # TODO: add examples

    """
    data_samples = load_data_samples()
    maps = load_maps()
    base_dist_mat = compute_base_dist()
    fiber_dist_mat = compute_fiber_dist(base_dist_mat, maps)
    diffusion_param_base = compute_base_epsilon(base_dist_mat)
    diffusion_param_fiber = compute_fiber_epsilon(fiber_dist_mat)
    horizontal_diffusion_laplacian = compute_horizontal_diffusion_laplacian(
        base_dist_mat,
        fiber_dist_mat,
        kernel_func_base,
        kernel_func_fiber,
        diffusion_param_base,
        diffusion_param_fiber,
    )
    eigenvalues, eigenvectors = eigendecomposition()


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    # HDM()
    pass
