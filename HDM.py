import numpy as np
from scipy.sparse import csr_matrix
from typing import Callable
import trimesh


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
    base_dist_mat = csr_matrix((k, k), dtype=np.float32)
    for i in range(k): # TODO: could be optimized
        for j in range(k):
            dist = base_norm_function(data_samples[i], data_samples[j])
            if dist <= sparsity_param_base and i != j:
                base_dist_mat[i, j] = dist
    return base_dist_mat


def compute_fiber_dist(base_dist_map: csr_matrix, maps, sparsity_param_fiber: float, fiber_norm_func_name, data_samples: list[np.ndarray]) -> csr_matrix:  # type: ignore
    fiber_dist_block_mat = csr_matrix(())
    pass


def compute_fiber_epsilon(base_dist_map: csr_matrix) -> float:  # type: ignore
    pass


def compute_base_epsilon(fiber_dist_mat: csr_matrix) -> float:  # type: ignore
    pass


def compute_horizontal_diffusion_laplacian(diffusion_param_base, base_dist_mat: csr_matrix, kernel_func_base, kernel_func_fiber, fiber_dist_mat, diffusion_param_fiber) -> csr_matrix:  # type: ignore
    pass


def eigendecomposition(horizontal_diffusion_laplacian: csr_matrix, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    pass


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
