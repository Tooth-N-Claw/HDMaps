import numpy as np
from scipy.sparse import coo_matrix
from typing import Callable
import trimesh


def compute_base_dist(data_samples: list[np.ndarray], base_norm_: Callable[np.ndarray, np.ndarray]) -> coo_matrix:  # type: ignore
    """
    >>> compute_base_dist()
    None
    """
    pass


def compute_fiber_dist(base_dist_map: coo_matrix, maps, sparsity_param_fiber: float, fiber_norm_func_name, data_samples: list[np.ndarray]) -> coo_matrix:  # type: ignore
    pass


def compute_fiber_epsilon(base_dist_map: coo_matrix) -> float:  # type: ignore
    pass


def compute_base_epsilon(fiber_dist_mat: coo_matrix) -> float:  # type: ignore
    pass


def compute_horizontal_diffusion_laplacian(diffusion_param_base, base_dist_mat: coo_matrix, kernel_func_base, kernel_func_fiber, fiber_dist_mat, diffusion_param_fiber) -> coo_matrix:  # type: ignore
    pass


def eigendecomposition(horizontal_diffusion_laplacian: coo_matrix, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    pass


def load_maps():  # ignore: type
    pass


def load_data_samples():  # ignore: type
    pass


def HDM(data_samples_path: str, map_path: str, sparsity_param_base: float, sparsity_param_fiber: float):
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
    horizontal_diffusion_laplacian = compute_horizontal_diffusion_laplacian()
    eigenvalues, eigenvectors = eigendecomposition()


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    HDM()
