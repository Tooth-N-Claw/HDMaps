import numpy as np
from typing import NamedTuple
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse import bmat

class HDMConfig(NamedTuple):
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    device: str | None = "cpu"
    base_metric: str = "frobenius"
    fiber_metric: str = "euclidean"
    base_sparsity: float = None
    base_knn: int = 4
    fiber_sparsity: float = None
    fiber_knn: int = 4

    
def compute_block_indices(data_samples: list[np.ndarray]) -> np.ndarray:
    """Compute cumulative start indices for a list of data samples."""
    lengths = np.array([len(s) for s in data_samples], dtype=np.int32)
    return np.concatenate([np.array([0], dtype=np.int32), np.cumsum(lengths)])


def compute_clusters(hdm_coords: np.ndarray, num_clusters: int, seed=None) -> np.ndarray:
    scaled_coords = StandardScaler().fit_transform(hdm_coords)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    labels = kmeans.fit_predict(scaled_coords)

    return labels

def visualize_by_eigenvectors(mesh, hdm_coords):
    pass

def get_backend(config: HDMConfig):
    """Return the appropriate backend based on the configuration."""
    if config.device == 'cpu':
        from .cpu import CPU
        return CPU()
    elif config.device == 'gpu':
        from .cupy import CuPy
        return CuPy()
    else:
        raise ValueError(f"Unsupported device: {config.device}")
    
    
def compute_fiber_kernel_from_maps(maps):
    num_rows, num_cols = maps.shape

    blocks = [
        [csr_matrix(maps[i, j]) for j in range(num_cols)]
        for i in range(num_rows)
    ]

    fiber_kernel = bmat(blocks, format='csr').tocoo()

    return fiber_kernel
