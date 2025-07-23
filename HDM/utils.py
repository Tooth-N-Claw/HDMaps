import numpy as np
from typing import NamedTuple
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


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
