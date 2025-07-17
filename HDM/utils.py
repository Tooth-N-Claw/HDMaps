import numpy as np
from typing import NamedTuple
from scipy.sparse import csr_matrix


class HDMConfig(NamedTuple):
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    device: str | None = "cpu"
    base_metric: str = "frobenius"
    fiber_metric: str = "euclidean"
    base_sparsity: float = 0.08
    fiber_sparsity: float = 0.08

    
def compute_block_indices(data_samples: list[np.ndarray]) -> np.ndarray:
    """Compute cumulative start indices for a list of data samples."""
    lengths = np.array([len(s) for s in data_samples], dtype=np.int32)
    return np.concatenate([np.array([0], dtype=np.int32), np.cumsum(lengths)])
