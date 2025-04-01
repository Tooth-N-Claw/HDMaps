from dataclasses import dataclass
import numpy as np


@dataclass
class HDMData:
    """State and intermediate data for Horizontal Diffusion Maps algorithm."""
    data_samples: list[np.ndarray] = None
    maps: any = None
    base_dist: any = None
    cumulative_block_indices: np.ndarray = None
    num_data_samples: int = 0
    distance_func: str = "fro"
    
@dataclass
class HDMConfig:
    """Configuration parameters for Horizontal Diffusion Maps algorithm."""
    num_neighbors: int = 4
    base_epsilon: float = 0.04
    num_eigenvectors: int = 4
    device: str | None = None