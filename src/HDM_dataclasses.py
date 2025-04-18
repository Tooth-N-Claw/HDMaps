from dataclasses import dataclass
import numpy as np


@dataclass
class HDMData:
    """State and intermediate data for Horizontal Diffusion Maps algorithm."""
    data_samples: any = None
    maps: any = None
    base_dist: any = None
    cumulative_block_indices: any = None
    num_data_samples: int = 0

    
@dataclass
class HDMConfig:
    """Configuration parameters for Horizontal Diffusion Maps algorithm."""
    num_neighbors: int = 4
    base_epsilon: float = 0.04
    fiber_epsilon: float = 0.08
    num_eigenvectors: int = 4
    base_dist_func: str = "fro"
    device: str | None = None
    calculate_fiber_kernel: bool = True