from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from HDM_dataclasses import HDMConfig, HDMData
import numpy as np
from tqdm import tqdm
import trimesh
from HDM_GPU import run_hdm_gpu
from HDM_CPU import run_hdm_cpu
from scipy.io import loadmat


    
def cumulative_indices(data_samples: list) -> np.ndarray:
    """Calculate cumulative indices for data samples."""
    return np.insert(
        np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 
        0, 0
    )


def run_HDM(backend, hdm_config, hdm_data):
    match backend:
        case "CPU":
            return run_hdm_cpu(hdm_config, hdm_data)
        case "GPU":
            return run_hdm_gpu(hdm_config, hdm_data)
        
          

def HDM(
    # sparsity_param_base: float,
    # sparsity_param_fiber: float,
    num_neighbors: int,
    base_epsilon: float,
    fiber_epsilon: float,
    #kernel_func_base: Callable[[np.ndarray, np.ndarray], np.float32],
    #kernel_func_fiber: Callable[[np.ndarray, np.ndarray], np.float32],
    num_eigenvectors: int,
    subsample_mapping: float,
    calculate_fiber_kernel: bool = True, # TODO: this argument is only meant to be here temporarily
    base_dist = None, # add type
    data_samples: list[np.ndarray] = None, # add type or at least check that this type hinting is correct
    maps=None,
    backend: str = "CPU",):
    """
    DOCUMENTATION HERE!
    """
    max_workers = max(1, multiprocessing.cpu_count() - 1)

        
    if data_samples is None:
        raise ValueError("Data_samples is None. Please provide data_samples.")

    if maps is None:
        print("Warning: maps are None. Using identity mapping.")

    cumulative_block_indices = cumulative_indices(data_samples)
    
    hdm_config = HDMConfig(
        num_neighbors,
        base_epsilon,
        fiber_epsilon,
        num_eigenvectors,
        calculate_fiber_kernel = calculate_fiber_kernel,
    )

    hdm_data = HDMData(
        data_samples,
        maps,
        None,
        cumulative_block_indices,
        num_data_samples=len(data_samples),
    )
    
    try:
        diffusion_coords = run_HDM(backend, hdm_config, hdm_data)
        return diffusion_coords
    except Exception as e:
        print(f"Error running HDM: {e}")