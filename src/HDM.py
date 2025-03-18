from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from HDM_dataclasses import HDMConfig, HDMData
from visualize import visualize
import numpy as np
from tqdm import tqdm
import trimesh
from HDM_GPU import run_hdm_gpu
from HDM_CPU import run_hdm_cpu
from scipy.io import loadmat


def load_maps(data_samples_path: str):
    try:
        maps = loadmat(data_samples_path)["softMapMatrix"]
        return maps
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find softMapMatrix.mat in {data_samples_path}")
    except Exception as e:
        raise Exception(f"Error loading map matrices: {e}")


def _load_single_sample(data_samples_path, name_tuple):
    name = name_tuple[0]
    path = os.path.join(data_samples_path, "ReparametrizedOFF", f"{name}.off")
    try:
        vertices = trimesh.load(path).vertices
        return vertices
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

def load_data_samples(data_samples_path, max_workers=1):
    try:
        names_path = os.path.join(data_samples_path, "Names.mat")
        names = loadmat(names_path)["Names"]
        
        data_samples = []
        
        # Use ProcessPoolExecutor for I/O bound operations
        with ProcessPoolExecutor(max_workers) as executor:
            futures = [executor.submit(_load_single_sample, data_samples_path, name) for name in names[0]]
            
            for future in tqdm(futures, desc="Loading data samples"):
                result = future.result()
                if result is not None:
                    data_samples.append(result)
        
        return data_samples
    except Exception as e:
        raise Exception(f"Error loading data samples: {e}")
    
    
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
    data_samples_path: str,
    map_path: str,
    base_dist_path: str,
    # sparsity_param_base: float,
    # sparsity_param_fiber: float,
    num_neighbors: int,
    base_epsilon: float,
    #kernel_func_base: Callable[[np.ndarray, np.ndarray], np.float32],
    #kernel_func_fiber: Callable[[np.ndarray, np.ndarray], np.float32],
    num_eigenvectors: int,
    subsample_mapping: float,
    backend: str = "GPU",):
    """
    DOCUMENTATION HERE!
    """
    max_workers = max(1, multiprocessing.cpu_count() - 1)

    base_dist = loadmat(base_dist_path)["dists"]
    data_samples = load_data_samples(data_samples_path, max_workers)
    maps = load_maps(map_path)
    cumulative_block_indices = cumulative_indices(data_samples)
    
    hdm_config = HDMConfig(
        num_neighbors,
        base_epsilon,
        num_eigenvectors,
    )
    
    hdm_data = HDMData(
        data_samples,
        maps,
        base_dist,
        cumulative_block_indices,
        num_data_samples=len(data_samples),
    )
    
    
    try:
        embedding = run_HDM(backend, hdm_config, hdm_data)
        visualize(embedding)
    except Exception as e:
        print(f"Error running HDM: {e}")


if __name__ == "__main__":
    HDM(
        data_samples_path="../platyrrhine",
        map_path="../platyrrhine/softMapMatrix.mat",
        base_dist_path="../platyrrhine/FinalDists.mat",
        # sparsity_param_base=0.04,
        # sparsity_param_fiber=1e-3,
        num_neighbors=4,
        base_epsilon=0.04,
        #kernel_func_base=HDM_CPU.kernel_func_base,
        #kernel_func_fiber=HDM_CPU.kernel_func_fiber,
        num_eigenvectors=4,
        subsample_mapping=0.1,
    )