import sys

import os


from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import trimesh
from HDM import HDM
from scipy.io import loadmat

from tests.utils import load_pikle_file, compare_arrays

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

def load_data_samples(data_samples_path, max_workers=None):
    try:
        names_path = os.path.join(data_samples_path, "Names.mat")
        names = loadmat(names_path)["Names"]
        
        data_samples = []
        
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(_load_single_sample, data_samples_path, name) for name in names[0]]
            
            for future in tqdm(futures, desc="Loading data samples"):
                result = future.result()
                if result is not None:
                    data_samples.append(result)
        
        return data_samples
    except Exception as e:
        raise Exception(f"Error loading data samples: {e}")
    


data_samples = load_data_samples("platyrrhine")
maps = load_maps("platyrrhine/softMapMatrix.mat")
base_dist = loadmat("platyrrhine/FinalDists.mat")["dists"]

teeth_points_test = HDM(
    data_samples=data_samples,
    maps=maps,
    base_dist=base_dist,
    # sparsity_param_base=0.04,
    # sparsity_param_fiber=1e-3,
    num_neighbors=4,
    base_epsilon=0.04,
    #kernel_func_base=HDM_CPU.kernel_func_base,
    #kernel_func_fiber=HDM_CPU.kernel_func_fiber,
    num_eigenvectors=4,
    subsample_mapping=0.1,
    calculate_fiber_kernel=False,
    backend="cpu",
)


teeth_points_target = load_pikle_file("tests/teeth_points.pkl")

assert compare_arrays(teeth_points_test, teeth_points_target), True