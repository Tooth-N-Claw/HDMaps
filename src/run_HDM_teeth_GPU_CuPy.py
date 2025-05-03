




from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from tqdm import tqdm
import trimesh
from HDM import HDM
from scipy.io import loadmat

from utils.visualize import visualize

def load_maps(data_samples_path: str):
    maps = loadmat(data_samples_path)["softMapMatrix"]
    return np.array(maps)


def _load_single_sample(data_samples_path, name_tuple):
    name = name_tuple[0]
    path = os.path.join(data_samples_path, "ReparametrizedOFF", f"{name}.off")
    vertices = trimesh.load(path).vertices
    return np.array(vertices)

def load_data_samples(data_samples_path, max_workers=None):
    names_path = os.path.join(data_samples_path, "Names.mat")
    names = loadmat(names_path)["Names"]
    
    data_samples = []
    
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(_load_single_sample, data_samples_path, name) for name in names[0]]
        
        for future in tqdm(futures, desc="Loading data samples"):
            result = future.result()
            if result is not None:
                data_samples.append(result)
    
    return np.array(data_samples)

data_samples = load_data_samples("platyrrhine")
maps = load_maps("platyrrhine/softMapMatrix.mat")
base_dist = loadmat("platyrrhine/FinalDists.mat")["dists"]

points = HDM(
    data_samples=data_samples,
    maps=maps,
    base_dist=base_dist,
    num_neighbors=4,
    base_epsilon=0.04,
    num_eigenvectors=4,
    subsample_mapping=0.1,
    calculate_fiber_kernel=False,
    backend="gpu_cupy",
)

visualize(points)