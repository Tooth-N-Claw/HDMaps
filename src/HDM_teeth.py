from scipy.io import loadmat
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from HDM import HDM
from tqdm import tqdm
import trimesh
from visualize import visualize



def load_data_sample(data_samples_path, name) -> np.ndarray:
    path = os.path.join(data_samples_path, f"{name}.off")
    vertices = trimesh.load(path).vertices
    return vertices




data_samples_path = "platyrrhine/ReparametrizedOFF"
names_path = "platyrrhine/Names.mat"
map_path = "platyrrhine/softMapMatrix.mat"
base_dist_path = "platyrrhine/FinalDists.mat"

maps = loadmat(map_path)["softMapMatrix"]
names = loadmat(names_path)["Names"]
data_samples = [load_data_sample(data_samples_path, name[0]) for name in names[0]]
base_dist = loadmat(base_dist_path)['dists']
# maps = load_maps(map_path, len(data_samples), len(data_samples))




points = HDM(
    data_samples=data_samples,
    base_dist=base_dist,
    maps=maps,
    fiber_epsilon=0.08,
    # sparsity_param_base=0.04,
    # sparsity_param_fiber=1e-3,
    num_neighbors=4,
    base_epsilon=0.04,
    #kernel_func_base=HDM_CPU.kernel_func_base,
    #kernel_func_fiber=HDM_CPU.kernel_func_fiber,
    num_eigenvectors=4,
    subsample_mapping=0.1,
    calculate_fiber_kernel=False,
)

visualize(points[:, :3])