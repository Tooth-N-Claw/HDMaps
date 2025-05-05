from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from HDM import HDM
from scipy.io import loadmat
from utils.visualize import visualize

def load_maps(data_samples_path: str, device):
    maps = loadmat(data_samples_path)["softMapMatrix"]
    row_len, col_len = maps.shape
    for i in range(row_len):
        for j in range(col_len):
            coo_matrix = maps[i, j].tocoo()
            indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
            values = torch.FloatTensor(coo_matrix.data)  # Use FloatTensor for float64 data

            sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(coo_matrix.shape), device=device)

            maps[i, j] = sparse_tensor.coalesce()
    
    return maps



def _load_single_sample(data_samples_path, name_tuple, device) -> np.ndarray:
    name = name_tuple[0]
    path = os.path.join(data_samples_path, "ReparametrizedOFF", f"{name}.off")
    vertices = trimesh.load(path).vertices
    return torch.tensor(vertices, device=device)


def load_data_samples(data_samples_path, device, max_workers=None):
    names_path = os.path.join(data_samples_path, "Names.mat")
    names = loadmat(names_path)["Names"]
            
    data_samples = []
    
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(_load_single_sample, data_samples_path, name, device) for name in names[0]]
        
        for future in tqdm(futures, desc="Loading data samples"):
            result = future.result()
            if result is not None:
                data_samples.append(result)
    return data_samples


def load_base_dist(data_samples_path: str, device):
    base_dist = loadmat(data_samples_path)["dists"]
    base_dist = torch.tensor(base_dist, device=device)
    return base_dist

# set device, to cuda, mps or cpu on multiple lines
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data_samples = load_data_samples("platyrrhine", device)
print("loading maps")
maps = load_maps("platyrrhine/softMapMatrix.mat", device)
print("loading base dist")
base_dist = load_base_dist("platyrrhine/FinalDists.mat", device)


diffusion_coords = HDM(
        data_samples=data_samples,
        maps=maps,
        base_dist=base_dist,
        num_neighbors=4,
        base_epsilon=0.04,
        num_eigenvectors=4,
        subsample_mapping=0.1,
        calculate_fiber_kernel=False,
        backend="gpu_pytorch_cupy",
        device=device,
    )

visualize(diffusion_coords)
