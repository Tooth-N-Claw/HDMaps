




from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import torch
from tqdm import tqdm

from src import HDM


import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def _load_single_sample(data_samples_path, name_tuple) -> np.ndarray:
    name = name_tuple[0]
    path = os.path.join(data_samples_path, "ReparametrizedOFF", f"{name}.off")
    try:
        vertices = np.load(path)
        return vertices
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

def load_data_samples(data_samples_path, max_workers=1, device='cuda'):
    """
    Load data samples and convert to PyTorch tensors.
    Returns a list of tensors since samples may have different sizes.
    """
    try:
        names = os.listdir(data_samples_path)
        tensor_samples = []
        
        with ProcessPoolExecutor(max_workers) as executor:
            futures = [executor.submit(_load_single_sample, data_samples_path, name) for name in names[0]]
            
            for future in tqdm(futures, desc="Loading data samples"):
                result = future.result()
                if result is not None:
                    # Convert each sample to a tensor individually
                    tensor_samples.append(torch.tensor(result, dtype=torch.float32, device=device))
        
        # Return list of tensors since samples may have different sizes
        return tensor_samples
            
    except Exception as e:
        raise Exception(f"Error loading data samples: {e}")

data_samples = load_data_samples('data/v3 Landmarks_and_centroids and intersection_1500/Landmarks')




diffusion_coords = HDM(
        data_samples=data_samples,
        maps=None,
        base_dist_path=None,
        num_neighbors=4,
        base_epsilon=0.04,
        num_eigenvectors=10,
        subsample_mapping=0.1,
    )
