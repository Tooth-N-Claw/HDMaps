import jax.numpy as jnp
import numpy as np
from typing import Optional
from scipy.sparse import coo_matrix

from backend import run_hdm
from HDM.utils.containers import HDMConfig, HDMData, JaxCoo

    
def cumulative_indices(data_samples: list) -> np.ndarray:
    """Calculate cumulative indices for data samples."""
    return np.insert(
        np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 
        0, 0
    )


# def run_HDM(backend, hdm_config, hdm_data):
#     match backend:
#         case "cpu":
#             print("Running HDM on CPU")
#             from HDM.backends.HDM_CPU import run_hdm_cpu 
#             return run_hdm_cpu(hdm_config, hdm_data)
#         case "gpu_pytorch":
#             from HDM.backends.HDM_GPU_PyTorch import run_hdm_gpu # import here to avoid torch being loaded unnecessarily, meaning you can run the code without having torch installed
#             return run_hdm_gpu(hdm_config, hdm_data)
#         case "gpu_pytorch_cupy":
#             # This is a speed up for pytorch as it does not have fast eigendecomposition yet for sparse matrix, so we use cupy instead. Though it has a overhead of converting the sparse matrix to cupy and back again
#             hdm_config.use_cupy = True
#             from HDM.backends.HDM_GPU_PyTorch import run_hdm_gpu
#             return run_hdm_gpu(hdm_config, hdm_data)
#         case "gpu_cupy":
#             from HDM.backends.HDM_GPU_CuPy import run_hdm_cupy
#             return run_hdm_cupy(hdm_config, hdm_data)
        
          

def HDM(
    data_samples: list[np.ndarray],
    base_epsilon: float = 0.04,
    fiber_epsilon: float = 0.08,
    num_eigenvectors: int = 4,
    device: str | None = "CPU",
    base_dist_func: str = "Frobenius",
    fiber_dist_func: str = "Euclidean",
    base_sparsity: float = 0.08,
    fiber_sparsity: float = 0.08,
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
    ):

        
    cumulative_block_indices = cumulative_indices(data_samples)


    if base_kernel is not None:
        base_kernel = JaxCoo.from_scipy(base_kernel)

    if fiber_kernel is not None:
        fiber_kernel = JaxCoo.from_scipy(fiber_kernel)
        
    if base_distances is not None:
        base_distances = JaxCoo.from_scipy(base_distances)

    if fiber_distances not None:
        fiber_distances = JaxCoo.from_scipy(fiber_distances)

    hdm_config = HDMConfig(
        base_epsilon,
        fiber_epsilon,
        num_eigenvectors,
        device,
    )

    hdm_data = HDMData(
        data_samples = data_samples,
        base_distances = base_dist,
    )
    
    diffusion_coords = run_hdm(backend, hdm_config, hdm_data)
