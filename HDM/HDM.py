from HDM.utils.HDM_dataclasses import HDMConfig, HDMData
import numpy as np




    
def cumulative_indices(data_samples: list) -> np.ndarray:
    """Calculate cumulative indices for data samples."""
    return np.insert(
        np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 
        0, 0
    )


def run_HDM(backend, hdm_config, hdm_data):
    match backend:
        case "cpu":
            print("Running HDM on CPU")
            from HDM.backends.HDM_CPU import run_hdm_cpu 
            return run_hdm_cpu(hdm_config, hdm_data)
        case "gpu_pytorch":
            from HDM.backends.HDM_GPU_PyTorch import run_hdm_gpu # import here to avoid torch being loaded unnecessarily, meaning you can run the code without having torch installed
            return run_hdm_gpu(hdm_config, hdm_data)
        case "gpu_pytorch_cupy":
            # This is a speed up for pytorch as it does not have fast eigendecomposition yet for sparse matrix, so we use cupy instead. Though it has a overhead of converting the sparse matrix to cupy and back again
            hdm_config.use_cupy = True
            from HDM.backends.HDM_GPU_PyTorch import run_hdm_gpu
            return run_hdm_gpu(hdm_config, hdm_data)
        case "gpu_cupy":
            from HDM.backends.HDM_GPU_CuPy import run_hdm_cupy
            return run_hdm_cupy(hdm_config, hdm_data)
        
          

def HDM(
    # sparsity_param_base: float,
    # sparsity_param_fiber: float,
    num_neighbors: int,
    base_epsilon: float,
    #kernel_func_base: Callable[[np.ndarray, np.ndarray], np.float32],
    #kernel_func_fiber: Callable[[np.ndarray, np.ndarray], np.float32],
    num_eigenvectors: int,
    subsample_mapping: float,
    fiber_epsilon: float = None,
    calculate_fiber_kernel: bool = True, # TODO: this argument is only meant to be here temporarily
    base_dist = None, # add type
    data_samples: list[np.ndarray] = None, # add type or at least check that this type hinting is correct
    maps=None,
    device=None, # device is only used when using pytorch
    backend: str = "CPU", 
    ):
    """
    DOCUMENTATION HERE!
    """

        
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
        device=device,
    )

    hdm_data = HDMData(
        data_samples,
        maps,
        base_dist,
        cumulative_block_indices,
        num_data_samples=len(data_samples),
    )
    
    try:
        diffusion_coords = run_HDM(backend, hdm_config, hdm_data)
        return diffusion_coords
    except Exception as e:
        print(f"Error running HDM: {e}")
        import traceback
        traceback.print_exc()
        raise
