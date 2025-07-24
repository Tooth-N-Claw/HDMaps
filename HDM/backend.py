from abc import ABC, abstractmethod

from HDM.utils import HDMConfig

class BackendBase(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def compute_joint_kernel(base_kernel, fiber_kernel, block_indices,):
        pass
    
    @staticmethod
    @abstractmethod
    def normalize_kernel(diffusion_matrix):
        pass
    
    @staticmethod
    @abstractmethod
    def spectral_embedding(config, kernel, inv_sqrt_diag,):
        pass


