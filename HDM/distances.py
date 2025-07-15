from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

from .utils import HDMConfig

METRICS = {"frobenius", "euclidean"}


def compute_base_distances(config: HDMConfig, data_samples: list[np.ndarray]) -> csr_matrix:
    print("Assumes all data samples has same shape")
    data = np.array([sample.flatten() for sample in data_samples])
    
    nn = NearestNeighbors(radius=config.base_sparsity, algorithm='ball_tree', metric='euclidean')
    nn.fit(data)
    
    sparse_dist_matrix = nn.radius_neighbors_graph(data, mode='distance')
    
    return sparse_dist_matrix


def compute_fiber_distances(config: HDMConfig, data_samples: list[np.ndarray]) -> csr_matrix:
    data = np.stack(data_samples)
    nn = NearestNeighbors(radius=config.fiber_sparsity, algorithm='ball_tree')
    nn.fit(data)
    sparse_dist_matrix = nn.radius_neighbors_graph(data, mode='distance')

    return sparse_dist_matrix
