import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from HDM import hdm_embed, HDMConfig, embed_vs_actual

directory_path = 'example-data/wing/'
files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
files = files[:50]
data_samples = [np.loadtxt(os.path.join(directory_path, file), delimiter=',') for file in files]

sample_length = len(data_samples[0])
# print(sample_length)
num_samples = len(data_samples)
# num_samples = 4

config = HDMConfig(
    base_epsilon = 0.004,
    fiber_epsilon = 0.0006,
    base_sparsity = 0.08,
    base_knn = None,
    fiber_sparsity = 0.08,
    fiber_knn = None,

)

diffusion_coords = hdm_embed(
    data_samples = data_samples,
    config = config
)

embed_vs_actual(diffusion_coords, data_samples, num_samples=4)
