import os
from scipy.sparse import bmat, csr_matrix, coo_matrix
from tqdm import tqdm
import trimesh
from HDM import HDM
from scipy.io import loadmat
import numpy as np
import scipy.sparse as sparse

from HDM.visualize import visualize
from HDM.utils import HDMConfig


names = loadmat("platyrrhine/Names.mat")["Names"][0]
mesh_path = "platyrrhine/ReparametrizedOFF/"
meshes = [trimesh.load(mesh_path + name[0] + ".off") for name in names]

data_samples = [mesh.vertices for mesh in meshes]

maps = loadmat("platyrrhine/softMapMatrix.mat")["softMapMatrix"]
num_rows, num_cols = maps.shape
blocks = [
    [csr_matrix(maps[i, j]) for j in range(num_cols)]
    for i in range(num_rows)
]

fiber_kernel = bmat(blocks, format='csr').tocoo()

base_distances = coo_matrix(loadmat("platyrrhine/FinalDists.mat")["dists"]).tocoo()


points = HDM.hdm_embed(
    data_samples = data_samples,
    fiber_kernel = fiber_kernel,
    base_distances = base_distances,
)

visualize(points[:, :3])
