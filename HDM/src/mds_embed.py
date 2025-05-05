import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

from utils.visualize import plot_embedding_3d


def embed_plot(dist_mat):
    
    # 3D MDS
    mds = MDS(n_components=3, dissimilarity='precomputed')
    embedding = mds.fit_transform(dist_mat)
    print(f"MDS Stress: {mds.stress_:.4f}")

    # Extract coordinates
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]
    points = np.column_stack((x, y, z))
    return points
