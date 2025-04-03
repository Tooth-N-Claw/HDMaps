import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

from visualize import shira_visualize, shire_visualize_3d, visualize


def embed_plot(dist_mat, fig_folder, data_samples_types):
    
    # 3D MDS
    mds = MDS(n_components=3, dissimilarity='precomputed')
    embedding = mds.fit_transform(dist_mat)
    print(f"MDS Stress: {mds.stress_:.4f}")

    # Extract coordinates
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]
    points = np.column_stack((x, y, z))
    visualize(points, fig_folder, data_samples_types)
    
    # 2D MDS
    # mds_2d = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    # embedding_2d = mds_2d.fit_transform(dist_mat)
    
    # print(f"2D MDS Stress: {mds_2d.stress_:.4f}")
    
    # # 2D matplotlib visualization
    # shira_visualize("figures/mds", embedding_2d, data_samples_types)
