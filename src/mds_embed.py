import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D


def embed_plot(dist_mat):
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(dist_mat)

    # Extract coordinates
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]

    # 3D Plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')

    # Annotate points
    for i in range(len(D)):
        ax.text(x[i], y[i], z[i], f"P{i}", color='black')

    ax.set_title("MDS Embedding in 3D")
    plt.show()
