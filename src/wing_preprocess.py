import os
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

# Load data
directory_path = 'data/v3 Landmarks_and_centroids and intersection_1500/Landmarks'
output_directory = 'data/aligned_landmarks'
os.makedirs(output_directory, exist_ok=True)

txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

data_samples = []
file_names = []
for file_name in txt_files:
    input_file_path = os.path.join(directory_path, file_name)

    try:
        matrix = np.loadtxt(input_file_path, delimiter=',')  # Load landmark coordinates
        data_samples.append(matrix[6:, :])  # Extract first 6 points (assuming rows are points, cols are x,y)
        file_names.append(file_name)
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}")
        exit(1)

# Define anchor sample
anchor = data_samples[0]

# Perform PCA on the anchor
pca_anchor = PCA(n_components=2)
anchor_pca = pca_anchor.fit_transform(anchor)

# Center the anchor points
anchor_pca -= np.mean(anchor_pca, axis=0)

# Find the longest eigenvector for scaling reference
anchor_cov = np.cov(anchor_pca.T)  # Covariance matrix
anchor_eigenvalues, anchor_eigenvectors = np.linalg.eig(anchor_cov)
sorted_indices = np.argsort(anchor_eigenvalues)[::-1]  # Sort in descending order
anchor_main_vector = anchor_eigenvectors[:, sorted_indices[0]]  # Longest eigenvector
anchor_scale = np.linalg.norm(anchor_main_vector)  # Scale = length of this vector

aligned_samples = []
for sample, file_name in zip(data_samples, file_names):
    # Perform PCA
    pca_sample = PCA(n_components=2).fit_transform(sample)

    # Center the sample
    pca_sample -= np.mean(pca_sample, axis=0)

    # Compute eigenvectors for the sample
    sample_cov = np.cov(pca_sample.T)
    sample_eigenvalues, sample_eigenvectors = np.linalg.eig(sample_cov)

    # Ensure consistent eigenvector selection
    sorted_indices = np.argsort(sample_eigenvalues)[::-1]  # Sort in descending order
    main_vector = sample_eigenvectors[:, sorted_indices[0]]
    sample_scale = np.linalg.norm(main_vector)

    # Normalize so that the longest eigenvector has length 1
    normalized_sample = pca_sample / sample_scale  

    # Align using Orthogonal Procrustes Analysis
    R, _ = orthogonal_procrustes(normalized_sample, anchor_pca)
    aligned_sample = normalized_sample @ R  # Apply rotation

    aligned_samples.append(aligned_sample)

    # Save output
    output_file_path = os.path.join(output_directory, file_name)
    np.savetxt(output_file_path, aligned_sample, delimiter=',')
    print(f"Saved: {output_file_path}")

print(file_names[:2])
print(data_samples[:2])
print(aligned_samples[:2])
