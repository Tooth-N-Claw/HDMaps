import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
import matplotlib.pyplot as plt


directory_path = 'data/v3 Landmarks_and_centroids and intersection_1500/Landmarks'
output_directory = 'data/aligned_landmarks'
os.makedirs(output_directory, exist_ok=True)

txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

data_samples = []
file_names = []
for file_name in txt_files:
    input_file_path = os.path.join(directory_path, file_name)

    try:
        sample = np.loadtxt(input_file_path, delimiter=',')  # Load landmark coordinates
        output_file_path = os.path.join(output_directory, file_name)
        sample_top = sample[:6, :]
        data_samples.append(sample_top)  # Extract first 6 points (assuming rows are points, cols are x,y)
        file_names.append(file_name)
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}")
        exit(1)


def center(sample):
    centroid = np.mean(sample, axis=0)
    return sample - centroid


# def rescale(sample):
#     n = len(sample)
#     cov_mat = (1/(n-1))*sample.T@sample
#     eigvals = np.linalg.eigvals(cov_mat)
#     return sample * (1 / np.max(eigvals))


def rescale(sample):
    # Compute the pairwise distances
    from scipy.spatial.distance import pdist

    distances = pdist(sample, metric='euclidean')
    diameter = np.max(distances)  # Maximum pairwise distance

    return sample / diameter if diameter != 0 else sample


def align(anchor, sample):
    R, _ = orthogonal_procrustes(sample, anchor)
    return sample @ R


centered_data_samples = [center(sample) for sample in data_samples]
rescaled_data_samples = [rescale(sample) for sample in centered_data_samples]

anchor = rescaled_data_samples[0]

aligned_data_samples = [anchor] + [align(anchor, sample) for sample in rescaled_data_samples[1:]]



def plot(data_samples1, data_samples2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot first dataset
    for i in range(100):
        pt_size = 5
        axs[0].scatter(data_samples1[i][:, 0], data_samples1[i][:, 1], s=pt_size)
    axs[0].set_title('Landmakrs aligned by aligning all points')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    
    for i in range(100):
        pt_size = 5
        axs[1].scatter(data_samples2[i][:, 0], data_samples2[i][:, 1], s=pt_size)
    axs[1].set_title('Landmarks aligned by aligning landmarks (new alignment)')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

plot(data_samples, aligned_data_samples)


for aligned_sample, file_name in zip(aligned_data_samples, file_names):
    output_file_path = os.path.join(output_directory, file_name)
    np.savetxt(output_file_path, aligned_sample, delimiter=',')
    print(f"Saved: {output_file_path}")
