from HDM import HDM
from visualize import visualize
import numpy as np
import os


directory_path = 'data/v3 Landmarks_and_centroids and intersection_1500/Landmarks'

txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

data_samples = []
data_sample_species = []
for filename in txt_files:
    input_file_path = os.path.join(directory_path, filename) 

    try:
        matrix = np.loadtxt(input_file_path, delimiter=',')
        data_samples.append(matrix)
        species = filename.split('_')[1]
        for i in range(6):
            data_sample_species.append(species)
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}")
        exit(1)    

data_samples = [mat[:6] for mat in data_samples]  
if __name__ == "__main__":
    diffusion_coords = HDM(
        data_samples=data_samples,
        maps=None,
        base_dist_path=None,
        num_neighbors=4,
        base_epsilon=0.04,
        num_eigenvectors=4,
        subsample_mapping=0.1,
    )
    visualize(diffusion_coords, data_sample_species)
