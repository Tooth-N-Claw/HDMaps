from HDM import HDM
from mds_embed import embed_plot
from scipy.spatial.distance import pdist, squareform
from visualize import plot_embedding_3d
from scipy.spatial import distance_matrix
import numpy as np
import os

TOTAL_SUB_SAMPLES = 300

def take_samples(txt_files, directory_path):
    data_samples = []
    metadata = []
    for filename in txt_files:
        input_file_path = os.path.join(directory_path, filename) 
        try:
            matrix = np.loadtxt(input_file_path, delimiter=',')
            data_samples.append(matrix)
            meta = parse_metadata(filename)
            for i in range(6):
                metadata.append(meta)
        except Exception as e:
            print(f"Error loading {input_file_path}: {e}")
            exit(1)    

    data_samples = [mat[:6] for mat in data_samples] 
    return data_samples, metadata 
  
def parse_metadata(file_name):
    meta = {}
    parts = file_name.split("_")
    meta['group'] = parts[0]
    meta['species'] = parts[1]
    meta['temp'] = parts[2]
    meta['sex'] = parts[3]
    meta['id'] = parts[4]
    meta['side'] = parts[5].split(".")[0]  # Remove file extension
    return meta  

def random_subsamples(txt_files):
    species = {f.split('_')[1] for f in txt_files}

    random_files = []
    for s in species:
        files = [f for f in txt_files if f.split('_')[1] == s]
        random_files.extend(np.random.choice(files, TOTAL_SUB_SAMPLES // len(species), replace=False))
    return random_files


# directory_path = 'data/v3 Landmarks_and_centroids and intersection_1500/Landmarks'
directory_path = 'data/aligned_landmarks'

txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

random_files = random_subsamples(txt_files)

data_samples, metadata = take_samples(random_files, directory_path)


data_samples = [mat[:6] for mat in data_samples]  


if __name__ == "__main__":
    diffusion_coords = HDM(
        data_samples=data_samples,
        maps=None,
        base_dist_path=None,
        num_neighbors=4,
        base_epsilon=0.04,
        num_eigenvectors=10,
        subsample_mapping=0.1,
    )
    dist_mat = distance_matrix(diffusion_coords, diffusion_coords)
    print("compute dist")
    # compute pairwise distances use frobenius norm
    
    

    # points = embed_plot(dist_mat)
    # plot_embedding_3d(points, metadata, color_key="species")
