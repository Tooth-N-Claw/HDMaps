import pickle
from scipy.io import loadmat
from scipy.sparse import block_array, linalg, diags
from visualize import visualize
import numpy as np
import trimesh


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 1/2*(matrix + matrix.T)


def load_maps(map_path: str) -> dict:
    with open(map_path, "rb") as f:
        loaded_mappings = pickle.load(f)
    return loaded_mappings


def load_data_samples() -> list[np.ndarray]:
    names = loadmat("../data/names.mat")["taxa_code"]
    ply_paths = ["../data/ply/" + name[0] + ".ply" for name in names[0]]
    data_samples = [trimesh.load(path).vertices for path in ply_paths[:10]]
    return data_samples


def compute_base_dist():
    base_dist_mat = loadmat("../data/cPMSTDistMatrix.mat")["ImprDistMatrix"]
    base_dist_mat = base_dist_mat - np.diag(np.diag(base_dist_mat))
    return base_dist_mat


def compute_diffusion_matrix(base_dist_mat, num_data_samples: int, maps: dict):
    blocks = []
    for i in range(num_data_samples):
        block_row = []
        for j in range(num_data_samples):
            block_row.append(maps[(i, j)] * base_dist_mat[i, j])
            #print(maps[(i, j)])
        blocks.append(block_row) 

    fibre_block_dist = block_array(blocks).tocsr()

    return fibre_block_dist


def compute_horizontal_diffusion_laplacian(diffusion_matrix) -> np.ndarray:
    symmetric_diffusion_matrix = symmetrize(diffusion_matrix)
    sqrt_diag = diags(1 / np.sqrt(symmetric_diffusion_matrix.sum(axis=1)))
    horizontal_diffusion_laplacian = sqrt_diag @ symmetric_diffusion_matrix @ sqrt_diag
    print(horizontal_diffusion_laplacian.shape)

    return horizontal_diffusion_laplacian


def eigendecomposition(horizontal_diffusion_laplacian, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    eigvals, eigvecs = linalg.eigsh(horizontal_diffusion_laplacian, k=num_return_eigen_vec, which="LM")

    return (eigvals, eigvecs)


def HDM():    
    data_samples = load_data_samples()
    print("loaded data samples")
    num_data_samples = len(data_samples)
    maps = load_maps("../data/mappings.pkl")
    print("loaded maps")
    base_distance_matrix = compute_base_dist()
    print("loaded base dist")
    base_diffusion_matrix = np.exp(-(base_distance_matrix ** 2) / 0.004)
    print("constructed base diffusion matrix")
    diffusion_matrix = compute_diffusion_matrix(base_diffusion_matrix, num_data_samples, maps)
    print("diffusion matrix computed")
    horizontal_diffusion_laplacian = compute_horizontal_diffusion_laplacian(diffusion_matrix)
    print("horizontal_laplacian constructed")
    eigvals, eigvecs = eigendecomposition(horizontal_diffusion_laplacian, 4)
    coords = eigvecs[:, 1:4] * np.sqrt(eigvals[1:4])
    visualize(coords)


if __name__ == "__main__":
    HDM()