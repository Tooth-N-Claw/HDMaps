import pickle
from scipy.io import loadmat
from scipy.sparse import block_array, linalg, spdiags, csr_matrix
import scipy
from visualize import visualize
import numpy as np
import trimesh
import scipy.sparse as sparse

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 1/2*(matrix + matrix.T)


def load_maps() -> dict:
    maps = loadmat("../platyrrhine/softMapMatrix.mat")["softMapMatrix"]
    return maps


def load_data_samples() -> list[np.ndarray]:
    names = loadmat("../platyrrhine/Names.mat")["Names"]
    ply_paths = ["../platyrrhine/ReparametrizedOFF/" + name[0] + ".off" for name in names[0]]
    data_samples = [trimesh.load(path).vertices for path in ply_paths]
    return data_samples


# def compute_base_kernel(num_neighbors, base_epsilon):

#     #base_dist_mat = loadmat("../data/cPMSTDistMatrix.mat")["ImprDistMatrix"]
#     base_dist = loadmat("../platyrrhine/FinalDists.mat")["dists"]
#     base_dist = base_dist - np.diag(np.diag(base_dist))
#     N = base_dist.shape[0]  # number of points/species


#     # Step 1: Sort each row to get the nearest neighbor indices.
#     sorted_idx = np.argsort(base_dist, axis=1)
#     # Exclude the first column (self-distance) and take the next num_neighbors.
#     row_nns = sorted_idx[:, 1:num_neighbors+1]
#     s_dists = np.take_along_axis(base_dist, row_nns, axis=1)

#     row_indices = np.tile(np.arange(N).reshape(-1, 1), (1, num_neighbors))
#     base_weights = csr_matrix((s_dists.flatten(), (row_indices.flatten(), row_nns.flatten())), shape=(N, N))

#     # For sparse matrices, use minimum elementwise comparison
#     base_weights = base_weights.minimum(base_weights.transpose())
#     #print(base_weights.toarray())
#     for j in range(N):
#         s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray()

#     base_weights = csr_matrix((s_dists.flatten(), (row_indices.flatten(), row_nns.flatten())), shape=(N, N))
   
#     base_kernel = np.exp(-(s_dists**2) / base_epsilon)
#     return base_kernel, row_nns


def compute_base_kernel(num_neighbors, base_epsilon):

    #base_dist_mat = loadmat("../data/cPMSTDistMatrix.mat")["ImprDistMatrix"]
    base_dist = loadmat("../platyrrhine/FinalDists.mat")["dists"]
    base_dist = base_dist - np.diag(np.diag(base_dist))
    N = base_dist.shape[0]  # number of points/species

    s_dists = np.sort(base_dist, axis=1)
    row_nns = np.argsort(base_dist, axis=1)

    # Step 1: Sort each row to get the nearest neighbor indices.
    # sorted_idx = np.argsort(base_dist, axis=1)
    # Exclude the first column (self-distance) and take the next num_neighbors.
    # row_nns = sorted_idx[:, 1:num_neighbors+1]
    # s_dists = np.take_along_axis(base_dist, row_nns, axis=1)
    
    s_dists = s_dists[:, 1:num_neighbors+1]
    row_nns = row_nns[:, 1:num_neighbors+1]
    
    
    

    # row_indices = np.tile(np.arange(N).reshape(-1, 1), (1, num_neighbors))
    rows = np.repeat(np.arange(N).reshape(-1, 1), num_neighbors, axis=1).flatten()
    cols = row_nns.flatten()
    vals = s_dists.flatten()
    
    # base_weights = csr_matrix((s_dists.flatten(), (row_indices.flatten(), row_nns.flatten())), shape=(N, N))
    base_weights = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    base_weights_array = base_weights.toarray()
    base_weights_array_T = base_weights.T.toarray()
    min_weights = np.minimum(base_weights_array, base_weights_array_T)
    base_weights = sparse.csr_matrix(min_weights)


    for j in range(N):
        s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray().flatten()

    # Apply exponential transformation
    s_dists = np.exp(-np.square(s_dists) / base_epsilon)

    # # For sparse matrices, use minimum elementwise comparison
    # base_weights = base_weights.minimum(base_weights.transpose())
    # #print(base_weights.toarray())
    # for j in range(N):
    #     s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray()

    # base_weights = csr_matrix((s_dists.flatten(), (row_indices.flatten(), row_nns.flatten())), shape=(N, N))
   
    # base_kernel = np.exp(-(s_dists**2) / base_epsilon)
    return s_dists, row_nns



def cumulative_indices(data_samples: list[np.ndarray]) -> np.ndarray:
    return np.insert(np.cumsum([len(sample) for sample in data_samples], dtype=np.int32), 0, 0)


# def compute_diffusion_matrix(base_diffusion_mat, num_data_samples: int, maps: dict, num_neighbors: int, row_nns, cumulative_block_indicies) -> csr_matrix:
#     mat_row_idx = []
#     mat_col_idx = []
#     vals = []

#     for j in range(num_data_samples):
#         print(f"Teeth {j + 1} of {num_data_samples}")
#         for nns in range(num_neighbors):
#             if base_diffusion_mat[j, nns] == 0:
#                 continue

#             k = row_nns[j, nns]
#             map = maps[j, k]
#             map = map.toarray()
            
#             row_idx, col_idx = map.nonzero()
#             val = map[row_idx, col_idx]

#             mat_row_idx.extend(row_idx + cumulative_block_indicies[j]) 
#             mat_col_idx.extend(col_idx + cumulative_block_indicies[k])
#             vals.extend(base_diffusion_mat[j, nns] * val)
            
#             row_idx, col_idx = map.T.nonzero()
#             mat_row_idx.extend(row_idx + cumulative_block_indicies[k]) 
#             mat_col_idx.extend(col_idx + cumulative_block_indicies[j])
#             vals.extend(base_diffusion_mat[j, nns] * val)
    
    
#     N = cumulative_block_indicies[-1]
#     mat_row_idx = np.array(mat_row_idx)
#     mat_col_idx = np.array(mat_col_idx)
#     vals = np.array(vals)
    
#     diffusion_matrix = csr_matrix((vals, (mat_row_idx, mat_col_idx)), shape=(N, N))

#     return diffusion_matrix



def compute_diffusion_matrix(base_diffusion_mat, num_data_samples: int, maps: dict, num_neighbors: int, row_nns, cumulative_block_indicies) -> csr_matrix:
    mat_row_idx = []
    mat_col_idx = []
    vals = []

    for j in range(num_data_samples):
        print(f"Teeth {j + 1} of {num_data_samples}")
        for nns in range(num_neighbors):
            if base_diffusion_mat[j, nns] == 0:
                continue

            k = row_nns[j, nns]
            map = maps[j, k]
    
            if sparse.issparse(map): # TODO: should either be sparse or dense, not both, so remove one of them
                # For sparse matrix
                coo = map.tocoo()
                row_idx, col_idx, val = coo.row, coo.col, coo.data
            else:
                # For dense matrix
                row_idx, col_idx = np.nonzero(map)
                val = map[row_idx, col_idx]
            

            mat_row_idx.extend(row_idx + cumulative_block_indicies[j]) 
            mat_col_idx.extend(col_idx + cumulative_block_indicies[k])
            vals.extend(base_diffusion_mat[j, nns] * val)
                        
            if sparse.issparse(map):# TODO: should either be sparse or dense, not both, so remove one of them
                coo = map.T.tocoo()
                rowIdx, colIdx, val = coo.row, coo.col, coo.data
            else:
                map_T = map.T
                rowIdx, colIdx = np.nonzero(map_T)
                val = map_T[rowIdx, colIdx]
            
            
            mat_row_idx.extend(row_idx + cumulative_block_indicies[k]) 
            mat_col_idx.extend(col_idx + cumulative_block_indicies[j])
            vals.extend(base_diffusion_mat[j, nns] * val)
    
    
    N = cumulative_block_indicies[-1]
    diffusion_matrix = sparse.csr_matrix((vals, (mat_row_idx, mat_col_idx)), shape=(N, N))
    return diffusion_matrix


def compute_horizontal_diffusion_laplacian(diffusion_matrix) -> tuple[np.ndarray, np.ndarray]:
    # diag_vals = 1 / np.sqrt(diffusion_matrix.sum(axis=0))
    #sqrt_diag = 1.0 / np.sqrt(np.sum(diffusion_matrix, axis=0))
    sqrt_diag = scipy.sparse.diags(1.0 / np.sqrt(np.sum(diffusion_matrix, axis=0).A1), 0)

    horizontal_diffusion_laplacian = sqrt_diag @ diffusion_matrix @ sqrt_diag
    horizontal_diffusion_laplacian = (horizontal_diffusion_laplacian + horizontal_diffusion_laplacian.T) / 2

    return (horizontal_diffusion_laplacian, sqrt_diag)


def eigendecomposition(horizontal_diffusion_laplacian, num_return_eigen_vec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(horizontal_diffusion_laplacian, k=num_return_eigen_vec, which="LM", maxiter=5000, tol=1e-10)
    print(eigvals)
    return (eigvals, eigvecs)


def HDM():    
    num_neighbors = 4
    data_samples = load_data_samples()
    print("loaded data samples")
    num_data_samples = len(data_samples)
    maps = load_maps()
    cumulative_block_indicies = cumulative_indices(data_samples)
    print("loaded maps")
    base_diffusion_matrix, row_nns = compute_base_kernel(num_neighbors, 0.04)
    print("Basedist:")
    #print(base_distance_matrix)
    print("loaded base dist")
    #base_diffusion_matrix = np.exp(-(s_dists ** 2) / 0.04)
    #print(base_diffusion_matrix)
    print("constructed base diffusion matrix")
    diffusion_matrix = compute_diffusion_matrix(base_diffusion_matrix, num_data_samples, maps, num_neighbors, row_nns, cumulative_block_indicies)
    #print(diffusion_matrix)
    print("diffusion matrix computed")
    horizontal_diffusion_laplacian, sqrt_diag = compute_horizontal_diffusion_laplacian(diffusion_matrix)
    print("horizontal_laplacian constructed")
    eigvals, eigvecs = eigendecomposition(horizontal_diffusion_laplacian, 4)
    
    sqrt_diag.data[np.isinf(sqrt_diag.data)] = 0

    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    sqrt_lambda = scipy.sparse.diags(np.sqrt(eigvals[1:]), 0)
    # np.diag(np.sqrt(eigvals[1:]))
    bundle_HDM_full = bundle_HDM @ sqrt_lambda
    #print(bundle_HDM_full)

    visualize(bundle_HDM_full[:, :3])


if __name__ == "__main__":
    HDM()