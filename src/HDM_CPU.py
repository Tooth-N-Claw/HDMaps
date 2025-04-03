import os
import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.io import loadmat
from tqdm import tqdm
from HDM_dataclasses import HDMConfig, HDMData
from scipy.spatial import distance_matrix

def symmetrize(matrix):
    """Symmetrize a matrix."""
    return 0.5 * (matrix + matrix.T)



def calculate_base_dist(hdm_data: HDMData) -> np.ndarray:
    matrix_array = hdm_data.data_samples
    n = len(hdm_data.data_samples)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        distance_matrix[i, :] = np.linalg.norm(matrix_array - matrix_array[i], axis=(1, 2), ord='fro')

    return distance_matrix



def compute_base_kernel(hdm_config: HDMConfig, hdm_data: HDMData) -> tuple[np.ndarray, np.ndarray]:
    """Compute the base kernel for diffusion maps."""
    try:
        if HDMData.base_dist is None:
            base_dist = calculate_base_dist(hdm_data)
        else:
            base_dist = HDMData.base_dist
            
            
        s_dists = np.sort(base_dist, axis=1)
        row_nns = np.argsort(base_dist, axis=1)
        
        s_dists = s_dists[:, 1:hdm_config.num_neighbors+1]
        row_nns = row_nns[:, 1:hdm_config.num_neighbors+1]
        
        rows = np.repeat(np.arange(hdm_data.num_data_samples).reshape(-1, 1), 
                        hdm_config.num_neighbors, axis=1).flatten()
        cols = row_nns.flatten()
        vals = s_dists.flatten()
        
        base_weights = sparse.csr_matrix(
            (vals, (rows, cols)), 
            shape=(hdm_data.num_data_samples, hdm_data.num_data_samples)
        )
        
        # Symmetrize the matrix using min operation
        base_weights_array = base_weights.toarray()
        base_weights_array_T = base_weights.T.toarray()
        min_weights = np.minimum(base_weights_array, base_weights_array_T)
        base_weights = sparse.csr_matrix(min_weights)
        
        # Update the distances with values from the symmetrized matrix
        for j in range(hdm_data.num_data_samples):
            s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray().flatten()
        
        # Apply kernel function
        s_dists = np.exp(-np.square(s_dists) / hdm_config.base_epsilon)
        
        return s_dists, row_nns
    except Exception as e:
        raise Exception(f"Error computing base kernel: {e}")

def compute_fiber_dist():
    pass

def compute_diffusion_matrix(state: HDMData, base_diffusion_mat: np.ndarray, row_nns: np.ndarray) -> sparse.csr_matrix:
    """Compute the diffusion matrix for HDM."""
    mat_row_idx = []
    mat_col_idx = []
    vals = []
    for j in tqdm(range(state.num_data_samples), desc="Computing diffusion matrix"):
        for nns in range(base_diffusion_mat.shape[1]):
            if base_diffusion_mat[j, nns] == 0:
                continue
            
            k = row_nns[j, nns]

            coo = scipy.sparse.csr_matrix(distance_matrix(state.data_samples[j], state.data_samples[k])).tocoo()


            """ Code for using maps """
            #map_matrix = state.maps[j, k]
            
            # Forward mapping
            #coo = map_matrix.tocoo()
            mat_row_idx.extend(coo.row + state.cumulative_block_indices[j])
            mat_col_idx.extend(coo.col + state.cumulative_block_indices[k])
            vals.extend(base_diffusion_mat[j, nns] * coo.data)
            
            # Backward mapping (transposed)
            #coo = map_matrix.T.tocoo()
            mat_row_idx.extend(coo.row + state.cumulative_block_indices[k])
            mat_col_idx.extend(coo.col + state.cumulative_block_indices[j])
            vals.extend(base_diffusion_mat[j, nns] * coo.data)
    
    # Determine proper dimensions for the matrix
    total_size = state.cumulative_block_indices[-1]
    
    diffusion_matrix = sparse.csr_matrix(
        (vals, (mat_row_idx, mat_col_idx)), 
        shape=(total_size, total_size)
    )
    
    return diffusion_matrix


def compute_horizontal_diffusion_laplacian(diffusion_matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Compute the horizontal diffusion Laplacian."""
    # Compute row sums and check for zeros
    print((diffusion_matrix.data < 0).any())
    row_sums = np.sum(diffusion_matrix, axis=1).A1
    if np.any(row_sums == 0):
        print("Warning: Zero row sums detected in diffusion matrix")
        row_sums[row_sums == 0] = 1e-10
    
    # Create diagonal matrix of inverse sqrt of row sums
    sqrt_diag = sparse.diags(1.0 / np.sqrt(row_sums), 0)
    
    # Compute normalized Laplacian

    # horizontal_diffusion_laplacian = sparse.eye(diffusion_matrix.shape[0]) - sqrt_diag @ diffusion_matrix @ sqrt_diag
    horizontal_diffusion_laplacian = sqrt_diag @ diffusion_matrix @ sqrt_diag
    
    # Ensure symmetry
    horizontal_diffusion_laplacian = symmetrize(horizontal_diffusion_laplacian)
    # print((horizontal_diffusion_laplacian.data < 0).any())
    
    return horizontal_diffusion_laplacian, sqrt_diag


def eigendecomposition(matrix: sparse.csr_matrix, num_eigenvectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigendecomposition on a sparse matrix."""
    try:
        eigvals, eigvecs = sparse.linalg.eigsh(
            matrix, 
            k=num_eigenvectors, 
            which="LM", 
            # maxiter=5000, 
            # tol=1e-10
        )
        
        # Sort in descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        
        return eigvals, eigvecs
    except Exception as e:
        raise Exception(f"Error in eigendecomposition: {e}")


def run_hdm_cpu(hdm_config: HDMConfig, hdm_data: HDMData) -> np.ndarray:
    """
    Run the complete HDM algorithm.
    
    Args:
        config: Configuration parameters for HDM algorithm. If None, default values are used.
        
    Returns:
        The final HDM embedding
    """
    try:
        
        base_diffusion_matrix, row_nns = compute_base_kernel(hdm_config, hdm_data)
        print("Computed base kernel")
        
        diffusion_matrix = compute_diffusion_matrix(hdm_data, base_diffusion_matrix, row_nns)
        print("Computed diffusion matrix")
        
        if diffusion_matrix.shape[0] == 0 or diffusion_matrix.nnz == 0:
            raise ValueError("Empty diffusion matrix created. Check previous steps.")
        
        horizontal_diffusion_laplacian, sqrt_diag = compute_horizontal_diffusion_laplacian(diffusion_matrix)
        # print(np.allclose(horizontal_diffusion_laplacian, horizontal_diffusion_laplacian.T))
        print("Computed horizontal diffusion Laplacian")
        # print(sqrt_diag)
        
        eigvals, eigvecs = eigendecomposition(horizontal_diffusion_laplacian, hdm_config.num_eigenvectors)
        print("Eigendecomposition done")
        # print(eigvals)
         
        # Handle potential numerical issues
        inf_values = np.sum(np.isinf(sqrt_diag.data))
        if inf_values > 0:
            print(f"Warning: {inf_values} infinite values found and replaced with zeros")
            sqrt_diag.data[np.isinf(sqrt_diag.data)] = 0
        
        bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
        # eigvals = np.maximum(eigvals, 1e-12)  # Clip negative values to near-zero
        sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
        sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
        bundle_HDM_full = bundle_HDM @ sqrt_lambda
        
        return bundle_HDM_full
        
    except Exception as e:
        print(f"Error in HDM processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
