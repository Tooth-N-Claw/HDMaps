import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cusplinalg
from tqdm import tqdm
from HDM_dataclasses import HDMConfig, HDMData


def symmetrize(matrix):
    """Symmetrize a matrix."""
    return 0.5 * (matrix + matrix.T)




def compute_base_kernel(hdm_config: HDMConfig, hdm_data: HDMData) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute the base kernel for diffusion maps using CuPy."""
    if hdm_data.base_dist is None:
        # TODO: Implement base distance computation for CuPy
        raise NotImplementedError("Base distances not provided. This backend does not support computing them yet.")
    else:
        base_dist = cp.array(hdm_data.base_dist)
    
    # Sort distances and get nearest neighbors
    s_dists = cp.sort(base_dist, axis=1)
    row_nns = cp.argsort(base_dist, axis=1)
    
    # Extract k nearest neighbors (excluding self)
    s_dists = s_dists[:, 1:hdm_config.num_neighbors+1]
    row_nns = row_nns[:, 1:hdm_config.num_neighbors+1]
    
    # Create sparse weight matrix
    rows = cp.repeat(cp.arange(hdm_data.num_data_samples).reshape(-1, 1),
                    hdm_config.num_neighbors, axis=1).flatten()
    cols = row_nns.flatten()
    vals = s_dists.flatten()
    
    base_weights = cusp.csr_matrix(
        (vals, (rows, cols)),
        shape=(hdm_data.num_data_samples, hdm_data.num_data_samples)
    )
    
    # Symmetrize the matrix using min operation
    base_weights_array = base_weights.toarray()
    base_weights_array_T = base_weights.T.toarray()
    min_weights = cp.minimum(base_weights_array, base_weights_array_T)
    base_weights = cusp.csr_matrix(min_weights)
    
    # Update the distances with values from the symmetrized matrix
    for j in range(hdm_data.num_data_samples):
        s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray().flatten()
    
    # Apply kernel function
    s_dists = cp.exp(-cp.square(s_dists) / hdm_config.base_epsilon)
    
    return s_dists, row_nns
        
    
def compute_diffusion_matrix(hdm_data: HDMData, hdm_config: HDMConfig, base_diffusion_mat: cp.ndarray, row_nns: cp.ndarray, batch_size: int = 64) -> cusp.csr_matrix:
    """Compute the diffusion matrix for HDM using CuPy with batching."""
    # Initialize lists for constructing sparse matrix
    all_mat_row_idx = []
    all_mat_col_idx = []
    all_vals = []
    
    # Convert cumulative_block_indices to CuPy array if it's not already
    if isinstance(hdm_data.cumulative_block_indices, np.ndarray):
        cumulative_block = cp.array(hdm_data.cumulative_block_indices)
    else:
        cumulative_block = hdm_data.cumulative_block_indices
    
    # Process data in batches
    num_batches = (hdm_data.num_data_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing diffusion matrix (batched)"):
        # Determine batch range
        start_j = batch_idx * batch_size
        end_j = min(start_j + batch_size, hdm_data.num_data_samples)
        
        # Initialize batch lists
        batch_row_idx = []
        batch_col_idx = []
        batch_vals = []
        
        for j in range(start_j, end_j):
            # Find non-zero entries in this row of base_diffusion_mat
            nonzero_nns = cp.nonzero(base_diffusion_mat[j])[0]
            
            if len(nonzero_nns) == 0:
                continue
                
            for nns in nonzero_nns:
                k = int(row_nns[j, nns].get())  # Convert to Python int
                
                # Compute fiber kernel or use provided maps
                if hdm_config.calculate_fiber_kernel:
                    raise NotImplementedError("Fiber kernel computation not implemented for CuPy.")
                else:
                    # Convert numpy sparse matrix to cupy sparse matrix if needed
                    map_matrix = hdm_data.maps[j, k]
                    if not isinstance(map_matrix, cusp.csr_matrix):
                        # Convert scipy sparse to cupy sparse
                        map_matrix = cusp.csr_matrix(
                            (cp.array(map_matrix.data), 
                             cp.array(map_matrix.indices), 
                             cp.array(map_matrix.indptr)),
                            shape=map_matrix.shape
                        )
                    coo = map_matrix.tocoo()
                
                # Forward mapping
                batch_row_idx.append(coo.row + int(cumulative_block[j].get()))
                batch_col_idx.append(coo.col + int(cumulative_block[k].get()))
                batch_vals.append(float(base_diffusion_mat[j, nns].get()) * coo.data)
                
                # Backward mapping (transposed)
                coo_t = map_matrix.transpose().tocoo()
                batch_row_idx.append(coo_t.row + int(cumulative_block[k].get()))
                batch_col_idx.append(coo_t.col + int(cumulative_block[j].get()))
                batch_vals.append(float(base_diffusion_mat[j, nns].get()) * coo_t.data)
        
        # Concatenate batch results if any exist
        if batch_row_idx:
            batch_row_tensor = cp.concatenate(batch_row_idx)
            batch_col_tensor = cp.concatenate(batch_col_idx)
            batch_val_tensor = cp.concatenate(batch_vals)
            
            all_mat_row_idx.append(batch_row_tensor)
            all_mat_col_idx.append(batch_col_tensor)
            all_vals.append(batch_val_tensor)
    
    # Determine proper dimensions for the matrix
    total_size = int(cumulative_block[-1].get())
    
    # Concatenate all batches
    if all_mat_row_idx:
        mat_row_idx = cp.concatenate(all_mat_row_idx)
        mat_col_idx = cp.concatenate(all_mat_col_idx)
        vals = cp.concatenate(all_vals)
        
        # Create sparse matrix
        diffusion_matrix = cusp.csr_matrix(
            (vals, (mat_row_idx, mat_col_idx)),
            shape=(total_size, total_size)
        )
    else:
        # Return empty matrix if no entries
        diffusion_matrix = cusp.csr_matrix((total_size, total_size))
    
    return diffusion_matrix


def create_diagonal_matrix(values):
    """Create a diagonal matrix using CuPy's eye function."""
    # Ensure values is a 1D CuPy array
    if not isinstance(values, cp.ndarray):
        values = cp.array(values)
    
    # Force to 1D if it's not already
    values = values.flatten()
    
    n = len(values)
    
    # Create an identity matrix and replace its diagonal values
    diag_mat = cusp.eye(n, format='csr')
    diag_mat.data = values
    
    return diag_mat


def compute_horizontal_diffusion_laplacian(diffusion_matrix: cusp.csr_matrix) -> tuple[cusp.csr_matrix, cusp.csr_matrix]:
    """Compute the horizontal diffusion Laplacian using CuPy."""
    # Compute row sums and check for zeros
    row_sums = diffusion_matrix.sum(axis=1)
    
    # Convert to a flat 1D array
    if isinstance(row_sums, cusp.spmatrix):
        row_sums = row_sums.toarray()
    
    # Ensure it's a flat 1D array
    row_sums = cp.ravel(row_sums)
    
    if cp.any(row_sums == 0):
        print("Warning: Zero row sums detected in diffusion matrix")
        row_sums[row_sums == 0] = 1e-10
    
    # Create diagonal matrix using eye function
    diag_values = 1.0 / cp.sqrt(row_sums)
    sqrt_diag = create_diagonal_matrix(diag_values)
    
    # Compute normalized Laplacian
    horizontal_diffusion_laplacian = sqrt_diag @ diffusion_matrix @ sqrt_diag
    
    # Ensure symmetry
    horizontal_diffusion_laplacian = symmetrize(horizontal_diffusion_laplacian)
    
    return horizontal_diffusion_laplacian, sqrt_diag


def eigendecomposition(matrix: cusp.csr_matrix, num_eigenvectors: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Perform eigendecomposition on a sparse matrix using CuPy."""
    try:
        eigvals, eigvecs = cusplinalg.eigsh(
            matrix,
            k=num_eigenvectors,
            which="LM",
            maxiter=5000,
            tol=1e-10
        )
        
        # Sort in descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        
        return eigvals, eigvecs
    except Exception as e:
        raise Exception(f"Error in eigendecomposition: {e}")


def run_hdm_cupy(hdm_config: HDMConfig, hdm_data: HDMData) -> np.ndarray:
    """
    Run the complete HDM algorithm using CuPy.
    
    Args:
        hdm_config: Configuration parameters for HDM algorithm.
        hdm_data: Data for HDM algorithm.
        
    Returns:
        The final HDM embedding as a NumPy array.
    """
    try:
        print("Running HDM with CuPy")
        
        base_diffusion_matrix, row_nns = compute_base_kernel(hdm_config, hdm_data)
        print("Computed base kernel")
        
        diffusion_matrix = compute_diffusion_matrix(hdm_data, hdm_config, base_diffusion_matrix, row_nns)
        print("Computed diffusion matrix")
        
        if diffusion_matrix.shape[0] == 0 or diffusion_matrix.nnz == 0:
            raise ValueError("Empty diffusion matrix created. Check previous steps.")
        
        horizontal_diffusion_laplacian, sqrt_diag = compute_horizontal_diffusion_laplacian(diffusion_matrix)
        print("Computed horizontal diffusion Laplacian")
        
        eigvals, eigvecs = eigendecomposition(horizontal_diffusion_laplacian, hdm_config.num_eigenvectors)
        print("Eigendecomposition done")
        
        inf_values = cp.sum(cp.isinf(sqrt_diag.data))
        if inf_values > 0:
            print(f"Warning: {inf_values} infinite values found and replaced with zeros")
            sqrt_diag.data[cp.isinf(sqrt_diag.data)] = 0
        
        bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
        
        sqrt_lambda_values = cp.sqrt(eigvals[1:])
        sqrt_lambda = create_diagonal_matrix(sqrt_lambda_values)
        
        bundle_HDM_full = bundle_HDM @ sqrt_lambda
        
        # Convert result back to NumPy array for compatibility
        return cp.asnumpy(bundle_HDM_full)
        
    except Exception as e:
        print(f"Error in HDM processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise