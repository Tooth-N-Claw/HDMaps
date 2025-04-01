import os
import torch
import numpy as np
from tqdm import tqdm
from HDM_dataclasses import HDMConfig, HDMData
from scipy import sparse


def symmetrize(matrix):
    """Symmetrize a matrix."""
    return 0.5 * (matrix + matrix.T)


def compute_base_kernel(hdm_config: HDMConfig, hdm_data: HDMData, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the base kernel for diffusion maps using GPU."""
    try:
        base_dist_gpu = torch.tensor(hdm_data.base_dist, device=device)
        
        s_dists, row_nns = torch.sort(base_dist_gpu, dim=1)
        
        s_dists = s_dists[:, 1:hdm_config.num_neighbors+1]
        row_nns = row_nns[:, 1:hdm_config.num_neighbors+1]
        
        
        # Convert to dense for easier manipulation (will convert back to sparse later)
        base_weights = torch.zeros((hdm_data.num_data_samples, hdm_data.num_data_samples), device=device)
        
        # Fill the matrix using advanced indexing
        rows = torch.arange(hdm_data.base_dist, device=device).unsqueeze(1).expand(-1, hdm_config.num_neighbors)
        base_weights[rows, row_nns] = s_dists
        
        # Symmetrize using minimum operation
        base_weights_T = base_weights.T
        min_weights = torch.minimum(base_weights, base_weights_T)
        
        # Update the distances with values from the symmetrized matrix
        for j in range(hdm_data.base_dist):
            s_dists[j, :] = min_weights[j, row_nns[j, :]]
        
        # Apply kernel function
        s_dists = torch.exp(-torch.square(s_dists) / hdm_config.base_epsilon)
        
        return s_dists, row_nns
    except Exception as e:
        raise Exception(f"Error computing base kernel on GPU: {e}")





def compute_diffusion_matrix(state: HDMData, base_diffusion_mat: torch.Tensor, row_nns: torch.Tensor, device='cuda'):
    """Compute the diffusion matrix for HDM using GPU where possible."""
    # We'll keep this part CPU-based since sparse matrix operations are complex
    # but use GPU for the base operations where beneficial
    
    mat_row_idx = []
    mat_col_idx = []
    vals = []
    
    # Move necessary tensors to CPU for this operation
    base_diffusion_mat_cpu = base_diffusion_mat.cpu().numpy()
    row_nns_cpu = row_nns.cpu().numpy()
    
    for j in tqdm(range(state.num_data_samples), desc="Computing diffusion matrix"):
        for nns in range(base_diffusion_mat_cpu.shape[1]):
            if base_diffusion_mat_cpu[j, nns] == 0:
                continue
            
            k = row_nns_cpu[j, nns]
            map_matrix = state.maps[j, k]
            
            # Forward mapping
            coo = map_matrix.tocoo()
            mat_row_idx.extend(coo.row + state.cumulative_block_indices[j])
            mat_col_idx.extend(coo.col + state.cumulative_block_indices[k])
            vals.extend(base_diffusion_mat_cpu[j, nns] * coo.data)
            
            # Backward mapping (transposed)
            coo = map_matrix.T.tocoo()
            mat_row_idx.extend(coo.row + state.cumulative_block_indices[k])
            mat_col_idx.extend(coo.col + state.cumulative_block_indices[j])
            vals.extend(base_diffusion_mat_cpu[j, nns] * coo.data)
    
    # Determine proper dimensions for the matrix
    total_size = state.cumulative_block_indices[-1]
    
    # Create PyTorch sparse tensor
    indices = torch.LongTensor([mat_row_idx, mat_col_idx]).to(device)
    values = torch.FloatTensor(vals).to(device)
    shape = torch.Size([total_size, total_size])
    
    diffusion_matrix = torch.sparse.FloatTensor(indices, values, shape)
    
    return diffusion_matrix


def compute_horizontal_diffusion_laplacian(diffusion_matrix, device='cuda'):
    """Compute the horizontal diffusion Laplacian using GPU."""
    # Calculate row sums efficiently for sparse matrix
    row_sums = torch.sparse.sum(diffusion_matrix, dim=1).to_dense()
    
    # Check for zeros
    zero_mask = row_sums == 0
    if torch.any(zero_mask):
        print("Warning: Zero row sums detected in diffusion matrix")
        row_sums[zero_mask] = 1e-10
    
    # Compute sqrt inverse
    sqrt_inv_diag = torch.diag(1.0 / torch.sqrt(row_sums))
    
    # Convert to sparse for efficiency (older PyTorch versions)
    sqrt_inv_diag_sparse = sqrt_inv_diag.to_sparse()
    
    # Compute normalized Laplacian using sparse operations
    # For newer PyTorch versions with sparse @ sparse support:
    horizontal_diffusion_laplacian = sqrt_inv_diag_sparse @ diffusion_matrix @ sqrt_inv_diag_sparse
    
    # Ensure symmetry (may need to compute explicitly for sparse tensors)
    # horizontal_diffusion_laplacian = symmetrize(horizontal_diffusion_laplacian)
    
    return horizontal_diffusion_laplacian, sqrt_inv_diag_sparse


def eigendecomposition(matrix, num_eigenvectors, device='cuda'):
    """Perform eigendecomposition on GPU."""
    try:
        # Convert sparse matrix to dense for eigendecomposition
        # (PyTorch's sparse eigendecomposition is limited)
        dense_matrix = matrix.to_dense()
        
        # Compute eigendecomposition - note: torch.linalg.eigh returns values in ascending order
        eigvals, eigvecs = torch.linalg.eigh(dense_matrix)
        
        # Sort in descending order
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        
        # Return top k eigenvectors
        return eigvals[:num_eigenvectors], eigvecs[:, :num_eigenvectors]
    except Exception as e:
        raise Exception(f"Error in GPU eigendecomposition: {e}")


def run_hdm_gpu(hdm_config: HDMConfig, hdm_data: HDMData, device='cuda') -> np.ndarray:
    """
    Run the complete HDM algorithm on GPU.
    
    Args:
        hdm_config: Configuration parameters for HDM algorithm
        hdm_data: Data for HDM algorithm
        device: PyTorch device ('cuda' or 'cuda:0', 'cuda:1', etc.)
        
    Returns:
        The final HDM embedding as NumPy array
    """
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        print(f"Using device: {device}")
        
        base_diffusion_matrix, row_nns = compute_base_kernel(hdm_config, hdm_data, device)
        print("Computed base kernel on GPU")
        
        diffusion_matrix = compute_diffusion_matrix(hdm_data, base_diffusion_matrix, row_nns, device)
        print("Computed diffusion matrix")
        
        if diffusion_matrix.shape[0] == 0 or diffusion_matrix._nnz() == 0:
            raise ValueError("Empty diffusion matrix created. Check previous steps.")
        
        horizontal_diffusion_laplacian, sqrt_diag = compute_horizontal_diffusion_laplacian(diffusion_matrix, device)
        print("Computed horizontal diffusion Laplacian")
        
        eigvals, eigvecs = eigendecomposition(horizontal_diffusion_laplacian, hdm_config.num_eigenvectors, device)
        print("Eigendecomposition done on GPU")
        
        # Handle potential numerical issues in sqrt_diag
        sqrt_diag_dense = sqrt_diag.to_dense()
        inf_values = torch.isinf(sqrt_diag_dense).sum().item()
        if inf_values > 0:
            print(f"Warning: {inf_values} infinite values found and replaced with zeros")
            sqrt_diag_dense[torch.isinf(sqrt_diag_dense)] = 0
            sqrt_diag = sqrt_diag_dense.to_sparse()
        
        # Compute final embedding
        bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
        
        sqrt_lambda = torch.diag(torch.sqrt(eigvals[1:]))
        bundle_HDM_full = bundle_HDM @ sqrt_lambda
        
        # Return as NumPy array for compatibility
        return bundle_HDM_full.cpu().numpy()
        
    except Exception as e:
        print(f"Error in HDM GPU processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise