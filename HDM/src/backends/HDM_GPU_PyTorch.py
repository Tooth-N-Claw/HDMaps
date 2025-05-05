
import torch
import numpy as np
from tqdm import tqdm
from utils.HDM_dataclasses import HDMConfig, HDMData




def symmetrize(matrix):
    """Symmetrize a matrix."""
    return 0.5 * (matrix + matrix.T)



def compute_base_kernel(hdm_config: HDMConfig, hdm_data: HDMData) -> tuple[torch.Tensor, torch.Tensor]:
    
    # TODO: add calculation of base_dist if not provided
    base_dist = hdm_data.base_dist

    N = hdm_data.num_data_samples
    k = hdm_config.num_neighbors

    s_dists_all, row_nns_all = torch.sort(base_dist, dim=1)

    s_dists = s_dists_all[:, 1:k+1]
    row_nns = row_nns_all[:, 1:k+1]


    row_indices_flat = torch.repeat_interleave(torch.arange(N, device=hdm_config.device), repeats=k)
    col_indices_flat = row_nns.flatten()
    values_flat = s_dists.flatten()    

    indices = torch.stack([row_indices_flat, col_indices_flat])
    base_weights_sparse = torch.sparse_coo_tensor(indices, values_flat, size=(N, N))


    base_weights_dense = base_weights_sparse.to_dense()
    base_weights_sym = torch.minimum(base_weights_dense, base_weights_dense.t())

    row_idx_for_gather = torch.arange(N, device=hdm_config.device).unsqueeze(1) 
    s_dists_updated = base_weights_sym[row_idx_for_gather, row_nns] 

    epsilon = hdm_config.base_epsilon
    kernel_vals = torch.exp(-torch.square(s_dists_updated) / epsilon)

    return kernel_vals, row_nns


def compute_diffusion_matrix(hdm_data: HDMData, hdm_config: HDMConfig, base_diffusion_mat: torch.Tensor, row_nns: torch.Tensor, batch_size: int = 64) -> torch.sparse_coo_tensor:    
    all_row_idx = []
    all_col_idx = []
    all_vals = []
    
    # Process data in batches
    num_batches = (hdm_data.num_data_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing diffusion matrix (batched)"):
        # Determine batch range
        start_j = batch_idx * batch_size
        end_j = min(start_j + batch_size, hdm_data.num_data_samples)
        
        # Process each sample in the batch
        batch_rows = []
        batch_cols = []
        batch_values = []
        
        for j in range(start_j, end_j):
            # Find non-zero entries in this row of base_diffusion_mat
            nonzero_nns = torch.nonzero(base_diffusion_mat[j], as_tuple=True)[0]
            
            if len(nonzero_nns) == 0:
                continue
                
            for nns in nonzero_nns:
                k = row_nns[j, nns]
                
                if hdm_config.calculate_fiber_kernel:
                    raise NotImplementedError("Fiber kernel calculation is not implemented for gpu yet.")
                else:
                    map_matrix = hdm_data.maps[j, k]
                
                # Forward mapping
                row, column = map_matrix.indices()
                val_multiplier = base_diffusion_mat[j, nns]
                
                # Add to batch tensors with proper offsets
                batch_rows.append(row + hdm_data.cumulative_block_indices[j])
                batch_cols.append(column + hdm_data.cumulative_block_indices[k])
                batch_values.append(val_multiplier * map_matrix.values())
                
                # Backward mapping (transposed)
                map_matrix_t = map_matrix.t().coalesce()
                row_t, column_t = map_matrix_t.indices()
                
                batch_rows.append(row_t + hdm_data.cumulative_block_indices[k])
                batch_cols.append(column_t + hdm_data.cumulative_block_indices[j])
                batch_values.append(val_multiplier * map_matrix_t.values())
        
        # Concatenate batch results if any exist
        if batch_rows:
            # Concatenate on GPU and append to main lists
            batch_rows_tensor = torch.cat(batch_rows)
            batch_cols_tensor = torch.cat(batch_cols)
            batch_values_tensor = torch.cat(batch_values)
            
            all_row_idx.append(batch_rows_tensor)
            all_col_idx.append(batch_cols_tensor)
            all_vals.append(batch_values_tensor)
    
    # Determine proper dimensions for the matrix
    total_size = hdm_data.cumulative_block_indices[-1]
    
    if all_row_idx:
        # Concatenate all batches
        all_row_idx_tensor = torch.cat(all_row_idx)
        all_col_idx_tensor = torch.cat(all_col_idx)
        all_vals_tensor = torch.cat(all_vals)
        
        # Create PyTorch sparse tensor directly
        indices = torch.stack([all_row_idx_tensor, all_col_idx_tensor])
        shape = torch.Size([total_size, total_size])
        
        diffusion_matrix = torch.sparse_coo_tensor(
            indices=indices, 
            values=all_vals_tensor,
            size=shape,
            device=hdm_config.device
        ).coalesce()
        
    
    return diffusion_matrix


def compute_horizontal_diffusion_laplacian(hdm_config: HDMConfig, diffusion_matrix: torch.sparse_csr_tensor):
    """Compute the horizontal diffusion Laplacian using GPU."""
    # Calculate row sums efficiently for sparse matrix
    row_sums = torch.sparse.sum(diffusion_matrix, dim=1).to_dense()
    
    # Check for zeros
    zero_mask = row_sums == 0
    if torch.any(zero_mask):
        print("Warning: Zero row sums detected in diffusion matrix")
        row_sums[zero_mask] = 1e-10
    
    n = row_sums.size(0)
    inv_sqrt_values = 1.0 / torch.sqrt(row_sums)
    indices = torch.arange(n, device=hdm_config.device)
    indices = torch.stack([indices, indices], dim=0)

    # Create sparse tensor directly
    sqrt_inv_diag_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=inv_sqrt_values,
        size=(n, n),
        device=hdm_config.device
    ).coalesce()
    
    # Compute normalized Laplacian
    horizontal_diffusion_laplacian = sqrt_inv_diag_sparse @ diffusion_matrix @ sqrt_inv_diag_sparse
    
    horizontal_diffusion_laplacian = symmetrize(horizontal_diffusion_laplacian)
    return horizontal_diffusion_laplacian, sqrt_inv_diag_sparse

def pytorch_coo_to_cupy_csr(coo):
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    # Get the CSR components from PyTorch sparse tensor
    coo = coo.cpu()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    shape = coo.shape
    
    # Create the CuPy COO matrix first (safer)
    coo_matrix = cusp.coo_matrix(
        (cp.array(values), (cp.array(indices[0]), cp.array(indices[1]))),
        shape=shape
    )
    
  
    return coo_matrix.tocsr()


def eigendecomposition(hdm_config: HDMConfig, matrix):
    """Perform eigendecomposition on GPU."""
    num_eigenvectors = hdm_config.num_eigenvectors
    
    if hdm_config.use_cupy:
        # Convert PyTorch sparse tensor to CuPy sparse matrix
        matrix = pytorch_coo_to_cupy_csr(matrix.coalesce())

        import cupyx.scipy.sparse.linalg as cusplinalg
        print("Using CuPy for eigendecomposition")
                
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
        
        return torch.tensor(eigvals.get(), device=hdm_config.device), torch.tensor(eigvecs.get(), device=hdm_config.device)
    else:
        
        eigvals, eigvecs = torch.lobpcg(matrix.to_sparse_csr(), k=num_eigenvectors, tol=1e-10, largest=True)
        return eigvals[:num_eigenvectors], eigvecs[:, :num_eigenvectors]


def run_hdm_gpu(hdm_config: HDMConfig, hdm_data: HDMData) -> np.ndarray:
    print("Running HDM on GPU")
    base_diffusion_matrix, row_nns = compute_base_kernel(hdm_config, hdm_data)
    print("Computed base kernel")
    
    diffusion_matrix = compute_diffusion_matrix(hdm_data, hdm_config, base_diffusion_matrix, row_nns)
    print("Computed diffusion matrix")

    horizontal_diffusion_laplacian, sqrt_diag = compute_horizontal_diffusion_laplacian(hdm_config, diffusion_matrix)
    print("Computed horizontal diffusion Laplacian")
    
    eigvals, eigvecs = eigendecomposition(hdm_config, horizontal_diffusion_laplacian)
    print("Eigendecomposition done")
    
    inf_mask = torch.isinf(sqrt_diag.values())
    inf_values = inf_mask.sum().item()
    if inf_values > 0:
        print(f"Warning: {inf_values} infinite values found and replaced with zeros")
        
        # Create a copy of the sparse matrix to avoid in-place modification issues
        indices = sqrt_diag.indices().clone()
        values = sqrt_diag.values().clone()
        values[inf_mask] = 0
        sqrt_diag = torch.sparse_coo_tensor(indices, values, sqrt_diag.size()).coalesce()
        
    # Compute final embedding
    bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
    
    sqrt_lambda = torch.diag(torch.sqrt(eigvals[1:]))
    bundle_HDM_full = bundle_HDM @ sqrt_lambda
    
    # Return as NumPy array for compatibility
    return bundle_HDM_full.cpu().numpy()
