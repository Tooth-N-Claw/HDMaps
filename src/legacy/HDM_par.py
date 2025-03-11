import os
import numpy as np
import scipy.sparse as sparse
from scipy.io import loadmat
import trimesh
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

class ParallelHorizontalDiffusionMaps:
    """
    Parallelized implementation of Horizontal Diffusion Maps algorithm.
    """
    
    def __init__(self, data_dir="../platyrrhine/", num_neighbors=4, base_epsilon=0.04, 
                 num_eigenvectors=4, n_jobs=None):
        self.data_dir = data_dir
        self.num_neighbors = num_neighbors
        self.base_epsilon = base_epsilon
        self.num_eigenvectors = num_eigenvectors
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        
        # Will be populated during processing
        self.data_samples = None
        self.maps = None
        self.cumulative_block_indicies = None
        self.num_data_samples = 0
        
    def symmetrize(self, matrix):
        return 0.5 * (matrix + matrix.T)
    
    def load_maps(self):
        try:
            map_path = os.path.join(self.data_dir, "softMapMatrix.mat")
            maps = loadmat(map_path)["softMapMatrix"]
            return maps
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find softMapMatrix.mat in {self.data_dir}")
        except Exception as e:
            raise Exception(f"Error loading map matrices: {e}")
    
    def _load_single_sample(self, name_tuple):
        name = name_tuple[0]
        path = os.path.join(self.data_dir, "ReparametrizedOFF", f"{name}.off")
        try:
            vertices = trimesh.load(path).vertices
            return vertices
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None
    
    def load_data_samples(self):
        try:
            names_path = os.path.join(self.data_dir, "Names.mat")
            names = loadmat(names_path)["Names"]
            
            data_samples = []
            
            # Use ProcessPoolExecutor for I/O bound operations
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._load_single_sample, name) for name in names[0]]
                
                for future in tqdm(futures, desc="Loading data samples"):
                    result = future.result()
                    if result is not None:
                        data_samples.append(result)
            
            return data_samples
        except Exception as e:
            raise Exception(f"Error loading data samples: {e}")
    
    def compute_base_kernel(self):
        try:
            dist_path = os.path.join(self.data_dir, "FinalDists.mat")
            base_dist = loadmat(dist_path)["dists"]
            base_dist = base_dist - np.diag(np.diag(base_dist))
            
            # These operations can be parallelized with NumPy's threading
            # Make sure NumPy is using a parallel BLAS implementation (MKL, OpenBLAS)
            s_dists = np.sort(base_dist, axis=1)
            row_nns = np.argsort(base_dist, axis=1)
            
            # Match the original implementation more closely
            s_dists = s_dists[:, 1:self.num_neighbors+1]
            row_nns = row_nns[:, 1:self.num_neighbors+1]
            
            # Build sparse matrix with proper indexing
            rows = np.repeat(np.arange(self.num_data_samples).reshape(-1, 1), 
                           self.num_neighbors, axis=1).flatten()
            cols = row_nns.flatten()
            vals = s_dists.flatten()
            
            # Create sparse matrix representation
            base_weights = sparse.csr_matrix(
                (vals, (rows, cols)), 
                shape=(self.num_data_samples, self.num_data_samples)
            )
            
            # Symmetrize the matrix using min operation
            base_weights_array = base_weights.toarray()
            base_weights_array_T = base_weights.T.toarray()
            min_weights = np.minimum(base_weights_array, base_weights_array_T)
            base_weights = sparse.csr_matrix(min_weights)
            
            # Parallel update of distances
            def update_row_dists(j):
                return base_weights[j, row_nns[j, :]].toarray().flatten()
            
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                updated_dists = list(executor.map(update_row_dists, range(self.num_data_samples)))
            
            for j, updated_dist in enumerate(updated_dists):
                s_dists[j, :] = updated_dist
            
            # Apply kernel function
            s_dists = np.exp(-np.square(s_dists) / self.base_epsilon)
            
            return s_dists, row_nns
        except Exception as e:
            raise Exception(f"Error computing base kernel: {e}")
    
    def cumulative_indices(self):
        return np.insert(
            np.cumsum([len(sample) for sample in self.data_samples], dtype=np.int32), 
            0, 0
        )
    
    def _process_diffusion_row(self, j, base_diffusion_mat, row_nns):
        """Process a single row of the diffusion matrix computation."""
        local_row_idx = []
        local_col_idx = []
        local_vals = []
        
        for nns in range(self.num_neighbors):
            if base_diffusion_mat[j, nns] == 0:
                continue
            
            k = row_nns[j, nns]
            map_matrix = self.maps[j, k]
            
            # Forward mapping
            coo = map_matrix.tocoo()
            local_row_idx.extend(coo.row + self.cumulative_block_indicies[j])
            local_col_idx.extend(coo.col + self.cumulative_block_indicies[k])
            local_vals.extend(base_diffusion_mat[j, nns] * coo.data)
            
            # Backward mapping (transposed)
            coo = map_matrix.T.tocoo()
            local_row_idx.extend(coo.row + self.cumulative_block_indicies[k])
            local_col_idx.extend(coo.col + self.cumulative_block_indicies[j])
            local_vals.extend(base_diffusion_mat[j, nns] * coo.data)
        
        return local_row_idx, local_col_idx, local_vals
    
    def compute_diffusion_matrix(self, base_diffusion_mat, row_nns):
        # Use ProcessPoolExecutor for CPU-bound operations
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for j in range(self.num_data_samples):
                futures.append(
                    executor.submit(
                        self._process_diffusion_row, 
                        j, base_diffusion_mat, row_nns
                    )
                )
            
            # Collect results
            all_results = []
            for future in tqdm(futures, desc="Computing diffusion matrix"):
                all_results.append(future.result())
        
        # Consolidate results
        mat_row_idx = []
        mat_col_idx = []
        vals = []
        
        for row_idx, col_idx, val in all_results:
            mat_row_idx.extend(row_idx)
            mat_col_idx.extend(col_idx)
            vals.extend(val)
        
        # Determine proper dimensions for the matrix
        total_size = self.cumulative_block_indicies[-1]
        
        diffusion_matrix = sparse.csr_matrix(
            (vals, (mat_row_idx, mat_col_idx)), 
            shape=(total_size, total_size)
        )
        
        return diffusion_matrix
    
    def compute_horizontal_diffusion_laplacian(self, diffusion_matrix):
        # Compute row sums and check for zeros
        row_sums = np.sum(diffusion_matrix, axis=1).A1
        if np.any(row_sums == 0):
            print("Warning: Zero row sums detected in diffusion matrix")
            row_sums[row_sums == 0] = 1e-10
        
        # Create diagonal matrix of inverse sqrt of row sums
        sqrt_diag = sparse.diags(1.0 / np.sqrt(row_sums), 0)
        
        # Compute normalized Laplacian
        horizontal_diffusion_laplacian = sqrt_diag @ diffusion_matrix @ sqrt_diag
        
        # Ensure symmetry
        horizontal_diffusion_laplacian = self.symmetrize(horizontal_diffusion_laplacian)
        
        return horizontal_diffusion_laplacian, sqrt_diag
    
    def eigendecomposition(self, horizontal_diffusion_laplacian):
        try:
            # SciPy's eigsh can leverage parallelism via underlying ARPACK/BLAS
            eigvals, eigvecs = sparse.linalg.eigsh(
                horizontal_diffusion_laplacian, 
                k=self.num_eigenvectors, 
                which="LM", 
                maxiter=5000, 
                tol=1e-10,
                # Use more Lanczos vectors for better convergence
                ncv=min(horizontal_diffusion_laplacian.shape[0], 2*self.num_eigenvectors+1)
            )
            
            # Sort in descending order
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            
            return eigvals, eigvecs
        except Exception as e:
            raise Exception(f"Error in eigendecomposition: {e}")
    
    def run(self):
        """
        Run the complete HDM algorithm with parallelization.
        
        Returns:
            The final HDM embedding
        """
        try:
            # Load data
            self.data_samples = self.load_data_samples()
            print("Loaded data samples")
            
            self.num_data_samples = len(self.data_samples)
            self.cumulative_block_indicies = self.cumulative_indices()
            
            self.maps = self.load_maps()
            print("Loaded maps")
            
            base_diffusion_matrix, row_nns = self.compute_base_kernel()
            print("Computed base kernel")
            
            diffusion_matrix = self.compute_diffusion_matrix(base_diffusion_matrix, row_nns)
            print("Computed diffusion matrix")
            
            if diffusion_matrix.shape[0] == 0 or diffusion_matrix.nnz == 0:
                raise ValueError("Empty diffusion matrix created. Check previous steps.")
            
            horizontal_diffusion_laplacian, sqrt_diag = self.compute_horizontal_diffusion_laplacian(diffusion_matrix)
            print("Computed horizontal diffusion Laplacian")
            
            eigvals, eigvecs = self.eigendecomposition(horizontal_diffusion_laplacian)
            print("Eigendecomposition done")
            
            # Handle potential numerical issues
            inf_values = np.sum(np.isinf(sqrt_diag.data))
            if inf_values > 0:
                print(f"Warning: {inf_values} infinite values found and replaced with zeros")
                sqrt_diag.data[np.isinf(sqrt_diag.data)] = 0
            
            bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
            sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
            bundle_HDM_full = bundle_HDM @ sqrt_lambda
            
            return bundle_HDM_full
            
        except Exception as e:
            print(f"Error in HDM processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Main entry point for running the parallelized HDM algorithm.
    """
    try:
        # Use all available CPUs except one
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"Running with {n_jobs} parallel workers")
        
        hdm = ParallelHorizontalDiffusionMaps(n_jobs=n_jobs)
        embedding = hdm.run()
        visualize(embedding)
    except Exception as e:
        print(f"Error running HDM: {e}")


if __name__ == "__main__":
    main()