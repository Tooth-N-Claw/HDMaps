import os
import numpy as np
import scipy.sparse as sparse
from scipy.io import loadmat
from scipy.sparse import linalg
import trimesh
from tqdm import tqdm  # Add this for better progress tracking

class HorizontalDiffusionMaps:
    """
    Implementation of Horizontal Diffusion Maps algorithm for analyzing shape data.
    """
    
    def __init__(self, data_dir="../platyrrhine/", num_neighbors=4, base_epsilon=0.04, num_eigenvectors=4):
        """
        Initialize HDM with configuration parameters.
        
        Args:
            data_dir: Directory containing the data files
            num_neighbors: Number of neighbors to consider for diffusion
            base_epsilon: Epsilon parameter for the base kernel
            num_eigenvectors: Number of eigenvectors to compute
        """
        self.data_dir = data_dir
        self.num_neighbors = num_neighbors
        self.base_epsilon = base_epsilon
        self.num_eigenvectors = num_eigenvectors
        
        # Will be populated during processing
        self.data_samples = None
        self.maps = None
        self.cumulative_block_indicies = None
        self.num_data_samples = 0
        
    def symmetrize(self, matrix):
        """
        Make a matrix symmetric.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Symmetrized matrix
        """
        return 0.5 * (matrix + matrix.T)
    
    def load_maps(self):
        """
        Load soft map matrices from MATLAB file.
        
        Returns:
            Soft map matrices
        """
        try:
            map_path = os.path.join(self.data_dir, "softMapMatrix.mat")
            maps = loadmat(map_path)["softMapMatrix"]
            return maps
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find softMapMatrix.mat in {self.data_dir}")
        except Exception as e:
            raise Exception(f"Error loading map matrices: {e}")
    
    def load_data_samples(self):
        """
        Load vertex data from OFF files.
        
        Returns:
            List of vertex data arrays
        """
        try:
            names_path = os.path.join(self.data_dir, "Names.mat")
            names = loadmat(names_path)["Names"]
            
            off_dir = os.path.join(self.data_dir, "ReparametrizedOFF")
            data_samples = []
            
            for name in tqdm(names[0], desc="Loading data samples"):
                path = os.path.join(off_dir, f"{name[0]}.off")
                try:
                    vertices = trimesh.load(path).vertices
                    data_samples.append(vertices)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
            
            return data_samples
        except Exception as e:
            raise Exception(f"Error loading data samples: {e}")
    
    def compute_base_kernel(self):
        """
        Compute the base kernel from distance matrix.
        
        Returns:
            Base diffusion matrix and neighbor indices
        """
        try:
            dist_path = os.path.join(self.data_dir, "FinalDists.mat")
            base_dist = loadmat(dist_path)["dists"]
            base_dist = base_dist - np.diag(np.diag(base_dist))
            
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
            
            # Update the distances with values from the symmetrized matrix
            for j in range(self.num_data_samples):
                s_dists[j, :] = base_weights[j, row_nns[j, :]].toarray().flatten()
            
            # Apply kernel function
            s_dists = np.exp(-np.square(s_dists) / self.base_epsilon)
            
            return s_dists, row_nns
        except Exception as e:
            raise Exception(f"Error computing base kernel: {e}")
    
    def cumulative_indices(self):
        """
        Compute cumulative indices for block matrices.
        
        Returns:
            Array of cumulative indices
        """
        return np.insert(
            np.cumsum([len(sample) for sample in self.data_samples], dtype=np.int32), 
            0, 0
        )
    
    def compute_diffusion_matrix(self, base_diffusion_mat, row_nns):
        """
        Compute the diffusion matrix.
        
        Args:
            base_diffusion_mat: Base diffusion matrix
            row_nns: Indices of nearest neighbors
            
        Returns:
            Diffusion matrix as CSR sparse matrix
        """
        mat_row_idx = []
        mat_col_idx = []
        vals = []
        
        for j in tqdm(range(self.num_data_samples), desc="Computing diffusion matrix"):
            for nns in range(self.num_neighbors):
                if base_diffusion_mat[j, nns] == 0:
                    continue
                
                k = row_nns[j, nns]
                map_matrix = self.maps[j, k]
                
                # Forward mapping
                coo = map_matrix.tocoo()
                mat_row_idx.extend(coo.row + self.cumulative_block_indicies[j])
                mat_col_idx.extend(coo.col + self.cumulative_block_indicies[k])
                vals.extend(base_diffusion_mat[j, nns] * coo.data)
                
                # Backward mapping (transposed)
                coo = map_matrix.T.tocoo()
                mat_row_idx.extend(coo.row + self.cumulative_block_indicies[k])
                mat_col_idx.extend(coo.col + self.cumulative_block_indicies[j])
                vals.extend(base_diffusion_mat[j, nns] * coo.data)
        
        # Determine proper dimensions for the matrix
        total_size = self.cumulative_block_indicies[-1]
        
        diffusion_matrix = sparse.csr_matrix(
            (vals, (mat_row_idx, mat_col_idx)), 
            shape=(total_size, total_size)
        )
        
        return diffusion_matrix
    
    def compute_horizontal_diffusion_laplacian(self, diffusion_matrix):
        """
        Compute the horizontal diffusion Laplacian.
        
        Args:
            diffusion_matrix: Diffusion matrix
            
        Returns:
            Tuple of (horizontal_diffusion_laplacian, sqrt_diag)
        """
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
        """
        Perform eigendecomposition of the Laplacian.
        
        Args:
            horizontal_diffusion_laplacian: The Laplacian matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            eigvals, eigvecs = sparse.linalg.eigsh(
                horizontal_diffusion_laplacian, 
                k=self.num_eigenvectors, 
                which="LM", 
                maxiter=5000, 
                tol=1e-10
            )
            
            # Sort in descending order
            reverse_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[reverse_idx]
            eigvecs = eigvecs[:, reverse_idx]
            
            return eigvals, eigvecs
        except Exception as e:
            raise Exception(f"Error in eigendecomposition: {e}")
    
    def run(self):
        """
        Run the complete HDM algorithm.
        
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
            
            # Compute base kernel
            base_diffusion_matrix, row_nns = self.compute_base_kernel()
            print("Computed base kernel")
            
            # Compute diffusion matrix
            diffusion_matrix = self.compute_diffusion_matrix(base_diffusion_matrix, row_nns)
            print("Computed diffusion matrix")
            
            # Check if diffusion_matrix was created successfully
            if diffusion_matrix.shape[0] == 0 or diffusion_matrix.nnz == 0:
                raise ValueError("Empty diffusion matrix created. Check previous steps.")
            
            # Compute horizontal diffusion Laplacian
            horizontal_diffusion_laplacian, sqrt_diag = self.compute_horizontal_diffusion_laplacian(diffusion_matrix)
            print("Computed horizontal diffusion Laplacian")
            
            # Perform eigendecomposition
            eigvals, eigvecs = self.eigendecomposition(horizontal_diffusion_laplacian)
            print("Eigendecomposition done")
            
            # Handle potential numerical issues
            inf_values = np.sum(np.isinf(sqrt_diag.data))
            if inf_values > 0:
                print(f"Warning: {inf_values} infinite values found and replaced with zeros")
                sqrt_diag.data[np.isinf(sqrt_diag.data)] = 0
            
            # Construct final embedding
            bundle_HDM = sqrt_diag @ eigvecs[:, 1:]
            sqrt_lambda = sparse.diags(np.sqrt(eigvals[1:]), 0)
            bundle_HDM_full = bundle_HDM @ sqrt_lambda
            
            return bundle_HDM_full
            
        except Exception as e:
            print(f"Error in HDM processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def visualize(self, embedding=None):
        """
        Visualize the embedding.
        
        Args:
            embedding: The embedding to visualize (default: use the first 3 components)
        """
        from visualize import visualize  # Import here to avoid circular imports
        
        if embedding is None:
            embedding = self.run()
            
        visualize(embedding[:, :3])


def main():
    """
    Main entry point for running the HDM algorithm.
    """
    try:
        hdm = HorizontalDiffusionMaps()
        embedding = hdm.run()
        print("Visualizing")
        hdm.visualize(embedding)
    except Exception as e:
        print(f"Error running HDM: {e}")


if __name__ == "__main__":
    main()