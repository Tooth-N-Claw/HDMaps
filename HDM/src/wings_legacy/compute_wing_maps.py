from scipy.spatial import KDTree
import os
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import save_npz

def process_matrix_pair(args):
    """Process a single pair of samples to compute their distance matrix."""
    i, j, file_i, file_j, data_path, output_dir, max_distance = args
    
    try:
        # Load samples with memory mapping
        sample_i = np.load(os.path.join(data_path, file_i), mmap_mode='r')
        sample_j = np.load(os.path.join(data_path, file_j), mmap_mode='r')
        
        # Build KDTrees
        tree_i = KDTree(sample_i)
        tree_j = KDTree(sample_j)
        
        # Compute distance matrix
        dist_matrix = KDTree.sparse_distance_matrix(
            tree_i, tree_j, max_distance=max_distance, output_type='coo_matrix'
        ).tocsr()
        
        # Save matrix
        output_path = os.path.join(output_dir, f'matrix_{i}_{j}.npz')
        save_npz(output_path, dist_matrix)
        
        return i, j, output_path, True
    except Exception as e:
        print(f"Error processing pair ({i}, {j}): {e}")
        return i, j, None, False

def parallel_kdtree_distances(data_samples_path, output_dir, max_distance=0.01, max_workers=None, batch_size=50):
    """
    Compute KDTree-based sparse distance matrices in parallel with controlled memory usage.
    
    This approach:
    1. Processes pairs in parallel using multiple workers
    2. Controls memory usage with batching
    3. Uses memory mapping for data files
    4. Saves matrices incrementally
    
    Args:
        data_samples_path: Path to directory with point cloud data
        output_dir: Directory to save sparse matrices
        max_distance: Maximum distance for sparse distance matrix computation
        max_workers: Maximum number of parallel workers (defaults to CPU count - 1)
        batch_size: Number of pairs to process in one batch
    """
    os.makedirs(output_dir, exist_ok=True)
    file_names = sorted(os.listdir(data_samples_path))
    num_samples = len(file_names)
    
    # Prepare all tasks
    tasks = []
    for i in range(num_samples):
        for j in range(num_samples):
            tasks.append((i, j, file_names[i], file_names[j], data_samples_path, output_dir, max_distance))
    
    # Determine max workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Process in batches to control memory usage
    metadata = np.empty((num_samples, num_samples), dtype=object)
    
    for batch_start in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch = tasks[batch_start:batch_end]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_matrix_pair, batch))
        
        # Update metadata
        for i, j, output_path, success in results:
            if success:
                metadata[i, j] = output_path
        
        # Force garbage collection
        gc.collect()
    
    # Save metadata
    np.save(os.path.join(output_dir, 'distance_matrix_metadata.npy'), metadata)
    
    print(f"Processing complete. Sparse matrices saved to {output_dir}")

def load_results(metadata_path):
    """
    Function to load the computed sparse matrices using the metadata file.
    
    Args:
        metadata_path: Path to the metadata file
    
    Returns:
        Dictionary with indices as keys and sparse matrices as values
    """
    from scipy.sparse import load_npz
    
    metadata = np.load(metadata_path, allow_pickle=True)
    result = {}
    
    for i in range(metadata.shape[0]):
        row_dict = {}
        for j in range(metadata.shape[1]):
            if metadata[i, j]:
                row_dict[j] = load_npz(metadata[i, j])
        result[i] = row_dict
    
    return result

if __name__ == "__main__":
    data_samples_path = "../data/ptc_02_aligned_npy/"
    output_dir = "distance_matrices"
    max_distance = 0.05
    
    parallel_kdtree_distances(data_samples_path, output_dir, max_distance)