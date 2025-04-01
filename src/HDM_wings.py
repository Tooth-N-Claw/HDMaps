from HDM import HDM


if __name__ == "__main__":
    HDM(
        data_samples_path="../data/ptc_02_aligned_npy",
        map_path=None,
        base_dist_path=None,
        num_neighbors=4,
        base_epsilon=0.04,
        num_eigenvectors=4,
        subsample_mapping=0.1,
    )