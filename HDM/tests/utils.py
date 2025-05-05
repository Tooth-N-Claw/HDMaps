import pickle

import numpy as np


def load_pikle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data






def compare_arrays(arr1, arr2):
    if arr1.shape != arr2.shape:
        print(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")
        return False
    if not np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8):
        print("Arrays are not equal")
        return False
    return True