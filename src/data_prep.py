"""
Converts maps in mat format to polar format
"""
import os
import glob
import re
import numpy as np
from scipy.io import loadmat
import pickle


mat_dir = "../maps/"  # Update to the directory where your .mat files are located
mat_files = glob.glob(os.path.join(mat_dir, "soften_mat_*.mat"))

mat_files = sorted(
    mat_files,
    key=lambda f: int(re.search(r"soften_mat_(\d+)\.mat", os.path.basename(f)).group(1))
)

mapping = {}


for name in mat_files:
    mat_data = loadmat(name)
    val = int(name.split("_")[-1].split(".")[0])*25
    start = val-25 
    for k in range(start, val):
        i = k//50
        j = k%50
        mapping[(i, j)] = mat_data["cPSoftMapsMatrix"][i, j]
        
        
with open("mapping.pkl", "wb") as f:
    pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Dictionary saved to mapping.pkl")
