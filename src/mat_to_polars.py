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

# print(loadmat(mat_files[3])["cPSoftMapsMatrix"][0, 0])

# Sort files numerically so that e.g. "soften_mat_2.mat" comes before "soften_mat_10.mat"
mat_files = sorted(
    mat_files,
    key=lambda f: int(re.search(r"soften_mat_(\d+)\.mat", os.path.basename(f)).group(1))
)

mapping = {}

#print(mat_files)


# for i, name in enumerate(mat_files):
#     mat_data = loadmat(name)
#     for j in range(25):
#         k = i // 2
#         t = j + (i % 2)*25
#         mapping[(k, t)] = mat_data["cPSoftMapsMatrix"][k, t]
    
#print(mapping)


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

# Later, to load (deserialize) the dictionary from the file:
# with open("mapping.pkl", "rb") as f:
#     loaded_mapping = pickle.load(f)

# print("Dictionary loaded from mapping.pkl")
# print(loaded_mapping)

# print(mapping[(26, 26)].shape)

#print(f"mapping: {len(mapping)}")

#print a single elemnt
#print(mapping[(0, 2)])
    
# # -----------------------------------
# # 2. Determine Block Layout and Dimensions
# # -----------------------------------
# # We assume 100 blocks arranged in a 10x10 grid.
# num_block_rows = 10
# num_block_cols = 10

# # Each block is assumed to be 5x5.
# block_height, block_width = blocks[0].shape
# print(f"Each block is of shape: {block_height}x{block_width}")

# # -----------------------------------
# # 3. Build a Dictionary Mapping (i, j) to Cell Content
# # -----------------------------------
# # The overall matrix will be 50x50, so we compute global indices as:
# #   global_row = block_row * block_height + local_row
# #   global_col = block_col * block_width  + local_col
# mapping = {}

# for block_row in range(num_block_rows):
#     for block_col in range(num_block_cols):
#         # Compute the index into the flat list of blocks.
#         block_index = block_row * num_block_cols + block_col
#         block = blocks[block_index]
#         for local_i in range(block_height):
#             for local_j in range(block_width):
#                 global_i = block_row * block_height + local_i
#                 global_j = block_col * block_width + local_j
#                 # Store the cell (which may be a sparse matrix or any other matrix)
#                 mapping[(global_i, global_j)] = block[local_i, local_j]

# # -----------------------------------
# # 4. (Optional) Check the Result
# # -----------------------------------
# print("Total cells in mapping:", len(mapping))
# # For example, print the type of the matrix in cell (10, 10)
# if (10, 10) in mapping:
#     print("Type of matrix at (10,10):", type(mapping[(10, 10)]))
# else:
#     print("Cell (10,10) not found in mapping")

# # df = pl.DataFrame(data, schema=["row_name", "col_name", "map_matrix"])

