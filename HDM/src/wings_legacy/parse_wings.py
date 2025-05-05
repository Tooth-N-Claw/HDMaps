import numpy as np
import os

# Specify the directory containing your text files
directory_path = 'data/ptc_02_aligned/'
# Rename this to make its purpose clearer (it's the output DIRECTORY)
output_dir_path = 'data/ptc_02_aligned_npy/'

# make output directory if it doesn't exist
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# Get all .txt files in the directory
txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

for filename in txt_files:
    input_file_path = os.path.join(directory_path, filename) # Renamed for clarity

    # Load the matrix from the text file
    try:
        matrix = np.loadtxt(input_file_path, delimiter=',')
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}")
        continue # Skip to the next file if loading fails

    # Construct the output FILENAME
    output_filename = filename.replace('.txt', '.npy')

    # Construct the full output FILE PATH using the original output DIRECTORY path
    output_file_path = os.path.join(output_dir_path, output_filename)

    # Save as .npy file using the correct full path
    try:
        np.save(output_file_path, matrix)
        print(f"Converted {filename} to {output_file_path}")
    except Exception as e:
        print(f"Error saving {output_file_path}: {e}")