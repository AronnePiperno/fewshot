import glob
import numpy as np
import os

base_folder = './data/Thumos14/support_videos_features'

# Iterate through each subfolder in the base folder
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Get all .npy files in the current subfolder
        npy_files = glob.glob(os.path.join(subfolder_path, '*.npy'))

        if not npy_files:
            print(f"No .npy files found in the folder {subfolder_path}.")
            continue

        arrays = [np.load(npy_file) for npy_file in npy_files]

        # Find the maximum shape
        max_shape = tuple(max(sizes) for sizes in zip(*[array.shape for array in arrays]))

        # Pad arrays to the maximum shape
        padded_arrays = []
        for array in arrays:
            pad_width = [(0, max_dim - array_dim) for array_dim, max_dim in zip(array.shape, max_shape)]
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
            padded_arrays.append(padded_array)

        # Sum the padded arrays
        sum_array = np.sum(padded_arrays, axis=0)

        # Calculate the average
        average_array = sum_array / len(npy_files)

        # Save the average array to a new file named after the subfolder
        average_file_path = os.path.join(subfolder_path, f'{subfolder}_average.npy')
        np.save(average_file_path, average_array)
        print(f"Saved {subfolder}_average.npy in {subfolder_path}")