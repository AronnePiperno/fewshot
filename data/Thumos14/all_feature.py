# for every folder in support_videos_features create a stack of features from the videos in the folder 
# save it with the name of the folder + _all.npy
# use time domain adjust to adjust the length of the features
import os

import numpy as np
import glob


base_folder = './data/Thumos14/support_videos_features'

def time_domain_adjust(array, target_length):
    # Adjust length in time domain through padding/truncation
    current_length = array.shape[0]
    
    if current_length > target_length:
        return array[:target_length]  # Truncate temporal dimension
    else:
        # Pad with zeros along temporal dimension
        pad_width = ((0, target_length - current_length),) + ((0, 0),) * (array.ndim - 1)
        return np.pad(array, pad_width, mode='constant')
    
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

        # Determine a common target length (median of array lengths)
        target_length = int(np.median([array.shape[0] for array in arrays]))

        # Adjust arrays to common length in time domain
        adjusted_arrays = [time_domain_adjust(array, target_length) for array in arrays]

        # Stack the features
        stacked_array = np.stack(adjusted_arrays, axis=0)

        # Save the stacked array
        stacked_file_path = os.path.join(subfolder_path, f'{subfolder}_all.npy')
        np.save(stacked_file_path, stacked_array)
        print(f"Saved {subfolder}_all.npy in {subfolder_path}")