import glob
import numpy as np
import os

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

        arrays = [np.mean(np.load(npy_file), axis=0) for npy_file in npy_files]

        #print dimensions of arrays



        # Simple element-wise average
        average_array = np.mean(arrays, axis=0)


        # Save the averaged array
        average_file_path = os.path.join(subfolder_path, f'{subfolder}_average.npy')
        np.save(average_file_path, average_array)
        print(f"Saved {subfolder}_average.npy in {subfolder_path}")