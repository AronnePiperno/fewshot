import glob
import numpy as np
import os

base_folder = './data/Thumos14/support_videos_features'

def fourier_transform(array, target_length):
    # Apply Fourier Transform along the temporal axis
    freq_array = np.fft.fft(array, axis=0)
    
    # Truncate or pad in the frequency domain
    current_length = freq_array.shape[0]
    if current_length > target_length:
        freq_array = freq_array[:target_length]  # Truncate
    else:
        pad_width = ((0, target_length - current_length),) + ((0, 0),) * (freq_array.ndim - 1)
        freq_array = np.pad(freq_array, pad_width, mode='constant')  # Pad
    
    return freq_array

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

        # Determine a common target length (e.g., median length)
        target_length = int(np.median([array.shape[0] for array in arrays]))

        # Apply Fourier transform and adjust shapes
        transformed_arrays = [fourier_transform(array, target_length) for array in arrays]

        # Average in the frequency domain
        average_freq_array = np.mean(transformed_arrays, axis=0)

        # Inverse Fourier Transform to return to the original domain
        average_array = np.fft.ifft(average_freq_array, axis=0).real  # Keep only the real part

        # Save the averaged array
        average_file_path = os.path.join(subfolder_path, f'{subfolder}_average.npy')
        np.save(average_file_path, average_array)
        print(f"Saved {subfolder}_average.npy in {subfolder_path}")
