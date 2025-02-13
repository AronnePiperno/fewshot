import glob
import numpy as np
import os

base_folder = './data/Thumos14/support_videos_features'

def pad_array(arr, target_shape):
    """Pad array with zeros to match target shape in all dimensions"""
    pad_width = []
    for dim, target_size in zip(arr.shape, target_shape):
        pad_total = max(target_size - dim, 0)
        pad_width.append((0, pad_total))
    return np.pad(arr, pad_width, mode='constant')

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Load all arrays in subfolder
    npy_files = glob.glob(os.path.join(subfolder_path, '*.npy'))
    arrays = [np.load(f) for f in npy_files]
    
    print(f"Loaded {len(arrays)} arrays from {subfolder}")
    # Find maximum dimensions across all arrays
    if not arrays:
        print(f"Skipping empty folder: {subfolder}")
        continue
    
    # Get maximum shape across all dimensions
    all_shapes = [arr.shape for arr in arrays]

    print(f"all shape {all_shapes}")
    max_shape = tuple(np.max(all_shapes, axis=0))
    
    # Pad all arrays to max shape
    padded_arrays = [pad_array(arr, max_shape) for arr in arrays]
    
    # Average all padded arrays
    average_array = np.mean(padded_arrays, axis=0)
    
    # Save result
    output_path = os.path.join(subfolder_path, f"{subfolder}_average.npy")
    np.save(output_path, average_array)
    print(f"Saved averaged features to {output_path}")