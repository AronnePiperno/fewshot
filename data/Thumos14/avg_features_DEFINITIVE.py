import glob
import numpy as np
import os

base_folder = './data/Thumos14/support_videos_features'

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    npy_files = glob.glob(os.path.join(subfolder_path, '*.npy'))
    arrays = [np.load(f) for f in npy_files]
    
    if not arrays:
        print(f"Skipping empty folder: {subfolder}")
        continue
    
    # Average along the first dimension for each array
    averaged_arrays = [arr.mean(axis=0) for arr in arrays]
    
    # Stack all averaged arrays and compute final average
    final_average = np.mean(np.stack(averaged_arrays), axis=0)
    
    # final acerage dim
    print(f"final average dim {final_average.shape}")


    # Save result
    output_path = os.path.join(subfolder_path, f"{subfolder}_average.npy")
    np.save(output_path, final_average)
    print(f"Saved averaged features to {output_path}")
