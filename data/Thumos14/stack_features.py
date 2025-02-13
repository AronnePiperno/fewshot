import glob
import numpy as np
import os


base_folder = './data/Thumos14/support_videos_features'



stacked_features = []

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        npy_files = glob.glob(os.path.join(subfolder_path, '*.npy'))

        if not npy_files:
            print(f"No .npy files found in the folder {subfolder_path}.")
            continue

        features = [np.mean(np.load(npy_file), axis=0) for npy_file in npy_files]
        
        #stack features
        stacked_features.append(np.stack(features))




print(stacked_features)

#stack all features
stacked_features = np.concatenate(stacked_features, axis=0)

#save stacked features
np.save(os.path.join(base_folder, 'stacked_features.npy'), stacked_features)

        
