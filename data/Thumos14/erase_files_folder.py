# erase every file in every folder in support_videos withour deleting the folder

import os
import random
import shutil

support_videos_path = './data/Thumos14/support_videos'

# all the files in the support_videos folder
support_videos_folders = os.listdir(support_videos_path)

for folder in support_videos_folders:
    folder_path = os.path.join(support_videos_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        os.remove(os.path.join(folder_path, file))
        print('Deleted', file, 'in', folder)