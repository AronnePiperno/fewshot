# erase all the file that ends with Zone.Identifier in the UCF101 folder

import os
import random
import shutil

ucf101_path = './data/Thumos14/UCF101'
support_videos_path = './data/Thumos14/support_videos'

# all the files in the UCF101 folder
ucf101_files = os.listdir(ucf101_path)

# delete the file that ends with Zone.Identifier
ucf101_files = [f for f in ucf101_files if f.endswith('Zone.Identifier')]

for file in ucf101_files:
    os.remove(os.path.join(ucf101_path, file))
    print('Deleted', file)