# for every folder in support_videos put 5 random videos from UCF101 folder

import os
import random
import shutil

ucf101_path = './data/Thumos14/UCF101'
support_videos_path = './data/Thumos14/support_videos'

# all the files in the UCF101 folder
ucf101_files = os.listdir(ucf101_path)

# delete the file that dont end with .avi
ucf101_files = [f for f in ucf101_files if f.endswith('.avi')]


# create a dictionary with the class name as key and the list of videos as value
ucf101_dict = {}
for file in ucf101_files:
    class_name = file.split('_')[1]
    if class_name not in ucf101_dict:
        ucf101_dict[class_name] = []
    ucf101_dict[class_name].append(file)

# take 5 random videos from each class (with a folder in support_videos) and copy them to the support_videos folder
print(support_videos_path)
for folder in os.listdir(support_videos_path):
    print('Processing', folder)
    class_name = folder
    videos = ucf101_dict[class_name]
    random_videos = random.sample(videos, 5)
    for video in random_videos:
        shutil.copy(os.path.join(ucf101_path, video), os.path.join(support_videos_path, folder, video))
        print('Copied', video, 'to', folder)

print('Done')