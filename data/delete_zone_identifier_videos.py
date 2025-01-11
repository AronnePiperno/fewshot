# delete all the videos that end with Zone.Identifier
import os
import sys
import shutil

def delete_zone_identifier_videos(video_path):
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith("Zone.Identifier"):
                os.remove(os.path.join(root, file))
                print(f"Deleted {file}")

if __name__ == "__main__":
    video_path = "data/Thumos14/videos/"
    delete_zone_identifier_videos(video_path)