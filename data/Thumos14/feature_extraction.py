import os
import cv2
import torch
import numpy as np
from PIL import Image
import open_clip

# Path to the support videos
support_videos_path = './data/Thumos14/support_videos'
# Path to the support videos features
support_videos_features_path = './data/Thumos14/support_videos_features'

# Load the CoCa model
tokenize = open_clip.get_tokenizer("coca_ViT-L-14")
model, _, _ = open_clip.create_model_and_transforms(model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create the folder if it doesn't exist
if not os.path.exists(support_videos_features_path):
    os.makedirs(support_videos_features_path)

# create the video features
for folder in os.listdir(support_videos_path):
    class_name = folder
    print('Extracting features for', class_name)
    class_path = os.path.join(support_videos_path, folder)

    #check if the class folder exists
    if not os.path.exists(os.path.join(support_videos_features_path, class_name)):
        os.makedirs(os.path.join(support_videos_features_path, class_name))
    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((224, 224))
            frame = np.array(frame)
            frames.append(frame)
        frames = np.array(frames)
        frames = frames / 255.0
        frames = frames.transpose(0, 3, 1, 2)
        frames = torch.tensor(frames).float().to(device)
        with torch.no_grad():
            features = model.encode_image(frames)
        features = features.cpu().numpy()
        np.save(os.path.join(support_videos_features_path, class_name, video.split('.')[0] + '.npy'), features)
        print('Saved', video.split('.')[0] + '.npy', 'in', class_name)
print('Done')