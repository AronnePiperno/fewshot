import os
import cv2
import torch
import numpy as np
from PIL import Image
import open_clip

# Paths
support_videos_path = './data/Thumos14/support_videos'
support_videos_features_path = './data/Thumos14/support_videos_features'

# Load CoCa model
tokenize = open_clip.get_tokenizer("coca_ViT-L-14")
model, _, _ = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14", 
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model.eval().to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Create output directory
os.makedirs(support_videos_features_path, exist_ok=True)

# Feature extraction loop
for folder in os.listdir(support_videos_path):
    class_path = os.path.join(support_videos_path, folder)
    output_path = os.path.join(support_videos_features_path, folder)
    os.makedirs(output_path, exist_ok=True)

    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)
        cap = cv2.VideoCapture(video_path)
        frames, frame_skip = [], 10
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).resize((224, 224))
                frames.append(np.array(frame))
            frame_count += 1

        frames = np.array(frames) / 255.0
        frames = (frames - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frames = frames.transpose(0, 3, 1, 2)  # (B, C, H, W)

        # Batch processing
        batch_size, all_features = 16, []
        for i in range(0, len(frames), batch_size):
            batch = torch.tensor(frames[i:i+batch_size]).float().to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = model.encode_image(batch)
            all_features.append(features.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        np.save(os.path.join(output_path, video.split('.')[0] + '.npy'), features)
        print(f'Saved {video.split(".")[0]}.npy in {folder}')

print('Feature extraction complete.')