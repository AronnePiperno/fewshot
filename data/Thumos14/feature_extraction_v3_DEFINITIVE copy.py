import os
import cv2
import torch
import numpy as np
from PIL import Image
import open_clip

# Paths
support_videos_path = './data/Thumos14/support_videos'
support_videos_features_path = './data/Thumos14/support_videos_features'

# Load CoCa model with transforms
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)

# Create output directory
os.makedirs(support_videos_features_path, exist_ok=True)

# Feature extraction loop
for class_name in os.listdir(support_videos_path):
    class_path = os.path.join(support_videos_path, class_name)
    output_path = os.path.join(support_videos_features_path, class_name)
    os.makedirs(output_path, exist_ok=True)

    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_skip = 10
        frame_count = 0

        # Frame processing with model's transforms
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Convert and preprocess frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                tensor_frame = preprocess(pil_image)  # Applies resize, normalization, etc.
                frames.append(tensor_frame)
            
            frame_count += 1

        cap.release()

        if not frames:
            continue  # Skip videos with no valid frames

        # Batch processing with memory optimization
        batch_size = 32  # Increased batch size for better GPU utilization
        all_features = []
        for i in range(0, len(frames), batch_size):
            batch = torch.stack(frames[i:i+batch_size]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = model.encode_image(batch)
            all_features.append(features.cpu().numpy())

        # Save features
        features = np.concatenate(all_features, axis=0)
        output_file = os.path.join(output_path, video_file.split('.')[0] + '.npy')
        np.save(output_file, features)
        print(f'Saved {output_file}')

print('Feature extraction complete.')