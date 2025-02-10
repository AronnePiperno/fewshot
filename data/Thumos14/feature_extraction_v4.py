import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Paths (keep your existing paths)
support_videos_path = './data/Thumos14/support_videos'
support_videos_features_path = './data/Thumos14/support_videos_features'

# Load DINOv2 model and processor
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(
    device := torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
model.eval()

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
        frames, frame_skip = [], 5
        frame_count = 0

        # Frame extraction
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))  # Keep as PIL Image
            frame_count += 1
        cap.release()

        # Process frames in batches
        batch_size = 16
        all_features = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # DINOv2 preprocessing
            inputs = processor(
                images=batch_frames,
                return_tensors="pt",
                do_resize=True,
                do_center_crop=True,
                crop_size=processor.crop_size["height"],
                size=processor.size["shortest_edge"]
            ).to(device)

            # Feature extraction
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(**inputs)
                # Use [CLS] token (index 0) for global features
                features = outputs.last_hidden_state[:, 0, :]
                features = torch.nn.functional.normalize(features, dim=-1)
            
            all_features.append(features.cpu().numpy())

        # Save concatenated features
        if len(all_features) > 0:
            features = np.concatenate(all_features, axis=0)
            np.save(
                os.path.join(output_path, video.split('.')[0] + '.npy'),
                features
            )
            print(f'Saved {video.split(".")[0]}.npy in {folder}')

print('Feature extraction complete.')