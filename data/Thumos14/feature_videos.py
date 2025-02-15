import os
import torch
import numpy as np
from PIL import Image
import decord
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Configuration
VIDEO_ROOT = './data/Thumos14/videos'
FEATURE_ROOT = './data/Thumos14/features'
FRAME_STRIDE = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DINOv2 model and processor
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)

def extract_video_features(video_path: str) -> np.ndarray:
    """Extract frame features using DINOv2"""
    try:
        # Read video with decord
        vr = decord.VideoReader(video_path, num_threads=4)
        frame_indices = range(0, len(vr), FRAME_STRIDE)
        
        if not frame_indices:
            return np.empty((0, 768), dtype=np.float16)  # dinov2-base features dim
        
        # Process frames in batches
        features = []
        for i in range(0, len(frame_indices), BATCH_SIZE):
            batch_indices = frame_indices[i:i+BATCH_SIZE]
            frames = vr.get_batch(batch_indices).asnumpy()
            
            # Convert to PIL Images and process
            pil_images = [Image.fromarray(frame) for frame in frames]
            inputs = processor(
                images=pil_images,
                return_tensors="pt",
                do_resize=True,
                do_center_crop=True,
                crop_size=processor.crop_size["height"],
                size=processor.size["shortest_edge"]
            ).to(DEVICE)
            
            # Extract features
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(**inputs)
                # Use [CLS] token features (dim 768 for base model)
                batch_features = outputs.last_hidden_state[:, 0, :]
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0).astype(np.float16)
    
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return np.empty((0, 768), dtype=np.float16)

def main():
    os.makedirs(FEATURE_ROOT, exist_ok=True)

    # Process videos based on existing features_old names
    for video_file in tqdm(os.listdir('./data/Thumos14/features_CoCa'), desc="Processing videos"):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEO_ROOT, f"{video_name}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Video {video_path} not found, skipping")
            continue
            
        features = extract_video_features(video_path)
        np.save(os.path.join(FEATURE_ROOT, f"{video_name}.npy"), features)
        print(f"Saved {video_name}.npy")

if __name__ == "__main__":
    main()