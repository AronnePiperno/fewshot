import os
import torch
import numpy as np
from PIL import Image
import open_clip
import decord
from tqdm import tqdm

# Configuration
VIDEO_ROOT = './data/Thumos14/support_videos'
FEATURE_ROOT = './data/Thumos14/support_videos_features'
FRAME_STRIDE = 10  # Sample every 5th frame
BATCH_SIZE = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with proper transforms
model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model = model.eval().to(DEVICE)

def extract_video_features(video_path: str) -> np.ndarray:
    """Extract frame features from a video file"""
    try:
        vr = decord.VideoReader(video_path, num_threads=4)
        frame_indices = range(0, len(vr), FRAME_STRIDE)
        
        if not frame_indices:
            return np.empty((0, model.visual.output_dim), dtype=np.float16)
        
        frames = vr.get_batch(frame_indices).asnumpy()
        preprocessed = torch.stack([
            preprocess(Image.fromarray(frame)) for frame in frames
        ]).to(DEVICE)

        features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in torch.split(preprocessed, BATCH_SIZE):
                features.append(model.encode_image(batch).cpu())
                
        return torch.cat(features).numpy().astype(np.float16)
    
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return np.empty((0, model.visual.output_dim), dtype=np.float16)

def main():
    os.makedirs(FEATURE_ROOT, exist_ok=True)

    # Process all class directories
    for class_dir in tqdm(os.listdir(VIDEO_ROOT), desc="Processing classes"):
        video_dir = os.path.join(VIDEO_ROOT, class_dir)
        save_dir = os.path.join(FEATURE_ROOT, class_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Process all videos in current class
        for video_file in tqdm(os.listdir(video_dir), desc=class_dir, leave=False):
            video_path = os.path.join(video_dir, video_file)
            features = extract_video_features(video_path)
            
            save_path = os.path.join(save_dir, f"{os.path.splitext(video_file)[0]}.npy")
            np.save(save_path, features)

if __name__ == "__main__":
    main()
    print("Feature extraction completed successfully.")