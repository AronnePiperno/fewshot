import cv2
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
from PIL import Image
import os


# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Process each video in the videos_to_caption folder
videos_folder = "./videos_to_caption"
frame_folder = "./frames"
captions_folder = "./captions"

videos = sorted([f for f in os.listdir(videos_folder) if f.endswith('.mp4')])

def generate_caption(task_prompt, image_path, text_input=None):
    image = Image.open(image_path)
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    return parsed_answer

for video in videos:
    video_path = os.path.join(videos_folder, video)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0
    frame_number = 0
    video_captions_folder = os.path.join("./captions", os.path.splitext(video)[0])
    
    while success:
        if count % 10 == 0:
            cv2.imwrite(f"{frame_folder}/{frame_number:04d}.png", frame)
            frame_number += 1
        success, frame = cap.read()
        count += 1

    # Process each frame in the frames folder
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])



    prompt = "<DETAILED_CAPTION>"

    for frame in frames:
        print(f"Processing {frame}")
        frame_path = os.path.join(frame_folder, frame)
        caption = generate_caption(prompt, frame_path)

        # put the caption in a txt called like the name of the video
        with open(f"{captions_folder}/{os.path.splitext(video)[0]}.txt", "a") as f:
            f.write(f"{frame} : {caption['<DETAILED_CAPTION>']}\n")

    # Clean up
    for frame in frames:
        os.remove(os.path.join(frame_folder, frame))

