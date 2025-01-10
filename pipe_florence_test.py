import cv2
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
from PIL import Image
import os

# Capture video frames
cap = cv2.VideoCapture('video_test_0000004.mp4')
success, frame = cap.read()
count = 0
frame_number = 0
while success:
    if count % 10 == 0:
        cv2.imwrite(f"captions/frame_{frame_number:04d}.jpg", frame)
        frame_number += 1
    success, frame = cap.read()
    count += 1

# Load model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
# Process each frame in the captions folder
captions_folder = "captions"
frames = sorted([f for f in os.listdir(captions_folder) if f.endswith('.jpg')])

def run_example(task_prompt, image_path, text_input=None):
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

prompt = "<DETAILED_CAPTION>"

for frame in frames:
    print(f"Processing {frame}")
    frame_path = os.path.join(captions_folder, frame)
    run_example(prompt, frame_path)

    # put the caption in a txt file
    with open('captions_not_detailed.txt', 'a') as f:
        f.write(f'{frame}: {run_example(prompt, frame_path)}\n')

# Clean up
for frame in frames:
    os.remove(os.path.join(captions_folder, frame))

