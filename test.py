import torch

# put device on gpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)