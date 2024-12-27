import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from device import get_device
from simple_nn import SimpleNN
from simple_cnn import SimpleCNN
from view_image import view_image, view_tensor_image

model_path = "mnist_model.pht"
# Load model
model = SimpleCNN()
with open(model_path, 'rb') as f:
    state_dict = torch.load(f, weights_only=True)
model.load_state_dict(state_dict)

# View model information
print(model)  # Display the model architecture

# For more detailed information about the model's parameters:
print(f"Model summary: {model}")

# You can also view the parameters' details (e.g., number of parameters, layers, etc.)
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape}")