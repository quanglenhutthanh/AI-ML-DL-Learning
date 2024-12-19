from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from simple_nn import SimpleNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    
def predict(model_path, image_path):
    model = SimpleNN()
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_tensor = transform(image).unsqueeze(0)  # Add batch and channel dimensions

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()

model_path = "mnist_simple_nn.pht"
image_test = "test/test001.png"
predicted = predict(model_path, image_test)
print(f"The number is {predicted}")