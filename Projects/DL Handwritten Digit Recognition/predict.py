import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from device import get_device
from simple_nn import SimpleNN
from simple_cnn import SimpleCNN
from view_image import view_image, view_tensor_image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def predict_image(image):
    
    model = SimpleCNN()
    with open('mnist_simple_cnn.pht', 'rb') as f:
        state_dict = torch.load(f, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    image = image.convert('RGBA')
    grayscale_image = Image.new("L", image.size, 255)  # Create a white background
    grayscale_image.paste(image.convert("L"), mask=image.split()[3])  # Use alpha channel as mask
    grayscale_image = grayscale_image.resize((28, 28))  # Resize to 28x28 pixels
    
    grayscale_image.save("processed_image.png")

    image_np = np.array(grayscale_image)
    image_np = 255 - image_np  # Invert colors (MNIST has white digits on black)

    # Normalize to range [0, 1]
    image_np = image_np / 255.0

    image_tensor = transform(image_np)  # Add batch and channel dimensions
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(torch.float32)

    # image_tensor = transform(grayscale_image).unsqueeze(0)  # Add batch and channel dimensions
    
    with torch.no_grad():
        output = model(image_tensor)
        #_, predicted = torch.max(output.data, 1)
        probabilities = torch.softmax(output, dim=1) 
    # Convert probabilities to a list of (class, probability)
    class_probabilities = {
        str(class_index): prob.item() for class_index, prob in enumerate(probabilities[0])
    }
    print(class_probabilities)
    # class_probabilities = {}
    return class_probabilities

def predict(model_path, image_path):
    model = SimpleCNN()
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    # view_image(image=image)
    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to NumPy array and invert colors if needed
    image_np = np.array(image)
    image_np = 255 - image_np  # Invert colors (MNIST has white digits on black)

    # Normalize to range [0, 1]
    image_np = image_np / 255.0

    # Convert to tensor
    
    image_tensor = transform(image_np)  # Add batch and channel dimensions
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(torch.float32)
    # Ensure the tensor is in the correct dtype
    
    # view_tensor_image(image_tensor=image_tensor)
    
    with torch.no_grad():
        output = model(image_tensor)
        #_, predicted = torch.max(output.data, 1)
        probabilities = torch.softmax(output, dim=1) 
    # Convert probabilities to a list of (class, probability)
    class_probabilities = {
        str(class_index): prob.item() for class_index, prob in enumerate(probabilities[0])
    }
    # return predicted.item()
    return class_probabilities
if __name__ == "__main__":
    device = get_device()
    model_path = "trained_model/mnist_simple_cnn.pht"
   
    # Loop through all files in the test folder
    test_folder = "test/"
    for filename in os.listdir(test_folder):
        if filename.endswith(".png"):  # Only process .png files (you can add more extensions if needed)
            image_path = os.path.join(test_folder, filename)
            predicted = predict(model_path = "mnist_model.pht",image_path=image_path)
            print(F"[INFO] The predicted results of the image {image_path} are: {predicted}")
            print()
           