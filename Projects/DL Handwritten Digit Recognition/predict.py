from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from simple_nn import SimpleNN
from simple_cnn import SimpleCNN
from view_image import view_image

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
    image_tensor = transform(grayscale_image).unsqueeze(0)  # Add batch and channel dimensions
    # view_image(image_tensor=image_tensor)
    
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
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_tensor = transform(image).unsqueeze(0)  # Add batch and channel dimensions
   
    view_image(image_tensor=image_tensor)
    
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

model_path = "mnist_simple_cnn.pht"
image_test = "test/test002.png"
predicted = predict(model_path, image_test)

print(f"The number is {predicted}")