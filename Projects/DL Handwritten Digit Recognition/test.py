import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from simple_nn import SimpleNN
from simple_cnn import SimpleCNN
from device import get_device
from view_image import view_image,view_batch_images, save_batch_images
from tqdm import tqdm  # Import tqdm for the progress bar

root_path = os.path.expanduser('data')

# Define transforms for training and testing
transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST
    ]),
    'valid_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

# Define dataset and dataloader
train_dataset = datasets.MNIST(root=root_path, download=True, train=True, transform=transforms['train'])
test_dataset = datasets.MNIST(root=root_path, download=True, train=False, transform=transforms['valid_test'])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = SimpleCNN()
device = get_device()

model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Initialize epoch loss
    correct = 0  # Track number of correct predictions
    total = 0  # Track total predictions

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Move data to device
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Add current batch loss to epoch loss

            # Calculate accuracy for this batch
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update the progress bar
            pbar.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
            pbar.update(1)

    # Calculate the average loss and accuracy for this epoch
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total


    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Now validation (on validation set)
    correct_val = 0
    total_val = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data, target in test_loader:  # Assuming `val_loader` is your validation data loader
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()

    accuracy_val = 100 * correct_val / total_val
    print(f'Validation Accuracy: {accuracy_val:.2f}%')


# Save model after training
torch.save(model.state_dict(), "mnist_simple_cnn.pht")
print(f'Model saved')


model.eval()

test_folder = "test/"

# Loop through all files in the test folder
for filename in os.listdir(test_folder):
    if filename.endswith(".png"):  # Only process .png files (you can add more extensions if needed)
        image_path = os.path.join(test_folder, filename)

        # Load and preprocess the image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image_tensor = transforms['valid_test'](image).unsqueeze(0)  # Add batch and channel dimensions
        image_tensor = image_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        # Convert probabilities to a list of (class, probability)
        class_probabilities = {
            str(class_index): prob.item() for class_index, prob in enumerate(probabilities[0])
        }

        # Print or store the predictions for the current image
        print(f"Predictions for {filename}: {class_probabilities}")
        print()  # Line break for separation