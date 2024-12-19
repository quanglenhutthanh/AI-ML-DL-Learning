import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from simple_nn import SimpleNN
from simple_cnn import SimpleCNN
from device import get_device
from view_image import view_batch_images


def train(model,criterion, optimizer, epochs = 5):
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/mnist_experiment')

    # Train the model
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log the average loss for this epoch to TensorBoard
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # After training, visualize with TensorBoard
    writer.close()

# Test the model
def validation():
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

# Save the trained model
def save_model(model_save_path = "mnist_simple_nn.pht"):
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


# Load the dataset
transforms = {
    'train' : transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
    'valid_test' : transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
}

root_path = os.path.expanduser('data')

train_dataset = datasets.MNIST(root=root_path, download=False, train=True, transform=transforms['train'])
test_dataset = datasets.MNIST(root=root_path, download=True, train=False, transform=transforms['valid_test'])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
view_batch_images(train_loader=train_loader)

# Initialize the model, loss function, and optimizer
device = get_device()
# model = SimpleNN()
model = SimpleCNN()
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 20
train(model=model,criterion=criterion,optimizer=optimizer,epochs = epochs)
validation()
save_model("mnist_simple_cnn.pht")