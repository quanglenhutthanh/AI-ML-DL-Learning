import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist_experiment")

# Step 1: Load the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
root_path = os.path.expanduser('/Users/quanglnt/Documents/AI_ML/Github Learning/AI_ML_Learning/Deep Learning/Torch/data')

print(root_path)
train_dataset = datasets.MNIST(root=root_path, download=True, train=True, transform=transform)
test_dataset = datasets.MNIST(root=root_path, download=True, train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
# Step 2: Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

   
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Step 3: Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Log a batch of training images
data_iter = iter(train_loader)
images, labels = next(data_iter)
img_grid = torchvision.utils.make_grid(images)  # Create a grid of images
writer.add_image("MNIST Images", img_grid)  # Log images to TensorBoard

# Step 4: Train the model
epochs = 15
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log predictions for the first batch of each epoch
        if batch_idx == 0:
            # Add predictions as images with labels
            _, preds = torch.max(output, 1)
            writer.add_images('Predictions', data, global_step=epoch)
            writer.add_text('Predicted Labels', str(preds.tolist()), global_step=epoch)

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar("Training Loss", avg_loss, epoch)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
writer.close()
# Step 5: Test the model
correct = 0
total = 0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')



