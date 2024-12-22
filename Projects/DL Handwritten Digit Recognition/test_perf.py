import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model
class BestPracticeCNN(nn.Module):
    def __init__(self):
        super(BestPracticeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x14x14
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x7x7
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10 classes for digits 0-9
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
# Load the dataset
transforms = {
    'train' : transforms.Compose([
                transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate images by 10%
                transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with 50% probability
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0, std 1
            ]),
    'valid_test' : transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
}

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms['train'], download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms['valid_test'], download=True)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
model = BestPracticeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation function
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Train and evaluate
train(model, criterion, optimizer, epochs=20)
evaluate(model)
torch.save(model.state_dict(), "mnist_simple_cnn.pht")
