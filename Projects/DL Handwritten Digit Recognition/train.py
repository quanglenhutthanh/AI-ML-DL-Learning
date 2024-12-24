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
from view_image import view_batch_images, save_batch_images
from tqdm import tqdm  # Import tqdm for the progress bar


def train(model, device, train_loader, test_loader, criterion, optimizer, epochs = 5):
    
    # Initialize TensorBoard writer
    # writer = SummaryWriter('runs/mnist_experiment')

    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
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
                
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Log the average loss for this epoch to TensorBoard
        # avg_loss = epoch_loss / len(train_loader)
        # writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        validation(model,device,test_loader)
    # After training, visualize with TensorBoard
    # writer.close()

# Test the model
def validation(model, device, data_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

# Save the trained model
def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)
    print(f'[INFO] Model saved to {model_save_path}')


if __name__ == "__main__":
    root_path = os.path.expanduser('data')
    # Load the dataset
    transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST
        ]),
        'valid_test' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }
    train_dataset = datasets.MNIST(root=root_path, download=False, train=True, transform=transforms['train'])
    test_dataset = datasets.MNIST(root=root_path, download=True, train=False, transform=transforms['valid_test'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    # view_batch_images(train_loader=train_loader)

    # Get a batch of images
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # Save the images
    # save_batch_images(images, save_dir="output_images", prefix="mnist_image", file_format="png")

    # Initialize the model, loss function, and optimizer
    device = get_device()
    # model = SimpleNN()
    model = SimpleCNN()
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    train(model=model,device=device,train_loader=train_loader,test_loader=test_loader,criterion=criterion,optimizer=optimizer,epochs = epochs)
    save_model(model=model,model_save_path="mnist_model.pht")