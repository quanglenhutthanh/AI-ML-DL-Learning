import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def imshow(img_loader):
    dataiter = iter(img_loader)

    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        image = images[i].numpy()
        plt.imshow(image.T.squeeze().T)
        plt.show()




data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# Load data
data_root = 'data'
train_dataset = datasets.MNIST(root=data_root, download=True, train=True, transform=data_transform)
test_dataset = datasets.MNIST(root=data_root, download=True, train=False, transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
imshow(test_loader)

# Define neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)