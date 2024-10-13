import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help="The directory that containts dataset for training and validation.")
    parser.add_argument('--save_dir',type=str, help="The directoy where the trained model will be saved.")
    parser.add_argument('--arch', type=str, default='vgg16', help="The name of the model architecture to use for training, suppored architectures includes: vgg13, vgg16, vgg19.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="The learning rate for training model")
    parser.add_argument('--hidden_unit', type=int, default=512, help="The number of units in the hidden layer of classifier.")
    parser.add_argument('--epochs',type=int, default=5, help="The number of training epochs.")

    return parser.parse_args()

def load_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid_test' : transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datesets = {
        'train' : datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid_test']),
        'test' : datasets.ImageFolder(root=test_dir, transform=data_transforms['valid_test']),
    }

    data_loaders = {
        'train' : DataLoader(dataset=image_datesets['train'], batch_size=64, shuffle=True),
        'valid' : DataLoader(dataset=image_datesets['valid'], batch_size=64),
        'test' : DataLoader(dataset=image_datesets['test'], batch_size=64)
    }
    return image_datesets, data_loaders

def validate_model(model, validate_data):
    device = get_device()
    valid_loss = 0
    correct = 0
    total = 0
    criterion = nn.NLLLoss()
    model.eval()
    with torch.no_grad():
        for inputs, labels in validate_data:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_loss = valid_loss/total
    accuracy = correct/total

    return valid_loss, accuracy

def train_model(args, image_datasets, data_loaders):
    device = get_device()
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_feature_of_pretrained_model = model.classifier[0].in_features
    number_of_data_classes = len(image_datasets['train'].classes)
    classifier = nn.Sequential(
        nn.Linear(in_feature_of_pretrained_model, args.hidden_unit),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_unit, number_of_data_classes),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device=device)
    for epoch in range(args.epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0
        progress_bar = tqdm(enumerate(data_loaders['train']), total=len(data_loaders['train']), desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            progress_bar.set_postfix({'batch_loss': loss.item()})
        valid_loss, accuracy = validate_model(model=model, validate_data=data_loaders['valid'])

    print(f"Epoch {epoch+1}/{args.epochs}.. "
        f"Train loss: {train_loss/len(data_loaders['train']):.3f}.. "
        f"Validation loss: {valid_loss:.3f}.. "
        f"Validation accuracy: {accuracy:.3f}")
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'classifier' : model.classifier,
        'state_dict' : model.state_dict(),
        'epochs' : args.epochs,
        'optim_state' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx
    }
    torch.save(checkpoint,'checkpoint.pth')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

if __name__ == '__main__':
    args = get_args()
    image_datasets, data_loaders = load_data(args=args)
    train_model(args, image_datasets=image_datasets, data_loaders=data_loaders)





