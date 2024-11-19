import torch
from torchvision import datasets, transforms
from pathlib import Path
from helpers import get_data_location, compute_mean_and_std



def get_data_loaders():
    base_path = get_data_location()
    data_loaders = {'train':None, 'valid':None, 'test':None}
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # Resize the image to 256 first, then crop them to 224, normalize

    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'valid' : transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms['train']
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms['valid']
    )

    # Obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)
    

get_data_loaders()


