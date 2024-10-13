# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Implementation

## Data Loading and Preprocessing
### Data Transformation

To perform data transformation using Pytorch, you can use torchvision.transform module.
Common transformation including resizing, cropping, normalization and converting image to tensor. You can define a series of transformation using `transforms.Compose`.

```python
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
            transforms.ToTensor,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
```

### Data Loading

Dataset should be organized in a directory structure like this:

```
root_dir/
    class1/
        img1
        img2
        ...
    class2/
        img1
        img2
        ...
    ...
```

Create dataset using `torchvision.datasets.ImageFolder` 

Use `torch.utils.data.DataLoader` to create an iterable over the dataset.

```python
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
```

## Building and Training Classifier

### Pre-trained model
Use pre-trained network and freeze its parameters. Freezing the parameters mean that these parameters will not be computed during backpropagation, effectively preventing them from being updated during training. This is useful when use want to use the pre-trained model as a feature extractor and only train the custom classifier on top of it.

`torchvision.models` are module in Pytorch provides access to variety of pre-trained models for computer vision tasks such as image classification, object detection, segmentation. Popular models are ResNet, VGG, Inception and MobileNet. These models are trained on large datasets like ImageNet. You can load these models with pre-trained weight and fine-tune them form specific task or use can use them as feature extractor.

```python
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
```

### Custom classifier

Define new, untrained feed-forward network as classifier, using ReLU activation and dropout. Use `torch.nn.Sequential` to define custom classifier to replace the default classifier of a pre-trained mode.

```python
in_feature_of_pretrained_model = model.classifier[0].in_features
number_of_data_classes = len(image_datasets['train'].classes)
classifier = nn.Sequential(
    nn.Linear(in_feature_of_pretrained_model, 2048),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(2048, number_of_data_classes),
    nn.LogSoftmax(dim=1)
)
model.classifier = classifier

```
### Loss, gradient and optimizer

- Loss function measure how well model's predcition match actual data. It calculate the differences between predicted values and the true values. The goal of training a models is to minimize this loss.

- Gradient indicates the direction and rate of change of the loss function. The gradient tell us how to adjust model's parameters to reduce the loss. It points in the direction of the steepest increase in loss, so we move in opposite direction to minimize the loss.

- Optimizer is an algorithm that updates the model's parameters based on the gradients. It decides how to change the parameters to reduce the loss.

### Train the classifier layers using backpropagation using the pre-trained network to get the features
We use pre-trained model for feature extraction, we train the classifier using back backpropagation.

Backpropagation is the process used in neural network to minimize error in prediction. It helps to update the weights of the network to reduce loss value (the difference between predicted values and actual values).


- Track the loss and accuracy on the validation set to determine the best hyperparameters

## Testing

## Classification