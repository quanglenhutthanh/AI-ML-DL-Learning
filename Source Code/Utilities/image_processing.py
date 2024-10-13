from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def process_image(image):
    width, height = image.size
    print(image.size)
    new_width, new_height = 2240, 2240
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bot = (height + new_height) / 2

    print(left, top, right, bot)
    image = image.crop((left, top, right, bot))

    np_image = np.array(image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image-mean) / std
    np_image = np_image.transpose((2,0,1))
    print(np_image)

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

image_path = '/Users/quanglnt/Documents/AI_ML/Github Learning/AI_ML_Learning/Utilities/test.jpg'
image = Image.open(image_path)
np_image = process_image(image)
tensor_image = torch.from_numpy(np_image).float()
print(tensor_image)
imshow(tensor_image)

# image.show()

