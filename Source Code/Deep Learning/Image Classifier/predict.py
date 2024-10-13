import argparse
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="The file path to the image that needs to be classified.")
    parser.add_argument('checkpoint_path', type=str, help="The file path to the saved model checkpoint.")
    parser.add_argument('--top_k', type=int, default=5, help="The number of top predicted classes to return.")
    parser.add_argument('--category_name', type=str, default='cat_to_name.json', help="The path to the JSON file that maps category indecies to human-readable class names.")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def process_image(image):
    image.thumbnail((256, 256))
    
    width, height = image.size
    print(image.size)
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bot = (height + new_height) / 2

    print(left, top, right, bot)
    image = image.crop((left, top, right, bot))

    np_image = np.array(image) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image-mean) / std
    np_image = np_image.transpose((2,0,1))
    tensor_image = torch.from_numpy(np_image).float()

    return tensor_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    plt.show()
    return ax

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(args):
    
    model = load_checkpoint(args.checkpoint_path)
    device = get_device()
    model = model.to(device)

    image = Image.open(args.image_path)
    image = process_image(image)
    processed_image = image
    image = image.unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
    output = torch.exp(output)
    with open(args.category_name, 'r') as f:
        cat_to_name = json.load(f)
    idx_to_flower = {idx:cat_to_name[cls] for cls, idx in model.class_to_idx.items()}
    top_probs, top_classes = torch.topk(output, args.top_k)
    predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]
    return processed_image, top_probs.tolist(), predicted_flowers

if __name__ == '__main__':
    args = get_args()
    processed_image, probs, classes = predict(args)
    print(f'Top predicted labels: {classes}')
    print(f'Top probabilities {probs}')
