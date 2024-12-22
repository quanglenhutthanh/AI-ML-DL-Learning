import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToPILImage

def view_image(image):
    plt.imshow(image, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")  # Hide axes for better visualization
    plt.show()

def view_tensor_image(image_tensor, title="Image"):
    image_np = image_tensor.squeeze().numpy()
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

def view_batch_images(train_loader, num_images=8):
    """
    Display a batch of images from the train_loader.
    
    Parameters:
        train_loader (DataLoader): The DataLoader containing the images.
        num_images (int): Number of images to display from the batch.
    """
    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # Get a batch of images and labels
    
    # Make a grid of images
    img_grid = torchvision.utils.make_grid(images[:num_images], nrow=num_images, normalize=True)
    img_np = img_grid.numpy().transpose((1, 2, 0))  # Rearrange dimensions for plotting
    
    plt.figure(figsize=(12, 6))
    plt.imshow(img_np, cmap="gray")
    plt.title("Batch of Images")
    plt.axis("off")
    plt.show()


def save_batch_images(images, save_dir, prefix="image", file_format="png", unnormalize=None):
    """
    Save each image in a batch to a specified directory.
    
    Parameters:
        images (torch.Tensor): Batch of images with shape (B, C, H, W).
        save_dir (str): Directory to save the images.
        prefix (str): Prefix for the saved image filenames.
        file_format (str): File format for the saved images (e.g., "png", "jpg").
        unnormalize (callable, optional): Function to unnormalize the images before saving.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    to_pil = ToPILImage()  # Converts tensors to PIL images
    
    for idx, image in enumerate(images):
        if unnormalize:
            image = unnormalize(image)  # Apply unnormalization if provided
        
        pil_image = to_pil(image)  # Convert to PIL Image
        filename = os.path.join(save_dir, f"{prefix}_{idx}.{file_format}")
        pil_image.save(filename)
        print(f"Saved: {filename}")