import matplotlib.pyplot as plt
import torchvision

def view_image(image_tensor, title="Image"):
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