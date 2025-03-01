import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def denormalize(tensor):
    """
    Converts a normalized tensor image (range [-1, 1]) back to [0, 1] for visualization.
    
    Args:
        tensor (torch.Tensor): Tensor image of shape (C, H, W) or (N, C, H, W).
    
    Returns:
        torch.Tensor: Denormalized tensor image.
    """
    return tensor * 0.5 + 0.5  # Assuming normalization was (-1,1)

def generate_samples(loader, model, device, title="Generated Samples"):
    """
    Generates and visualizes a batch of images from a virtual try-on generator model.
    
    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing input images.
        model (torch.nn.Module): Trained generator model.
        device (torch.device): Device to run inference on (CPU/GPU).
        title (str): Title for the generated image plot.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        person_img, clothing_img, keypoints = batch  # Ensure correct dataset structure
        
        person_img, clothing_img, keypoints = (
            person_img.to(device),
            clothing_img.to(device),
            keypoints.to(device),
        )

        generated_img = model(person_img, clothing_img)  # Pass inputs correctly

        # Prepare visualization
        person_img = denormalize(person_img)
        clothing_img = denormalize(clothing_img)
        generated_img = denormalize(generated_img)

        # Create a grid: [Person | Clothing | Output]
        grid = make_grid(
            torch.cat([person_img, clothing_img, generated_img], dim=0),
            nrow=3,
            padding=2,
            normalize=True,
        )

        np_grid = grid.cpu().numpy().transpose((1, 2, 0))

        plt.figure(figsize=(10, 5))
        plt.imshow(np_grid)
        plt.title(title)
        plt.axis("off")
        plt.show()
