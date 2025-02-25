import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_samples(loader, model, device, title="Generated Samples"):
    """
    Generates and visualizes a batch of images from a generator model.
    
    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing input images.
        model (torch.nn.Module): Trained generator model.
        device (torch.device): Device to run inference on (CPU/GPU).
        title (str): Title for the generated image plot.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = batch.to(device)
        output = model(batch)
        
        # Denormalize the output for visualization
        output = output * 0.5 + 0.5  # Assuming input was normalized to [-1, 1]
        grid = make_grid(output, nrow=4, padding=2, normalize=True)
        np_grid = grid.cpu().numpy().transpose((1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(np_grid)
        plt.title(title)
        plt.axis("off")
        plt.show()
