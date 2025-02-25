import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.data_preprocessing import viton_loader
from model import Generator

def generate_samples(loader, model, device, title="Generated Samples"):
    """Generates and visualizes a batch of images from a generator model."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = batch.to(device)
        output = model(batch)
        # Denormalize the output for visualization
        output = output * 0.5 + 0.5
        grid = make_grid(output, nrow=4, padding=2, normalize=True)
        np_grid = grid.cpu().numpy().transpose((1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(np_grid)
        plt.title(title)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained Generator model (adjust path as needed)
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
    netG.load_state_dict(torch.load("path/to/generator.pth", map_location=device))
    
    # Assuming viton_loader is defined elsewhere
    generate_samples(viton_loader, netG, device, title="Domain X -> Y Samples (G)")
