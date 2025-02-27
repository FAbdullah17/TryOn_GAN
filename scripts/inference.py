import os
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from model.generator import Generator

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
    netG.load_state_dict(checkpoint["netG_state_dict"])
    netG.eval()
    return netG

def load_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def visualize_generated(input_tensor, output_tensor, title_input="Input", title_output="Output"):
    grid_input = make_grid(denormalize(input_tensor), nrow=1, padding=2, normalize=False)
    grid_output = make_grid(denormalize(output_tensor), nrow=1, padding=2, normalize=False)
    np_input = grid_input.cpu().numpy().transpose((1, 2, 0))
    np_output = grid_output.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np_input)
    plt.title(title_input)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(np_output)
    plt.title(title_output)
    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="CycleGAN Inference")
    parser.add_argument("--checkpoint", type=str, default="pretrained model outputs/CycleGan.sth", help="Path to checkpoint file")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for inference")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = load_checkpoint(args.checkpoint, device)
    input_image = load_image(args.image, args.image_size).to(device)
    
    with torch.no_grad():
        output_image = netG(input_image)
    
    visualize_generated(input_image, output_image, title_input="Test Input", title_output="Generated Output")

if __name__ == "__main__":
    main()
