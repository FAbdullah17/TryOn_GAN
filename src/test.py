import os
import torch
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model.generator import Generator
from data.dataset import UnpairedImageDataset
from utils.visualization import denormalize

def load_checkpoint(checkpoint_path, device):
    """Loads the pretrained CycleGAN generator model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
    netG.load_state_dict(checkpoint["netG_state_dict"])
    netG.eval()
    return netG

def save_generated_images(input_tensor, output_tensor, output_dir, image_name):
    """Saves the generated images after denormalization."""
    os.makedirs(output_dir, exist_ok=True)
    input_image_path = os.path.join(output_dir, f"{image_name}_input.png")
    output_image_path = os.path.join(output_dir, f"{image_name}_output.png")
    save_image(denormalize(input_tensor), input_image_path)
    save_image(denormalize(output_tensor), output_image_path)

def test_model(dataloader, netG, output_dir, device):
    """Performs inference on the test dataset and saves the generated images."""
    print(f"Starting model evaluation... Output images will be saved in: {output_dir}")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_image = data["image"].to(device)
            output_image = netG(input_image)

            # Save the images
            save_generated_images(input_image, output_image, output_dir, f"sample_{i+1}")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} images...")

    print("Testing completed. All images saved.")

def main():
    parser = argparse.ArgumentParser(description="CycleGAN Model Testing")
    parser.add_argument("--checkpoint", type=str, default="pretrained model outputs/CycleGan.sth", help="Path to the generator checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results", help="Path to save the generated images")
    parser.add_argument("--image_size", type=int, default=256, help="Image resizing dimensions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = load_checkpoint(args.checkpoint, device)

    test_dataset = UnpairedImageDataset(args.data_dir, transform_size=args.image_size, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_model(test_loader, netG, args.output_dir, device)

if __name__ == "__main__":
    main()
