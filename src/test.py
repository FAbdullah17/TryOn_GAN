import os
import torch
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from model.generator import Generator
from data_preprocessing import viton_path, deepfashion_path
from utils.visualization import denormalize

def load_checkpoint(checkpoint_path, device):
    """Loads the pretrained Generator model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG = Generator().to(device)
    netG.load_state_dict(checkpoint)
    netG.eval()
    return netG

def save_generated_images(input_tensor, output_tensor, output_dir, image_name):
    """Saves the generated images after denormalization."""
    os.makedirs(output_dir, exist_ok=True)
    save_image(denormalize(input_tensor), os.path.join(output_dir, f"{image_name}_input.png"))
    save_image(denormalize(output_tensor), os.path.join(output_dir, f"{image_name}_output.png"))

def test_model(dataloader, netG, output_dir, device):
    """Performs inference on the test dataset and saves the generated images."""
    print(f"Starting model evaluation... Output images will be saved in: {output_dir}")

    with torch.no_grad():
        for i, (person_img, clothing_img, keypoints) in enumerate(dataloader):
            person_img, clothing_img, keypoints = person_img.to(device), clothing_img.to(device), keypoints.to(device)
            
            # Warping and Parsing should be applied before feeding into the generator
            generated_image = netG(person_img, clothing_img)  # Ensure correct input format

            # Save the images
            save_generated_images(person_img, generated_image, output_dir, f"sample_{i+1}")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} images...")

    print("Testing completed. All images saved.")

def main():
    parser = argparse.ArgumentParser(description="Virtual Try-On Model Testing")
    parser.add_argument("--checkpoint", type=str, default="outputs/generator.pth", help="Path to the trained generator checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results", help="Path to save generated images")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = load_checkpoint(args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_dataset1 = viton_path(root_dir=args.data_dir, transform=transform)
    test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_dataset2 = deepfashion_path(root_dir=args.data_dir, transform=transform)
    test_loader2 = DataLoader(test_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_model(test_loader1, test_loader2, netG, args.output_dir, device)

if __name__ == "__main__":
    main()
