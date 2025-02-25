import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.data_loader import get_data_loader,viton_path,deepfashion_path,visualize_batch
from PIL import Image
import os

# Define image transformations (including augmentation)
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply slight color changes
    transforms.RandomRotation(degrees=15),  # Rotate images randomly up to ±15 degrees
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images (-1 to 1 range)
])

# Function to clean dataset by removing corrupted images
def clean_dataset(folder_path):
    """Removes corrupted images that cannot be opened."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if the image is valid
            except (IOError, SyntaxError):
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

# Apply dataset cleaning
clean_dataset("/kaggle/input/high-resolution-viton-zalando-dataset")
clean_dataset("/kaggle/input/deepfashion-dataset/Deepfashion_dataset/img")

# Reload DataLoaders with preprocessing transformations
class PreprocessedDataset(ImageFolder):
    """Custom dataset loader with preprocessing transformations."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

# Create data loaders
preprocessed_viton_loader = DataLoader(
    PreprocessedDataset("/kaggle/input/high-resolution-viton-zalando-dataset/train", transform=data_transforms),
    batch_size=8, shuffle=True, num_workers=2
)

preprocessed_deepfashion_loader = DataLoader(
    PreprocessedDataset("/kaggle/input/deepfashion-dataset/Deepfashion_dataset/img", transform=data_transforms),
    batch_size=8, shuffle=True, num_workers=2
)

# Verify preprocessing by loading and displaying a batch
def visualize_preprocessed_batch(data_loader, title):
    """Visualizes a batch of preprocessed images."""
    batch = next(iter(data_loader))
    images, _ = batch
    visualize_batch(images, title=title)

visualize_preprocessed_batch(preprocessed_viton_loader, "VITON-Zalando Preprocessed Samples")
visualize_preprocessed_batch(preprocessed_deepfashion_loader, "DeepFashion Preprocessed Samples")
