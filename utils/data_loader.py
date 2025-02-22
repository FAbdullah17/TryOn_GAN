import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch

class SingleFolderDataset(Dataset):
    """
    A custom Dataset for a folder that may contain images inside subdirectories.
    This is useful for datasets like the High-Resolution VITON-Zalando Dataset.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        # Recursively find all image files in subdirectories
        self.files = sorted([
            os.path.join(root, f) for root, _, files in os.walk(folder)
            for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {folder}. Check dataset structure.")
        
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(dataset_path, image_size=256, batch_size=32, num_workers=4, augment=False, folder_type="imagefolder"):
    """
    Returns a PyTorch DataLoader for a dataset of images.
    
    Parameters:
        dataset_path (str): Path to the dataset folder.
        image_size (int): The target height and width for images.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of worker threads for loading data.
        augment (bool): Whether to apply data augmentation.
        folder_type (str): Specifies dataset structure:
                           "imagefolder" - images are in subdirectories (e.g., DeepFashion)
                           "single" - images are scattered in a folder/subfolders (e.g., VITON-Zalando)
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    if augment:
        data_transforms = transforms.Compose([
            transforms.Resize(int(image_size * 1.2)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    if folder_type == "imagefolder":
        dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
    elif folder_type == "single":
        dataset = SingleFolderDataset(folder=dataset_path, transform=data_transforms)
    else:
        raise ValueError("Unsupported folder_type. Use 'imagefolder' or 'single'.")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

if __name__ == "__main__":
    viton_path = "/kaggle/input/high-resolution-viton-zalando-dataset/test/image-densepose"  # Adjusted to actual image folder
    loader_viton = get_data_loader(viton_path, image_size=256, batch_size=8, num_workers=2, augment=True, folder_type="single")
    
    deepfashion_path = "/kaggle/input/deepfashion-dataset/Deepfashion_dataset/img"  # Adjusted to actual image folder
    loader_deepfashion = get_data_loader(deepfashion_path, image_size=256, batch_size=8, num_workers=2, augment=True, folder_type="imagefolder")
    
    for images in loader_viton:
        print("VITON batch shape:", images.shape)
        break

    for images, _ in loader_deepfashion:
        print("DeepFashion batch shape:", images.shape)
        break
    