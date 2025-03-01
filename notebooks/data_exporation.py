import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.data_loader import get_data_loader

def visualize_batch(batch, title="Sample Batch"):
    images = batch.numpy().transpose((0, 2, 3, 1))
    images = (images - images.min()) / (images.max() - images.min())
    grid = make_grid(torch.tensor(images).permute(0, 3, 1, 2), nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.show()

def explore_dataset(dataset_path, image_size=256, batch_size=8, num_workers=2, folder_type="single"):
    data_loader = get_data_loader(dataset_path, image_size, batch_size, num_workers, augment=True, folder_type=folder_type)
    total_images = len(data_loader.dataset)
    print(f"Total images in {dataset_path}: {total_images}")
    
    sample_batch, _ = next(iter(data_loader))
    print(f"Sample batch shape: {sample_batch.shape}")
    visualize_batch(sample_batch, title=f"Sample Images from {dataset_path}")

def main():
    viton_path = "/kaggle/input/high-resolution-viton-zalando-dataset"
    deepfashion_path = "/kaggle/input/deepfashion-dataset/Deepfashion_dataset/img"
    
    print("Exploring VITON-Zalando Dataset...")
    explore_dataset(viton_path, folder_type="single")
    
    print("Exploring DeepFashion Dataset...")
    explore_dataset(deepfashion_path, folder_type="imagefolder")

if __name__ == "__main__":
    main()