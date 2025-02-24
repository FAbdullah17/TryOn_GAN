import torch
import torch.nn as nn
from utils.data_loader import get_data_loader, viton_path, deepfashion_path, visualize_batch, visualize_sample
from data_preprocessing import data_transforms, PreprocessedDataset, clean_dataset, visualize_preprocessed_batch
# Step 1: Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

# Step 2: Generator Architecture
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Downsampling (2 layers)
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling (2 layers)
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        # Output layer
        model += [
            nn.Conv2d(in_features, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# Step 3: Discriminator Architecture (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_features = 64
        for _ in range(3):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features
        model += [nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# Step 4: Test the Models
if __name__ == "__main__":
    # Instantiate the Generator and Discriminator (assume RGB images)
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
    netD = Discriminator(input_nc=3)
    
    # Create a dummy input image tensor of shape [1, 3, 256, 256]
    x = torch.randn(1, 3, 256, 256)
    
    # Get outputs from the Generator and Discriminator
    gen_output = netG(x)
    disc_output = netD(x)
    
    print("Generator output shape:", gen_output.shape)
    print("Discriminator output shape:", disc_output.shape)