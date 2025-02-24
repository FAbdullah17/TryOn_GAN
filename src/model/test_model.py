import torch
from generator import Generator
from discriminator import Discriminator

if __name__ == "__main__":
    # Instantiate models
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
    netD = Discriminator(input_nc=3)

    # Create a dummy input image tensor
    x = torch.randn(1, 3, 256, 256)

    # Get outputs from the Generator and Discriminator
    gen_output = netG(x)
    disc_output = netD(x)

    print("Generator output shape:", gen_output.shape)
    print("Discriminator output shape:", disc_output.shape)
