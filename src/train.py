import torch
import torch.nn as nn
import torch.optim as optim
from model.generator import Generator
from model.discriminator import Discriminator

def train(loader_viton, loader_deepfashion, num_epochs=50, lr=0.0002, beta1=0.5, lambda_cycle=10.0, lambda_idt=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)  # G: X -> Y
    netF = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)  # F: Y -> X
    netD_Y = Discriminator(input_nc=3).to(device)  # Discriminator for domain Y
    netD_X = Discriminator(input_nc=3).to(device)  # Discriminator for domain X
    
    # Define losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Define optimizers
    optimizer_G = optim.Adam(list(netG.parameters()) + list(netF.parameters()), lr=lr, betas=(beta1, 0.999))
    optimizer_D_Y = optim.Adam(netD_Y.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_X = optim.Adam(netD_X.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (data_X, data_Y) in enumerate(zip(loader_viton, loader_deepfashion)):
            real_X = data_X[0].to(device) if isinstance(data_X, (tuple, list)) else data_X.to(device)
            real_Y = data_Y[0].to(device) if isinstance(data_Y, (tuple, list)) else data_Y.to(device)

            # Generators
            idt_Y = netG(real_Y)
            loss_idt_Y = criterion_identity(idt_Y, real_Y) * lambda_idt
            
            idt_X = netF(real_X)
            loss_idt_X = criterion_identity(idt_X, real_X) * lambda_idt

            fake_Y = netG(real_X)
            fake_X = netF(real_Y)
            
            loss_GAN_G = criterion_GAN(netD_Y(fake_Y), torch.ones_like(netD_Y(fake_Y)).to(device))
            loss_GAN_F = criterion_GAN(netD_X(fake_X), torch.ones_like(netD_X(fake_X)).to(device))

            loss_cycle = (criterion_cycle(netF(fake_Y), real_X) + criterion_cycle(netG(fake_X), real_Y)) * lambda_cycle

            loss_G = loss_GAN_G + loss_GAN_F + loss_cycle + loss_idt_Y + loss_idt_X
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Discriminators
            loss_D_Y = 0.5 * (criterion_GAN(netD_Y(real_Y), torch.ones_like(netD_Y(real_Y)).to(device)) +
                              criterion_GAN(netD_Y(fake_Y.detach()), torch.zeros_like(netD_Y(fake_Y)).to(device)))
            optimizer_D_Y.zero_grad()
            loss_D_Y.backward()
            optimizer_D_Y.step()
            
            loss_D_X = 0.5 * (criterion_GAN(netD_X(real_X), torch.ones_like(netD_X(real_X)).to(device)) +
                              criterion_GAN(netD_X(fake_X.detach()), torch.zeros_like(netD_X(fake_X)).to(device)))
            optimizer_D_X.zero_grad()
            loss_D_X.backward()
            optimizer_D_X.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}] "
                      f"Loss_G: {loss_G.item():.4f} "
                      f"Loss_D_Y: {loss_D_Y.item():.4f} Loss_D_X: {loss_D_X.item():.4f}")

if __name__ == "__main__":
    loader_viton = [] 
    loader_deepfashion = []
    train(loader_viton, loader_deepfashion)
