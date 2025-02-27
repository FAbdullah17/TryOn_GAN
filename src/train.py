import torch
import torch.nn as nn
import torch.optim as optim

# Mean Squared Error loss for GANs (Least Squares GAN)
criterion_GAN = nn.MSELoss()
# L1 Loss for cycle consistency and identity mapping
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Loss weighting factors
lambda_cycle = 10.0
lambda_idt = 5.0

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)  # G: X -> Y
netF = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)  # F: Y -> X
netD_Y = Discriminator(input_nc=3).to(device)  # Discriminator for domain Y
netD_X = Discriminator(input_nc=3).to(device)  # Discriminator for domain X

# Define optimizers
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(list(netG.parameters()) + list(netF.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_Y = optim.Adam(netD_Y.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_X = optim.Adam(netD_X.parameters(), lr=lr, betas=(beta1, 0.999))

# Number of training epochs
num_epochs = 50  

# Training loop
for epoch in range(num_epochs):
    for i, (data_X, data_Y) in enumerate(zip(loader_viton, loader_deepfashion)):
        # Set models to training mode
        netG.train()
        netF.train()
        netD_Y.train()
        netD_X.train()

        # Ensure images are converted to tensors and sent to the correct device
        real_X = data_X[0].to(device) if isinstance(data_X, (tuple, list)) else data_X.to(device)
        real_Y = data_Y[0].to(device) if isinstance(data_Y, (tuple, list)) else data_Y.to(device)

        # ------------------ Generators (G & F) ------------------

        # Identity loss: G(Y) ≈ Y, F(X) ≈ X
        idt_Y = netG(real_Y)
        loss_idt_Y = criterion_identity(idt_Y, real_Y) * lambda_idt
        
        idt_X = netF(real_X)
        loss_idt_X = criterion_identity(idt_X, real_X) * lambda_idt

        # GAN loss: Fake images should fool discriminators
        fake_Y = netG(real_X)
        pred_fake_Y = netD_Y(fake_Y)
        valid_Y = torch.ones_like(pred_fake_Y).to(device)
        loss_GAN_G = criterion_GAN(pred_fake_Y, valid_Y)

        fake_X = netF(real_Y)
        pred_fake_X = netD_X(fake_X)
        valid_X = torch.ones_like(pred_fake_X).to(device)
        loss_GAN_F = criterion_GAN(pred_fake_X, valid_X)

        # Cycle consistency loss: Reconstructed images should match original images
        rec_X = netF(fake_Y)
        loss_cycle_X = criterion_cycle(rec_X, real_X)

        rec_Y = netG(fake_X)
        loss_cycle_Y = criterion_cycle(rec_Y, real_Y)

        loss_cycle = (loss_cycle_X + loss_cycle_Y) * lambda_cycle

        # Total loss for generators
        loss_G = loss_GAN_G + loss_GAN_F + loss_cycle + loss_idt_Y + loss_idt_X

        # Update generators
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ------------------ Discriminator D_Y ------------------
        pred_real_Y = netD_Y(real_Y)
        loss_D_Y_real = criterion_GAN(pred_real_Y, valid_Y)

        pred_fake_Y = netD_Y(fake_Y.detach())
        fake_Y_label = torch.zeros_like(pred_fake_Y).to(device)
        loss_D_Y_fake = criterion_GAN(pred_fake_Y, fake_Y_label)

        loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5

        # Update discriminator D_Y
        optimizer_D_Y.zero_grad()
        loss_D_Y.backward()
        optimizer_D_Y.step()

        # ------------------ Discriminator D_X ------------------
        pred_real_X = netD_X(real_X)
        loss_D_X_real = criterion_GAN(pred_real_X, valid_X)

        pred_fake_X = netD_X(fake_X.detach())
        fake_X_label = torch.zeros_like(pred_fake_X).to(device)
        loss_D_X_fake = criterion_GAN(pred_fake_X, fake_X_label)

        loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5

        # Update discriminator D_X
        optimizer_D_X.zero_grad()
        loss_D_X.backward()
        optimizer_D_X.step()

        # Print progress every 50 batches
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}] "
                  f"Loss_G: {loss_G.item():.4f} "
                  f"Loss_D_Y: {loss_D_Y.item():.4f} Loss_D_X: {loss_D_X.item():.4f}")