import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_preprocessing import viton_path, deepfashion_path, visualize_batch, get_data_loader, VirtualTryOnDataset
from model.warping import WarpingModule
from model.parsing import ParsingModule
from model.generator import Generator
from model.discriminator import Discriminator

EPOCHS = 50
BATCH_SIZE = 16
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = VirtualTryOnDataset(root_dir="data", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

warping = WarpingModule().to(DEVICE)
parsing = ParsingModule().to(DEVICE)
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = torch.nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    for i, (person_img, clothing_img, keypoints) in enumerate(dataloader):
        person_img, clothing_img, keypoints = person_img.to(DEVICE), clothing_img.to(DEVICE), keypoints.to(DEVICE)

        parsed_human = parsing(person_img)
        warped_clothing = warping(clothing_img, keypoints)
        generated_image = generator(parsed_human, warped_clothing)

        real_labels = torch.ones((person_img.size(0), 1), device=DEVICE)
        fake_labels = torch.zeros((person_img.size(0), 1), device=DEVICE)

        disc_optimizer.zero_grad()
        real_loss = criterion(discriminator(person_img), real_labels)
        fake_loss = criterion(discriminator(generated_image.detach()), fake_labels)
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        disc_optimizer.step()

        gen_optimizer.zero_grad()
        gen_loss = criterion(discriminator(generated_image), real_labels)
        gen_loss.backward()
        gen_optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

print("Training complete!")

torch.save(generator.state_dict(), "outputs/generator.pth")
torch.save(discriminator.state_dict(), "outputs/discriminator.pth")
