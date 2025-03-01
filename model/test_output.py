import torch
from warping import WarpingModule
from parsing import ParsingModule
from generator import Generator
from discriminator import Discriminator

dummy_person = torch.randn(1, 3, 256, 256)  
dummy_clothing = torch.randn(1, 3, 256, 256)
dummy_keypoints = torch.randn(1, 8)

warping = WarpingModule()
parsing = ParsingModule()
generator = Generator()
discriminator = Discriminator()

parsed_human = parsing(dummy_person)
warped_clothing = warping(dummy_clothing, dummy_keypoints)
generated_img = generator(parsed_human, warped_clothing)
disc_output = discriminator(generated_img)

print("Parsing Output Shape:", parsed_human.shape)
print("Warping Output Shape:", warped_clothing.shape)
print("Generator Output Shape:", generated_img.shape)
print("Discriminator Output Shape:", disc_output.shape)
