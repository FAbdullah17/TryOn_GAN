import torch
import torchvision.transforms as transforms
from PIL import Image
from model.generator import Generator
from model.warping import WarpingModule
from model.parsing import ParsingModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load("outputs/model.pth", map_location=DEVICE))
generator.eval()

warping = WarpingModule().to(DEVICE).eval()
parsing = ParsingModule().to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def virtual_tryon(person_img_path, clothing_img_path, keypoints):
    person_img = transform(Image.open(person_img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    clothing_img = transform(Image.open(clothing_img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    keypoints = torch.tensor(keypoints).unsqueeze(0).to(DEVICE)

    parsed_human = parsing(person_img)
    warped_clothing = warping(clothing_img, keypoints)
    
    if parsed_human.shape[1] != 3:
        parsed_human = parsed_human.repeat(1, 3, 1, 1)  
    if warped_clothing.shape[1] != 3:
        warped_clothing = warped_clothing.repeat(1, 3, 1, 1)
    
    generated_img = generator(parsed_human, warped_clothing)
    generated_img = generated_img.squeeze(0).detach().cpu()
    generated_img = transforms.ToPILImage()(generated_img)
    
    return generated_img
