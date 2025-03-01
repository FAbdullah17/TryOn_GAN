from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.generator import Generator
from model.warping import WarpingModule
from model.parsing import ParsingModule

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
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
    """Runs virtual try-on using the generator model."""
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

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the interactive HTML page."""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/tryon/")
async def tryon(person_img: UploadFile = File(...), clothing_img: UploadFile = File(...)):
    """Receives images, applies virtual try-on, returns output image."""
    person_path = os.path.join(UPLOAD_FOLDER, person_img.filename)
    clothing_path = os.path.join(UPLOAD_FOLDER, clothing_img.filename)

    with open(person_path, "wb") as buffer:
        shutil.copyfileobj(person_img.file, buffer)

    with open(clothing_path, "wb") as buffer:
        shutil.copyfileobj(clothing_img.file, buffer)

    keypoints = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    output_img = virtual_tryon(person_path, clothing_path, keypoints)
    output_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    output_img.save(output_path)

    return FileResponse(output_path, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
