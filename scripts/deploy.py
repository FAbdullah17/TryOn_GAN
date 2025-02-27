import io
import os
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from model.generator import Generator

app = FastAPI()

# Mount static files (HTML, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Set up device (use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG = Generator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
    netG.load_state_dict(checkpoint["netG_state_dict"])
    netG.eval()
    return netG

# Load the pretrained generator model from the checkpoint
model = load_model("outputs/CycleGan.pth")

def transform_image(image: Image.Image, image_size: int = 256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform_image(image).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_tensor = denormalize(output_tensor).squeeze(0)
    output_image = transforms.ToPILImage()(output_tensor.cpu())
    
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("scripts.deploy:app", host="0.0.0.0", port=8000, reload=True)
