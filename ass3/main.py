from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from torchvision.transforms import ToPILImage
from gan import Generator


app = FastAPI()

# device
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

z_dim = 100
gen = Generator(z_dim).to(device)

checkpoint = torch.load("checkpoints/generator_final.pth", map_location = device)

gen.load_state_dict(checkpoint)

gen.eval()

@app.get("/")
def root():
    return {
        "message": "MNIST GAN API is running"
    }

@app.get("/generate")
def generate():

    noise = torch.randn(1, z_dim).to(device)

    with torch.no_grad():
        generated_image = gen(noise)

    generated_image = generated_image.squeeze(0).cpu()

    # since Generator ends up with Tanh, which output in range [-1, 1], covert to [0, 1] for displaying
    generated_image = (generated_image + 1) / 2
    generated_image = generated_image.clamp(0, 1)

    image = ToPILImage()(generated_image)
    buffer = io.BytesIO()

    image.save(buffer, format = "PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type = "image/png")