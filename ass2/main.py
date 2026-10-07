from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from cnn import CIFARClassifier


app = FastAPI()


# CIFAR-10 class names
classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


# device
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# model
model = CIFARClassifier().to(device)


# trained checkpoint
checkpoint = torch.load(
    "checkpoints_cnn/best/fcnn_epoch_010.pth",
    map_location = device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 CNN classifier API is running"
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # transform image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # inference
    with torch.inference_mode():

        output = model(image_tensor)
        probabilities = torch.softmax(output, dim = 1)
        predicted_class = torch.argmax(probabilities, dim = 1).item()

        confidence = probabilities[0, predicted_class].item()

    return {
        "class_id": predicted_class,
        "class_name": classes[predicted_class],
        "confidence": confidence
    }