# inference.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model_definitions import CNN, ResNetModel, PatchSelection, DIF, UFD
from utils import load_model, weighted_voting

# Constants
MODEL_PATH = "D:/AI_Image_Detection_WebApp/src/newly_trained_model"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["fake", "real"]  # Adjust if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all models
models = [
    load_model(CNN, "CNN", MODEL_PATH, DEVICE),
    load_model(ResNetModel, "ResNetModel", MODEL_PATH, DEVICE),
    load_model(PatchSelection, "PatchSelection", MODEL_PATH, DEVICE),
    load_model(DIF, "DIF", MODEL_PATH, DEVICE),
    load_model(UFD, "UFD", MODEL_PATH, DEVICE),
]

# Model weights (equal weights)
weights = [1.0] * len(models)

# Image transformation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = []
        for model in models:
            output = model(image_tensor)
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            preds.append(pred.item())

    final_prediction = weighted_voting(preds, weights, len(CLASS_NAMES))
    return CLASS_NAMES[final_prediction]

if __name__ == "__main__":
    # Replace with an actual test image path
    test_image_path = "D:\AI_Image_Detection_WebApp\sm1.jpeg"

    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
    else:
        prediction = predict_image(test_image_path)
        print(f"Prediction: {prediction}")
