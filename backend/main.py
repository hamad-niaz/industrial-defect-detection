import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException


# -----------------------------
# 1. Basic configuration
# -----------------------------

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Detects industrial bottle defects using a PatchCore-like model.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "patchcore_final.pth"

device = torch.device("cpu")

# Demo threshold. This is provisional, not a production-validated threshold.
ANOMALY_THRESHOLD = 2.85


# -----------------------------
# 2. Feature extractor
# -----------------------------

class FeatureExtractor(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)

        features_layer2 = self.layer2(x)
        features_layer3 = self.layer3(features_layer2)

        return features_layer2, features_layer3


# -----------------------------
# 3. Load saved model
# -----------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

backbone = models.resnet18(weights=None)

feature_extractor = FeatureExtractor(backbone)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

feature_extractor.load_state_dict(checkpoint["feature_extractor"])
feature_extractor.to(device)
feature_extractor.eval()

memory_bank = checkpoint["memory_bank"].float().to(device)

print("Model loaded successfully!")
print("Memory bank shape:", memory_bank.shape)
print("Device:", device)


# -----------------------------
# 4. Image preprocessing
# -----------------------------

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_tensor = (image_tensor - mean) / std

    return image_tensor.unsqueeze(0).to(device)


# -----------------------------
# 5. Generate heatmap
# -----------------------------

def create_heatmap(anomaly_map):

    heatmap = anomaly_map.squeeze().cpu().numpy()

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    heatmap = (heatmap - heatmap_min) / (
        heatmap_max - heatmap_min + 1e-8
    )

    # Convert grayscale intensity into a simple RGB heatmap.
    red = np.clip(heatmap * 3, 0, 1)
    green = np.clip(heatmap * 3 - 1, 0, 1)
    blue = np.clip(heatmap * 3 - 2, 0, 1)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    heatmap_image = Image.fromarray(rgb)

    buffer = io.BytesIO()
    heatmap_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------
# 6. Anomaly detection
# -----------------------------

def detect_anomaly(image):

    input_tensor = preprocess_image(image)

    with torch.no_grad():

        features_layer2, features_layer3 = feature_extractor(
            input_tensor
        )

        features_layer3 = F.interpolate(
            features_layer3,
            size=features_layer2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        combined_features = torch.cat(
            [features_layer2, features_layer3],
            dim=1
        )

        patches = combined_features.permute(
            0, 2, 3, 1
        ).reshape(-1, 384)

        distances = torch.cdist(patches, memory_bank)

        nearest_distances = distances.min(dim=1).values

        anomaly_map = nearest_distances.reshape(1, 1, 28, 28)

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        anomaly_score = torch.quantile(
            nearest_distances,
            0.99
        ).item()

    heatmap_base64 = create_heatmap(anomaly_map)

    prediction = (
        "DEFECTIVE"
        if anomaly_score > ANOMALY_THRESHOLD
        else "NORMAL"
    )

    return {
        "anomaly_score": round(anomaly_score, 4),
        "prediction": prediction,
        "threshold": ANOMALY_THRESHOLD,
        "threshold_is_provisional": True,
        "heatmap_base64": heatmap_base64
    }


# -----------------------------
# 7. API endpoints
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Industrial Defect Detection API is running!",
        "model": "PatchCore-like ResNet18",
        "status": "active"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = detect_anomaly(image)

        return {
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )