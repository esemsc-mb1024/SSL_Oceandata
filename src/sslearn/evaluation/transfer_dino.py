import torch
from torch.utils.data import DataLoader

# Model
from models.dino_arc import VisionTransformer

# Dataset
from datasets.tiff_dataset import TiffDataset

# Feature extraction
from sslearn.features.transfer_features import extract_features_dino

# Linear probe utilities
from sslearn.classical.logistic_regression import train_logistic_regression

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ----------------------------
# Load DINO student model
# ----------------------------
student = VisionTransformer(out_dim=2048).to(device)

# NOTE: weights are not included. To use a trained student:
# state = torch.load("path/to/dino_student_checkpoint.pt", map_location=device)
# student.load_state_dict(state)

student.eval()  # assumes student has been trained

# ----------------------------
# Load dataset
# ----------------------------
dataset = TiffDataset("Path/to/processed/GeoTIFF/")   # <-- update this path
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# Feature extraction (CLS embeddings from student)
# ----------------------------
features, labels = extract_features_dino(student, dataloader, device)

# ----------------------------
# Linear probe (train/eval)
# ----------------------------
train_logistic_regression(features, labels)
