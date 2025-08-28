import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sslearn.preprocessing.tiff_ds import TiffDataset

"""
Transfer Learning with Pretrained ResNet18
------------------------------------------

This script evaluates ImageNet-pretrained ResNet18 features on the
Sentinel-1 WV GeoTIFF dataset using a linear logistic regression probe
"""

 # --- Data ---
dataset = TiffDataset("Path/to/processed/GeoTIFF/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Imagenet-pretrained ResNet18 encoder ---
try:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception:
    resnet = models.resnet18(pretrained=True)
encoder = nn.Sequential(*list(resnet.children())[:-1]).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)

# Imagenet normalization & channel handling
def imagenet_preprocess(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    return (x - mean) / std

# --- Feature extraction ---
features_list, labels_list = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True).float()
        images = imagenet_preprocess(images)
        feats = encoder(images)                 # (N, 512, 1, 1)
        feats = torch.flatten(feats, 1).cpu()   # (N, 512)
        features_list.append(feats)
        labels_list.append(labels.cpu())

X = torch.cat(features_list).numpy()
y = torch.cat(labels_list).numpy()

# --- Logistic regression probe ---
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=4))