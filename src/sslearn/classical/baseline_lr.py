import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from datasets.tiff_dataset import TiffDataset


# Load dataset
dataset = TiffDataset("data/TIFF_processed_1/GeoTIFF/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Extract raw pixels + labels
X_parts, y_parts = [], []

for images, labels in dataloader:
    # Flatten: [B, 1, H, W] → [B, H*W]
    B = images.shape[0]
    images = images.view(B, -1)
    
    X_parts.append(images.numpy())
    y_parts.append(np.array(labels))

X = np.vstack(X_parts)  # [N, H*W]
y = np.concatenate(y_parts)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train logistic regression
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_scaled, y_train)


# Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"📉 Baseline (raw pixels, no SSL) Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))
