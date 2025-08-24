import torch
from torch.utils.data import DataLoader
from models.simclr import SimCLR, simple_cnn
from datasets.tiff_dataset import TiffDataset
from scripts.extract_features import extract_features
from scripts.train_classifier import train_logistic_regression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimCLR(simple_cnn()).to(device)
model.load_state_dict(torch.load("path/to/checkpoint.pt", map_location=device))
model.eval()

# Load dataset
dataset = TiffDataset("data/TIFF_processed_1/GeoTIFF/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Feature extraction
features, labels = extract_features(model, dataloader, device)

# Classification
train_logistic_regression(features, labels)
