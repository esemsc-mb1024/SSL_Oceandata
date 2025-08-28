import torch
from torch.utils.data import DataLoader
from models.simclr_arc import SimCLR, simple_cnn
from datasets.tiff_dataset import TiffDataset
from sslearn.features.transfer_features import extract_features_Sim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sslearn.classical.logistic_regression import train_logistic_regression, plot_confusion_matrix,  label_efficiency_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model
# ----------------------------
model = SimCLR(simple_cnn()).to(device)

# NOTE:
# Model weights are not included in the repository.
# You can incorporate them by uncommenting and pointing to your checkpoint:
# model.load_state_dict(torch.load("path/to/checkpoint.pt", map_location=device))

model.eval() #assumes model has been trained

# Load dataset
dataset = TiffDataset("Path/to/processed/GeoTIFF/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ----------------------------
# Feature extraction
# ----------------------------
features, labels = extract_features_Sim(model, dataloader, device)

# ----------------------------
# 1) Train & evaluate linear probe
# ----------------------------
clf, scaler, test_tuple = train_logistic_regression(features, labels)

# ----------------------------
# 2) Confusion matrices (counts + normalized)
# ----------------------------
os.makedirs("plots", exist_ok=True)
plot_confusion_matrix(test_tuple, labels, class_names=None, out_prefix="plots/logreg")

# ----------------------------
# 3) Label-efficiency curve
# ----------------------------
label_efficiency_curve(
    features,
    labels,
    fractions=(0.01, 0.05, 0.10, 0.25, 0.50, 1.00),
    seeds=(0, 1, 2, 3, 4),
    test_size=0.20,
    random_state=42,
    title="Label-efficiency (SimCLR linear probe)",
    out_file="plots/label_efficiency.png",
    max_iter=5000,
    verbose=True,
)