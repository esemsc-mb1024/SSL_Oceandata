"""
run_clustering.py
-----------------
Example usage of the clustering pipeline.

Note:
- This script will not run without data (NumPy array) and trained model weights.
- Adjust paths to your dataset (.npy) and weight file (.pth).
"""

import numpy as np
import torch

from torch.utils.data import DataLoader


from sslearn.preprocessing.datasets import InMemoryContrastiveDataset
from sslearn.transforms.transforms import ModerateContrastiveAug
from sslearn.models.sim_arc import SimCLR, SimpleCNN
from sslearn.clustering.simclr_cluster import (
    extract_features,
    pca_reduce,
    cluster_features_kmeans,
    tsne_embed,
    visualize_clusters,
    show_top_cluster_images,
)
from sslearn.classical.elbow_kmeans import elbow_kmeans

# ----------------------------
# Configuration 
# ----------------------------
data_path = "/path/to/sigma0_WV/processed"         # path to preprocessed dataset
# weights_path = "simclr_trained.pth"    path to trained weights not included in repository
batch_size = 128
n_pca = 50
k_min, k_max = 2, 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load dataset
# ----------------------------
X = np.load(data_path).astype("float32")
if X.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
    X = X[:, None, :, :]

dataset = InMemoryContrastiveDataset(X, transform=ModerateContrastiveAug())
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Load model
# ----------------------------
model = SimCLR(SimpleCNN()).to(device)
state = torch.load(weights_path, map_location=device) 
encoder = model.encoder

# state = torch.load(weights_path, map_location=device)
# encoder = model.encoder.load_state_dict(state)   # weights not included assumes model has been trained 
# ----------------------------
# Run clustering pipeline
# ----------------------------
features, originals = extract_features(encoder, loader, device)
X_pca = pca_reduce(features, n_components=n_pca)

best_k, _, _ = elbow_kmeans(X_pca, k_min=k_min, k_max=k_max)
print(f"Chosen k: {best_k}")

cluster_ids, centers = cluster_features_kmeans(X_pca, best_k, return_centers=True)

embedded = tsne_embed(X_pca)
visualize_clusters(embedded, cluster_ids)

show_top_cluster_images(X_pca, cluster_ids, centers, originals, top_k=15)
