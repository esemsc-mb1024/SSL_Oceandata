import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dino import VisionTransformer
from datasets.dataset import InMemoryDinoDataset
from datasets.transforms import DinoMultiCropTransform
from utils.collate import dino_collate_fn

# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_clusters = 15
pca_dim = 50
checkpoint_path = "checkpoints/epoch_50_loss_0.2792.pt"
array_path = "final_stack.pt"

# ----------------------------
# Load model and weights
# ----------------------------
student = VisionTransformer(out_dim=256).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
student.load_state_dict(checkpoint["student_state_dict"])
student.eval()

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(array_path):
    raise FileNotFoundError(f"{array_path} not found")
array_stack = torch.load(array_path)

transform = DinoMultiCropTransform()
dataset = InMemoryDinoDataset(array_stack, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dino_collate_fn)

# ----------------------------
# Feature extraction
# ----------------------------
all_features = []
all_originals = []

with torch.no_grad():
    for crops, originals in tqdm(dataloader, desc="Extracting features"):
        originals = originals.to(device)
        features = student(originals)  # [B, N, D]
        all_features.append(features.cpu())
        all_originals.append(originals.cpu())

all_features = torch.cat(all_features, dim=0)  # [B, N, D]
all_originals = torch.cat(all_originals, dim=0)  # [B, 1, H, W]

# Reduce patch tokens to mean vector
features = all_features.mean(dim=1).numpy()  # [B, D]
originals = all_originals.mean(dim=1).numpy()  # [B, H, W]

# ----------------------------
# Dimensionality Reduction
# ----------------------------
def reduce_dimensionality(features, n_pca=50):
    features = features[~np.isnan(features).any(axis=1)]
    pca = PCA(n_components=n_pca)
    return pca.fit_transform(features)

reduced_features = reduce_dimensionality(features, n_pca=pca_dim)

# ----------------------------
# Clustering (KMeans only)
# ----------------------------
def cluster_features_cosine(features, num_clusters):
    normed_features = normalize(features, norm="l2", axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_ids = kmeans.fit_predict(normed_features)
    return cluster_ids, kmeans.cluster_centers_

cluster_ids, cluster_centers = cluster_features_cosine(reduced_features, num_clusters=num_clusters)

# ----------------------------
# Visualization
# ----------------------------
def visualize_clusters(features, cluster_ids, title="t-SNE Clustering"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    embedded = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=cluster_ids, cmap='tab10', s=5)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

visualize_clusters(reduced_features, cluster_ids, title="t-SNE Clustering (KMeans)")

# ----------------------------
# Visualisation
# ----------------------------
def show_top_cluster_images(reduced_features, cluster_ids, cluster_centers, original_images, method_name="kmeans", top_k=5):
    print(f"Showing top-{top_k} images per cluster for {method_name.upper()}")

    from scipy.spatial.distance import cdist
    distances = cdist(reduced_features, cluster_centers)

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_distances = distances[cluster_mask, cluster_id]
        top_indices = cluster_indices[np.argsort(cluster_distances)[:top_k]]

        fig, axes = plt.subplots(1, top_k, figsize=(12, 3))
        for i, idx in enumerate(top_indices):
            axes[i].imshow(original_images[idx], cmap="gray")
            axes[i].axis("off")
        plt.suptitle(f"Cluster {cluster_id} ({method_name.upper()})")
        plt.show()

show_top_cluster_images(reduced_features, cluster_ids, cluster_centers, originals, method_name="kmeans")
