import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clusters = 15
n_pca = 50


# === STEP 1: Extract Features from SimCLR encoder ===
def extract_features(encoder, train_loader):
    encoder.eval()
    all_features = []
    all_originals = []
    with torch.no_grad():
        for x_i, _, originals in tqdm(train_loader, desc="Extracting features"):
            x_i = x_i.to(device)
            feats = encoder(x_i)
            all_features.append(feats.cpu())
            all_originals.extend(originals)  # keep unaugmented images
    return torch.cat(all_features, dim=0).numpy(), all_originals

# === STEP 2: (PCA) ===
X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(features)

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative explained variance @50 PCs: {cumvar[-1]:.4f}")

# (Optional) quick plot
plt.figure()
plt.plot(np.arange(1, 51), cumvar, marker='o')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative explained variance')
plt.title('PCA cumulative explained variance')
plt.tight_layout()
plt.show()

# === STEP 3: Dimensionality Reduction (PCA) ===
def reduce_dimensionality(features, n_pca=50):
    features = features[~np.isnan(features).any(axis=1)]
    pca = PCA(n_components=n_pca)
    return pca.fit_transform(features)


# After you have X_pca from your reduce_dimensionality(...)
best_k, ks, inertias = elbow_kmeans(X_pca, k_min=2, k_max=40, sample_size=None)
print("Chosen k:", best_k)


# === STEP 3: KMeans Clustering ===
def cluster_features_kmeans(features, num_clusters, return_centers=False):
    features = normalize(features, axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(features)
    if return_centers:
        return cluster_ids, kmeans.cluster_centers_
    return cluster_ids


# === STEP 4: t-SNE Visualization ===
def visualize_clusters(features, cluster_ids, title="t-SNE Clustering"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    embedded = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedded[:, 0], embedded[:, 1], c=cluster_ids, cmap='tab10', s=5)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# === STEP 5: Show top images per cluster ===
def show_top_cluster_images(reduced_features, cluster_ids, cluster_centers, original_images, method_name="kmeans"):
    unique_clusters = np.unique(cluster_ids)

    for cluster_id in unique_clusters:
        members_idx = np.where(cluster_ids == cluster_id)[0]
        member_features = reduced_features[members_idx]
        center = cluster_centers[cluster_id]

        dists = euclidean_distances(member_features, center.reshape(1, -1)).flatten()
        closest_idx = members_idx[np.argsort(dists)[:25]]

        plt.figure(figsize=(12, 12))
        for i, img_idx in enumerate(closest_idx):
            img = original_images[img_idx]
            if hasattr(img, "numpy"):
                img = img.squeeze().numpy()
            plt.subplot(5, 5, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"C{cluster_id}", fontsize=8)
            plt.axis("off")
        plt.suptitle(f"{method_name.upper()} Cluster {cluster_id}", fontsize=14)
        plt.tight_layout()
        plt.show()