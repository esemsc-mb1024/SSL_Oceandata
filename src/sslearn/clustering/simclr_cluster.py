import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import euclidean_distances


# 1) Feature extraction --------------------------------------------------------
def extract_features(encoder, train_loader):
    encoder.eval()
    feats_all, originals_all = [], []
    with torch.no_grad():
        for x_i, _, originals in tqdm(train_loader, desc="Extracting features"):
            x_i = x_i.to(device)
            z = encoder(x_i)          # shape [B, D]
            feats_all.append(z.cpu())
            originals_all.extend(originals)
    features = torch.cat(feats_all, dim=0).numpy()  # (N, D)
    return features, originals_all                   # list of images/arrays

# Use your encoder and loader:
# features, originals = extract_features(encoder, train_loader)

# 2) Scale + PCA ---------------------------------------------------------------
def pca_reduce(features, n_components=50):
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(features)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"Cumulative explained variance @{n_components} PCs: {cumvar[-1]:.4f}")
    return X_pca

# X_pca = pca_reduce(features, n_components=n_pca)

# 3) Choose k with elbow -------------------------------------------------------
# best_k, ks, inertias = elbow_kmeans(X_pca, k_min=2, k_max=40, sample_size=None)
# print("Chosen k:", best_k)

# 4) KMeans clustering ---------------------------------------------------------
def cluster_features_kmeans(features, num_clusters, return_centers=False):
    Xn = normalize(features, axis=1)  # optional but often helpful
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    cluster_ids = kmeans.fit_predict(Xn)
    if return_centers:
        # centers are in the normalized feature space
        return cluster_ids, kmeans.cluster_centers_
    return cluster_ids

# cluster_ids, centers = cluster_features_kmeans(X_pca, best_k, return_centers=True)

# 5) t-SNE visualization -------------------------------------------------------
def visualize_clusters(features_2d, cluster_ids, title="t-SNE Clustering"):
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_ids, cmap="tab10", s=5)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def tsne_embed(features, perplexity=30, random_state=0):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    return tsne.fit_transform(features)

# embedded = tsne_embed(X_pca, perplexity=30)
# visualize_clusters(embedded, cluster_ids)

# 6) Show top images per cluster ----------------------------------------------
def show_top_cluster_images(reduced_features, cluster_ids, cluster_centers, original_images, top_k=25, method_name="kmeans"):
    unique_clusters = np.unique(cluster_ids)
    for cid in unique_clusters:
        idx = np.where(cluster_ids == cid)[0]
        feats = reduced_features[idx]
        center = cluster_centers[cid]

        dists = euclidean_distances(feats, center.reshape(1, -1)).ravel()
        top_idx = idx[np.argsort(dists)[:top_k]]

        cols = int(np.sqrt(top_k))
        rows = int(np.ceil(top_k / cols))
        plt.figure(figsize=(12, 12))
        for i, j in enumerate(top_idx):
            img = original_images[j]
            if hasattr(img, "numpy"):
                img = img.squeeze().numpy()
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
        plt.suptitle(f"{method_name.upper()} Cluster {cid}", fontsize=14)
        plt.tight_layout()
        plt.show()
