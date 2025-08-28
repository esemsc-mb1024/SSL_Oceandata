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
 """
    Run the encoder in eval mode across a DataLoader and collect embeddings.

    Parameters
    ----------
    encoder : 
        Trained feature encoder producing embeddings of shape [B, D].
    train_loader :
        Loader yielding tuples (x_i, _, originals), where `x_i` is a batch of
        tensors on which to compute embeddings, and `originals` are the
        corresponding original images used later for visualization.

    Returns
    -------
    features : np.ndarray, shape (N, D)
        Stacked embeddings for all samples.
    originals_all : list
        List of original images aligned with `features`.

    """
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
    """
    Standardize features and reduce dimensionality with PCA.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Input embeddings.
    n_components : int, optional (default=50)
        Number of principal components to retain.

    Returns
    -------
    X_pca : np.ndarray, shape (N, n_components)
        PCA-reduced features.

    Side Effects
    ------------
    Prints cumulative explained variance at `n_components`.
    """
    
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
        """
    Cluster features with k-means (on L2-normalized rows).

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Input feature vectors (e.g., PCA outputs).
    num_clusters : int
        Number of clusters to form.
    return_centers : bool, optional
        If True, also return cluster centers.

    Returns
    -------
    cluster_ids : np.ndarray, shape (N,)
        Cluster assignment per sample.
    centers : np.ndarray, shape (num_clusters, D), optional
        Cluster centers in normalized feature space (returned if `return_centers=True`).
    """
    
    Xn = normalize(features, axis=1) 
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    cluster_ids = kmeans.fit_predict(Xn)
    if return_centers:
        # centers are in the normalized feature space
        return cluster_ids, kmeans.cluster_centers_
    return cluster_ids

# cluster_ids, centers = cluster_features_kmeans(X_pca, best_k, return_centers=True)

# 5) t-SNE visualization -------------------------------------------------------
def visualize_clusters(features_2d, cluster_ids, title="t-SNE Clustering"):
    """
    Scatter-plot 2D embeddings colored by cluster assignment.

    Parameters
    ----------
    features_2d : np.ndarray, shape (N, 2)
        2D embeddings (e.g., from t-SNE).
    cluster_ids : np.ndarray, shape (N,)
        Cluster labels for each sample.
    title : str, optional
        Matplotlib figure title.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_ids, cmap="tab10", s=5)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def tsne_embed(features, perplexity=30, random_state=0):
    """
    Embed high-dimensional features into 2D with t-SNE.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        High-dimensional features (e.g., PCA outputs).
    perplexity : float, optional (default=30)
        t-SNE perplexity parameter (roughly, local neighborhood size).
    random_state : int, optional (default=0)
        RNG seed for reproducibility.

    Returns
    -------
    embedded : np.ndarray, shape (N, 2)
        2D t-SNE coordinates.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    return tsne.fit_transform(features)

# embedded = tsne_embed(X_pca, perplexity=30)
# visualize_clusters(embedded, cluster_ids)

# 6) Show top images per cluster ----------------------------------------------
def show_top_cluster_images(reduced_features, cluster_ids, cluster_centers, original_images, top_k=25, method_name="kmeans"):
        """
    Display the top-K images closest to each cluster center.

    Parameters
    ----------
    reduced_features : np.ndarray, shape (N, D')
        Reduced feature representations (e.g., PCA outputs) used for distance ranking.
    cluster_ids : np.ndarray, shape (N,)
        Cluster assignment per sample.
    cluster_centers : np.ndarray, shape (K, D')
        Cluster centers in the same reduced feature space.
    original_images : list or array-like
        Original images corresponding to `reduced_features`; used for visualization.
    top_k : int, optional (default=25)
        Number of closest images to display per cluster.
    method_name : str, optional
        Label used in plot titles (e.g., 'kmeans').

    Notes
    -----
    - Uses Euclidean distance in the reduced feature space to rank images.
    - Expects grayscale images when displaying with `cmap="gray"`.
    """
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


#7) Evaluate cluster quality diagnostics ------------------------------------------
def cluster_diagnostics(features, labels):
    """
    Compute and print basic cluster quality diagnostics.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Feature representations used for clustering (e.g., PCA outputs).
    labels : np.ndarray, shape (N,)
        Cluster assignments from k-means or another clustering method.

    Prints
    ------
    - Whether NaNs exist in the features
    - Cluster sizes (counts per cluster)
    - Silhouette score (higher is better, range: [-1, 1])
    """
    print("NaNs in features?", np.isnan(features).any())
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:", dict(zip(unique, counts)))

    sil = silhouette_score(features, labels, metric='euclidean')
    print(f"Silhouette score: {sil:.4f}")
    return sil
