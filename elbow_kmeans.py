"""
elbow_kmeans.py
---------------
Utility to scan k in [k_min, k_max], compute k-means inertia, detect elbow (via kneed if available,
else geometric fallback), and optionally plot.

Usage:
    from elbow_kmeans import elbow_kmeans
    best_k, ks, inertias = elbow_kmeans(X, k_min=2, k_max=30)

CLI example (requires numpy):
    python -m elbow_kmeans --demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def elbow_kmeans(
    X,
    k_min: int = 2,
    k_max: int = 30,
    n_init: int = 20,
    max_iter: int = 300,
    random_state: int = 42,
    sample_size: int | None = None,   # e.g., 50000 for very large datasets
    show_plot: bool = True
):
    """
    Compute k-means inertia over k in [k_min, k_max] and pick the elbow.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Input data.
    k_min, k_max : int
        Range of cluster counts (inclusive).
    n_init : int
        Number of initializations for KMeans.
    max_iter : int
        Maximum iterations for KMeans.
    random_state : int
        Random seed for reproducibility.
    sample_size : int or None
        If set and N > sample_size, subsample without replacement for speed.
    show_plot : bool
        If True, show the elbow plot.

    Returns
    -------
    best_k : int
        Selected number of clusters by elbow.
    ks : np.ndarray
        Array of scanned k values.
    inertias : np.ndarray
        Corresponding inertia values.
    """
    X = np.asarray(X)

    # Optional speed-up by subsampling (without replacement)
    rng = np.random.RandomState(random_state)
    if sample_size is not None and X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    ks = np.arange(k_min, k_max + 1)
    inertias = []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        km.fit(X_use)
        inertias.append(km.inertia_)

    inertias = np.array(inertias)

    # Try kneed for elbow detection; otherwise geometric fallback
    best_k = None
    try:
        from kneed import KneeLocator
        kl = KneeLocator(
            ks, inertias,
            curve="convex",         # inertia vs k should be convex
            direction="decreasing", # inertia decreases with k
            interp_method="polynomial"
        )
        best_k = int(kl.knee) if kl.knee is not None else None
    except Exception:
        best_k = None

    if best_k is None:
        # Fallback: max perpendicular distance to the line from (k_min, I_min) to (k_max, I_max)
        x1, y1 = ks[0], inertias[0]
        x2, y2 = ks[-1], inertias[-1]
        num = np.abs((y2 - y1) * ks - (x2 - x1) * inertias + (x2 * y1 - y2 * x1))
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        d = num / (den + 1e-12)
        best_k = int(ks[np.argmax(d)])

    if show_plot:
        plt.figure()
        plt.plot(ks, inertias, marker='o')
        plt.axvline(best_k, linestyle='--')
        plt.xlabel('k (number of clusters)')
        plt.ylabel('Inertia (sum of squared distances)')
        plt.title(f'Elbow method (best k = {best_k})')
        plt.tight_layout()
        plt.show()

    return best_k, ks, inertias


def _demo():
    rng = np.random.RandomState(0)
    # Make a simple synthetic dataset: 3 Gaussian blobs
    centers = np.array([[0, 0], [5, 5], [0, 6]])
    X = np.vstack([rng.randn(300, 2) + c for c in centers])
    elbow_kmeans(X, k_min=1, k_max=10, show_plot=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Elbow KMeans utility")
    p.add_argument("--demo", action="store_true", help="Run a quick synthetic demo")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo:
        _demo()
