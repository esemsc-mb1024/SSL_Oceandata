import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_kmeans(
    X,
    k_min=2,
    k_max=30,
    n_init="auto",
    max_iter=300,
    random_state=42,
    show_plot=True,
):
    """
    Compute k-means inertia over k in [k_min, k_max] and pick the elbow.

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

    ks = np.arange(k_min, k_max + 1)
    inertias = []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        km.fit(X)
        inertias.append(km.inertia_)

    inertias = np.array(inertias)

    # Try kneed for elbow detection; otherwise geometric fallback
    best_k = None
    try:
        from kneed import KneeLocator
        kl = KneeLocator(
            ks, inertias,
            curve="convex",
            direction="decreasing",
            interp_method="polynomial"
        )
        if kl.knee is not None:
            best_k = int(kl.knee)
    except Exception:
        pass

    if best_k is None:
        # Fallback: max perpendicular distance to line
        x1, y1 = ks[0], inertias[0]
        x2, y2 = ks[-1], inertias[-1]
        num = np.abs((y2 - y1) * ks - (x2 - x1) * inertias + (x2 * y1 - y2 * x1))
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
        best_k = int(ks[np.argmax(num / den)])

    if show_plot:
        plt.figure()
        plt.plot(ks, inertias, marker='o')
        plt.axvline(best_k, linestyle='--')
        plt.xlabel('k (number of clusters)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow method (best k = {best_k})')
        plt.tight_layout()
        plt.show()

    return best_k, ks, inertias
