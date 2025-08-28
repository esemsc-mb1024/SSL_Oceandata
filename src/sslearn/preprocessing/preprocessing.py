import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

def pad_to_200(array: np.ndarray) -> np.ndarray:
    """
    Pad a 2D numpy array (grayscale image) to (200,200) 
    only if it's smaller. Uses mean pixel value as fill.

    Args:
        array (np.ndarray): 2D array, shape (H, W).

    Returns:
        np.ndarray: Array padded to at least (200,200).
    """
    target_h, target_w = 200, 200
    h, w = array.shape

    # If already big enough, return unchanged
    if h >= target_h and w >= target_w:
        return array

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    if pad_h > 0 or pad_w > 0:
        padding = (
            (pad_h // 2, pad_h - pad_h // 2),  # (top, bottom)
            (pad_w // 2, pad_w - pad_w // 2),  # (left, right)
        )
        mean_val = int(array.mean())
        array = np.pad(array, padding, mode="constant", constant_values=mean_val)

    return array


def nearest_neighbor_fill(arr):
    """Fill NaN values in a 2D array using the nearest neighbor interpolation."""
    nan_mask = np.isnan(arr)
    valid_mask = ~nan_mask

    if np.all(nan_mask):
        return np.zeros_like(arr)

    coords = np.array(np.nonzero(valid_mask)).T
    nan_coords = np.array(np.nonzero(nan_mask)).T

    tree = cKDTree(coords)
    nearest_indices = tree.query(nan_coords, k=1)[1]
    nearest_values = arr[tuple(coords[nearest_indices].T)]

    filled_arr = arr.copy()
    filled_arr[tuple(nan_coords.T)] = nearest_values

    return filled_arr




