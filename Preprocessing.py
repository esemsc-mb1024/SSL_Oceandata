import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

def pad_to_max_with_mean(arr, target_height=204, target_width=216):
    """Pad a 2D array to target dimensions using the mean value of the array."""
    h, w = arr.shape
    pad_value = np.mean(arr)
    
    padded = np.full((target_height, target_width), pad_value, dtype=arr.dtype)
    
    y_offset = (target_height - h) // 2
    x_offset = (target_width - w) // 2
    
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = arr
    
    return padded

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

def crop_sigma_arrays(sigma0_arrays, target_height=204, target_width=216):
    cropped_sigma_arrays = []
    for arr in sigma0_arrays:
        h, w = arr.shape
        cropped = arr[:target_height, :target_width]  # crop both height and width
        cropped_sigma_arrays.append(cropped)
    return cropped_sigma_arrays

def crop_numpy(img_array, crop_size=200):
    c, h, w = img_array.shape
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    cropped = img_array[ : , start_y:start_y+crop_size, start_x:start_x+crop_size]
    return cropped

# Example of loading a memmap stack (readonly)
def load_memmap_stack(path="clean_combined_stack.dat", shape=(49402, 204, 216), dtype=np.float32):
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)
