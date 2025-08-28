import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

"""
TiffDataset for Labelled GeoTIFF Images
=======================================

This module defines a PyTorch `Dataset` for loading and preprocessing 
labelled GeoTIFF imagery (e.g., Sentinel-1 WV images).
"""

class TiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.transform = transform

        for idx, label_name in enumerate(sorted(os.listdir(root_dir))):
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                self.label_map[label_name] = idx
                for fname in os.listdir(label_path):
                    if fname.endswith('.tiff') or fname.endswith('.tif'):
                        self.samples.append(os.path.join(label_path, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # Load and normalize image
        with Image.open(path) as img:
            arr = np.array(img).astype(np.float32)

            # Normalize image to [0, 1] 
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

            # Add channel dimension [1, H, W]
            arr = np.expand_dims(arr, axis=0)

            tensor_img = torch.from_numpy(arr)

            # Apply transform if provided
            if self.transform:
                tensor_img = self.transform(tensor_img)

        return tensor_img, label