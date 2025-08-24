# datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np


class InMemoryContrastiveDataset(Dataset):
    def __init__(self, array_stack, transform=None, return_index=False):
        self.data = array_stack
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32)

        if np.std(arr) == 0:
            arr += np.random.normal(0, 1e-6, size=arr.shape)

        # Assuming arr is your image array (grayscale, 2D)
        P01 = np.percentile(arr, 1)
        P99 = np.percentile(arr, 99)
        
        # Apply the normalization formula
        arr = 255.0 * (arr - P01) / (P99 - P01 + 1e-8)
        arr = np.clip(arr, 0, 255)  # Ensure values are within [0, 255]
        
        # Convert to float32 and normalize to [0, 1] if needed for torch models
        arr = arr.astype(np.float32) / 255.0
        
        # Add channel dimension and convert to tensor
        arr = np.expand_dims(arr, axis=0)  # shape: [1, H, W]
        img = torch.from_numpy(arr)

        if self.return_index:
            return img, idx  #  Used for feature extraction

        if self.transform:
            aug1, aug2, original = self.transform(img)
            return aug1, aug2, original

        return img, img, img



class InMemoryDinoDataset(Dataset):
    def __init__(self, array_stack, transform=None):
        self.data = array_stack
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32)
        if np.std(arr) == 0:
            arr += np.random.normal(0, 1e-6, size=arr.shape)

        # Assuming arr is your image array (grayscale, 2D)
        P01 = np.percentile(arr, 1)
        P99 = np.percentile(arr, 99)
        
        # Apply the normalization formula
        arr = 255.0 * (arr - P01) / (P99 - P01 + 1e-8)
        arr = np.clip(arr, 0, 255)  # Ensure values are within [0, 255]
        
        # Convert to float32 and normalize to [0, 1] if needed for torch models
        arr = arr.astype(np.float32) / 255.0
        
        # Add channel dimension and convert to tensor
        arr = np.expand_dims(arr, axis=0)  # shape: [1, H, W]
        img = torch.from_numpy(arr)

        # also store the original, untransformed normalized version
        original = img.clone()

        if self.transform:
            crops = self.transform(img)
            return crops, original  # return both
        else:
            return [img], original