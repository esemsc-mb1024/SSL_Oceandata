import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

"""
Augmentation Transforms for SAR WV Self-Supervised Learning
===========================================================

This module defines torchvision-based data augmentation pipelines for
grayscale Sentinel-1 Wave Mode (WV) images used in contrastive SSL (SimCLR)
and self-distillation (DINO).
"""

class LightContrastiveAug:
    """
    Light contrastive transform for grayscale images.
    Suitable for subtle local differences.
    """
    def __init__(self):
        self.transform = T.Compose([
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(200, scale=(0.7, 1.0)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x), x


class ModerateContrastiveAug:
    """
    Moderately aggressive contrastive transform.
    Includes crop, rotation, blur, affine.
    """
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=200, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=3),
            T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5) # adjust for your dataset
])

    def __call__(self, x):
        return self.transform(x), self.transform(x), x


class HeavyContrastiveAug:
    """
    Stronger contrastive augmentation with erasing and sharpness.
    """
    def __init__(self):
        self.transform = T.Compose([        
            T.RandomResizdedCrop(size=(200, 200), scale=(0.5,1.0)),  # no zoom
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=5),
            T.RandomApply([T.RandomAdjustSharpness(2)], p=0.5),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.5, 1.5), value=0.0)  # normalize  
])

    def __call__(self, x):
        return self.transform(x), self.transform(x), x


class DinoMultiCropLight:
    """
    DINO-style multi-crop transform for grayscale data (less aggressive).
    2 global crops and 6 local crops, all resized to 200x200.
    """
    def __init__(self):
        self.global_transform = T.Compose([
            T.RandomResizedCrop((200, 200), scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop((96, 96), scale=(0.1, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            T.Resize((200, 200))
        ])

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(2)]
        crops += [self.local_transform(x) for _ in range(6)]
        return crops


class DinoMultiCropStrong:
    """
    More aggressive version of DINO multi-crop with affine and sharpness.
    """
    def __init__(self):
        self.global_transform = T.Compose([
            T.RandomResizedCrop((200, 200), scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.Resize((200, 200))
        ])

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(2)]
        crops += [self.local_transform(x) for _ in range(6)]
        return crops

    