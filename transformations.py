class contrastive_transforms3:
    def __init__(self): 
        self.transform = T.Compose([
    T.RandomCrop(size=(160, 160)),  # no zoom
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
    T.GaussianBlur(kernel_size=5),
    T.RandomApply([T.RandomAdjustSharpness(2)], p=0.5),
    T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),  # normalize
  
])

    def __call__(self, x):
        return self.transform(x), self.transform(x), x

class contrastive_transforms2:
    def __init__(self):
        self.transform = T.Compose([
    T.RandomResizedCrop(size=160, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
    T.GaussianBlur(kernel_size=3),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5) 
])

    def __call__(self, x):
        return self.transform(x), self.transform(x), x

class contrastive_transforms:
    def __init__(self):
        self.transform = T.Compose([
            T.RandomRotation(degrees=30),
            T.RandomResizedCrop(190, scale=(0.8, 1.0)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)), 
            # Normalize to [0, 1] again
            T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        ])# lighter blur
       

    def __call__(self, x):
        return self.transform(x), self.transform(x), x

import torchvision.transforms as T

class DinoMultiCropTransform1:
    def __init__(self):
        # Global crops (stronger spatial and intensity variation)
        self.global_transform = T.Compose([
            T.RandomResizedCrop((200, 200), scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.Resize((200, 200))
        ])

        # Local crops (smaller, subtle variations)
        self.local_transform = T.Compose([
            T.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.Resize((200, 200))
        ])


    def __call__(self, x):
        crops = []
        # 2 global
        for _ in range(2):
            crops.append(self.global_transform(x))
        # 6 local
        for _ in range(6):
            crops.append(self.local_transform(x))
        return crops

import torchvision.transforms as T

class DinoMultiCropTransform:
    def __init__(self):
        self.global_transform = T.Compose([
            T.RandomResizedCrop((200, 200), scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.Resize((200, 200))
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop((96, 96), scale=(0.1, 0.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            T.Resize((200, 200))
        ])

    def __call__(self, x):
        crops = []
        for _ in range(2):
            crops.append(self.global_transform(x))
        for _ in range(6):
            crops.append(self.local_transform(x))
        return crops

    