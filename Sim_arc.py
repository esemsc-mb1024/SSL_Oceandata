import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A CNN backbone for feature extraction.
    Input: Grayscale images [B, 1, H, W]
    Output: Feature vector [B, 512]
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 96 → 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48 → 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 → 12

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 → 6

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6 → 3

            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 512, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)         # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten to [B, 512]
        return x


class SimCLR(nn.Module):
    """
    SimCLR model: encoder + projection head
    Args:
        base_encoder: feature extractor (e.g., SimpleCNN)
        out_dim: dimensionality of the projected representation z
    """
    def __init__(self, base_encoder, out_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.
    
    Args:
        z_i: Tensor of shape [B, D] — embeddings from view 1
        z_j: Tensor of shape [B, D] — embeddings from view 2
        temperature: Scaling factor for similarity
        
    Returns:
        Scalar loss
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)         # [2B, D]
    z = F.normalize(z, dim=1)                # Unit norm

    # Cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature # [2B, 2B]

    # Remove similarity to self
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: i <-> i + B
    labels = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ])

    loss = F.cross_entropy(sim, labels)
    return loss
