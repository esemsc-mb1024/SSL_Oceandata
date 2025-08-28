import math
import torch
import torch.nn as nn

@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
    """EMA update: teacher = m*teacher + (1-m)*student."""
    for q, k in zip(student.parameters(), teacher.parameters()):
        k.mul_(momentum).add_(q, alpha=1.0 - momentum)

def get_teacher_momentum(progress: float, base_m: float = 0.996):
    """
    Cosine schedule in [base_m, 1.0]. 
    progress \in [0,1], e.g. (epoch + it/num_it)/num_epochs.
    """
    return 1.0 - (1.0 - base_m) * 0.5 * (1.0 + math.cos(math.pi * progress))

def dino_collate_fn(batch):
    """
    batch: list of (views, original)
      - views: list[Tensor] per sample (e.g., 2 global + n local)
      - original: Tensor [C,H,W]
    returns:
      - stacked_views: list[Tensor] shape [B,C,H_i,W_i] per view index
      - originals: Tensor [B,C,H,W]
    """
    views_list = [b[0] for b in batch]
    originals = torch.stack([b[1] for b in batch], dim=0)

    n_views = len(views_list[0])
    stacked_views = [torch.stack([v[i] for v in views_list], dim=0) for i in range(n_views)]
    return stacked_views, originals

