import torch
import torch.nn as nn


def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
    """
    Update teacher weights as EMA of student weights.
    teacher = m * teacher + (1 - m) * student
    """
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data


def get_teacher_momentum(epoch: int, base_m=0.996, max_epochs=100):
    """
    Cosine schedule for teacher momentum.
    """
    return 1.0 - (1.0 - base_m) * (0.5 * (1. + torch.cos(torch.tensor(epoch / max_epochs * 3.1415926)))).item()


def dino_collate_fn(batch):
    """
    Collate function for DINO multi-crop input.

    Args:
        batch: List of tuples (crops, original) from dataset

    Returns:
        stacked_views: list of [B, C, H, W] tensors for each crop view
        originals:     tensor of [B, C, H, W]
    """
    crops_list = [item[0] for item in batch]
    originals = [item[1] for item in batch]

    num_views = len(crops_list[0])
    stacked_views = []

    for i in range(num_views):
        stacked = torch.stack([crops[i] for crops in crops_list], dim=0)
        stacked_views.append(stacked)

    stacked_originals = torch.stack(originals, dim=0)

    return stacked_views, stacked_originals
