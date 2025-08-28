import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Loss Functions for Self-Supervised Learning
===========================================

This module implements two loss functions commonly used in self-supervised 
representation learning for vision models.

Functions / Classes
-------------------
nt_xent_loss(z_i, z_j, temperature=0.5):
    - Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) from SimCLR.
    - Encourages representations of two augmented views of the same image to 
      be close, while pushing apart views from different images.
    - Reference: Chen et al. (2020), 
      "A Simple Framework for Contrastive Learning of Visual Representations."

DinoLoss(out_dim=2048, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
    - Loss function for DINO (self-distillation with no labels).
    - Computes cross-entropy between teacher and student distributions 
      across multiple augmented views.
    - Teacher outputs are centered, temperature-scaled, and stop-grad applied.
    - Student outputs are softened with a higher temperature.
    - Maintains an exponential moving average (EMA) "center" to stabilize training.
    - Reference: Caron et al. (2021), 
      "Emerging Properties in Self-Supervised Vision Transformers."

      AI assisted with the creation of loss fucntions
"""
# ---------------------------------------------
# SimCLR NT-Xent Loss
# ---------------------------------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.

    Args:
        z_i: Tensor of shape [B, D] from view 1
        z_j: Tensor of shape [B, D] from view 2
        temperature: scaling factor

    Returns:
        Scalar contrastive loss
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)         # L2 normalize

    # Cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # Remove self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair indices
    positives = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ])
    labels = positives

    # Cross entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss

# ---------------------------------------------
# DINO Self-Distillation Loss
# ---------------------------------------------
class DinoLoss(nn.Module):
    def __init__(self, out_dim=2048, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # center over prototype dimension D (matches out_dim)
        self.register_buffer("center", torch.zeros(1, out_dim))  # [1, D]

    def forward(self, student_outputs, teacher_outputs):
        """
        student_outputs: list of [B, D] tensors (student logits per view)
        teacher_outputs: list of [B, D] tensors (teacher logits per view)
        Returns: scalar loss
        """
        device = teacher_outputs[0].device
        center = self.center.to(device)

        # scale logits and make distributions
        # teacher: centered, temperature, softmax; detached from graph
        teacher_probs = [
            F.softmax((t - center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_outputs
        ]
        # student: temperature (log_softmax inside CE term)
        student_logits = [s / self.student_temp for s in student_outputs]

        total_loss, n_terms = 0.0, 0
        for iq, t_prob in enumerate(teacher_probs):      # iterate teacher views
            for v, s_logit in enumerate(student_logits): # iterate student views
                if v == iq:                               # skip same-view pairs
                    continue
                # cross-entropy: - sum p_t * log p_s
                total_loss += -(t_prob * F.log_softmax(s_logit, dim=-1)).sum(dim=-1).mean()
                n_terms += 1

        total_loss /= max(n_terms, 1)

        with torch.no_grad():
            concat_t = torch.cat(teacher_outputs, dim=0)  # [V*B, D]
            batch_center = concat_t.mean(dim=0, keepdim=True)  # [1, D]

        # make sure buffer is on the same device as batch_center
            if self.center.device != batch_center.device:
                self.center = self.center.to(batch_center.device)

            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

        return total_loss
