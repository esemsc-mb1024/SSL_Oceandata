import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, out_dim=256, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        """
        DINO Loss for self-distillation without labels.

        Args:
            out_dim: dimensionality of projected features
            student_temp: temperature for student outputs
            teacher_temp: temperature for teacher outputs
            center_momentum: momentum for updating the center
        """
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, 1, out_dim))  # [1, 1, D]

    def forward(self, student_outputs, teacher_outputs):
        """
        Args:
            student_outputs: list of [B, N, D] tensors
            teacher_outputs: list of [B, N, D] tensors

        Returns:
            Scalar loss
        """
        device = teacher_outputs[0].device
        center = self.center.to(device)

        # Apply temperature
        student_logits = [s / self.student_temp for s in student_outputs]
        teacher_probs = [
            F.softmax((t - center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_outputs
        ]

        total_loss = 0.0
        n_loss_terms = 0

        for iq, t_out in enumerate(teacher_probs):
            for v, s_out in enumerate(student_logits):
                if v == iq:
                    continue  # skip matching views

                # Match shapes
                min_b = min(t_out.size(0), s_out.size(0))
                min_n = min(t_out.size(1), s_out.size(1))
                t_trim = t_out[:min_b, :min_n, :]
                s_trim = s_out[:min_b, :min_n, :]

                # Patch-wise cross-entropy
                loss = -torch.sum(t_trim * F.log_softmax(s_trim, dim=-1), dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center
        concat_teacher = torch.cat(teacher_outputs, dim=0)  # [2B, N, D]
        batch_center = concat_teacher.mean(dim=(0, 1), keepdim=True)
        self.center = self.center.to(batch_center.device)
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

        return total_loss
