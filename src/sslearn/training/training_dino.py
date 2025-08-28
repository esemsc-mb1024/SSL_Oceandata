import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# models / loss
from sslearn.models.dino_arc import VisionTransformer
from sslearn.loss_fn import DinoLoss

# transforms + datasets via shims (cleaner)
from sslearn.transformations import DinoMultiCropTransform
from sslearn.datasets import InMemoryDinoDataset

# training helpers
from sslearn.utils import get_teacher_momentum, update_teacher, dino_collate_fn

# Transforms
from sslearn.transforms.transforms import DinoMultiCropLight, DinoMultiCropStrong

# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 32
student_temp = 0.2
out_dim = 2048
array_path = "/path/to/singma0_WV/processed"   # must exist (NumPy format, not provided)

# ----------------------------
# Model setup
# ----------------------------
student = VisionTransformer(out_dim=out_dim).to(device)
teacher = VisionTransformer(out_dim=out_dim).to(device)
teacher.load_state_dict(student.state_dict())

optimizer = optim.Adam(student.parameters(), lr=0.0005)
loss_fn = DinoLoss(out_dim=out_dim, student_temp=student_temp)

# ----------------------------
# Data loading 
# ----------------------------
array_stack = np.load(array_path)         # (N, C, H, W)
transform = DinoMultiCropTransform()
dataset = InMemoryDinoDataset(array_stack, transform=DinoMultiCropLight())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=dino_collate_fn)


for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()

# ----------------------------
# Training loop 
# ----------------------------
for epoch in range(num_epochs):
    student.train()
    teacher.eval()
    total_loss = 0.0

    for crops, originals in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        crops = [c.to(device) for c in crops]

        # forward
        student_outputs = [student(c) for c in crops]
        with torch.no_grad():
            teacher_outputs = [teacher(c) for c in crops[:2]]  # global views only

        loss = loss_fn(student_outputs, teacher_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA momentum update
        teacher_momentum = get_teacher_momentum(epoch)
        update_teacher(student, teacher, teacher_momentum)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}")




# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
