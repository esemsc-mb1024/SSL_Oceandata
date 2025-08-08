import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.dino import VisionTransformer, DinoLoss
from datasets.dataset import InMemoryDinoDataset
from datasets.transforms import DinoMultiCropTransform
from utils.collate import dino_collate_fn
from utils.momentum import update_teacher, get_teacher_momentum

from torch.utils.data import DataLoader

# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 64
student_temp = 0.2
out_dim = 256
checkpoint_dir = "checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "epoch_50_loss_0.2792.pt")  # Modify as needed

# ----------------------------
# Model setup
# ----------------------------
student = VisionTransformer(out_dim=out_dim).to(device)
teacher = VisionTransformer(out_dim=out_dim).to(device)
teacher.load_state_dict(student.state_dict())  # Initialize teacher same as student

optimizer = optim.Adam(student.parameters(), lr=1e-4)
loss_fn = DinoLoss(out_dim=out_dim, student_temp=student_temp)

start_epoch = 0
resume_training = os.path.exists(checkpoint_path)

# ----------------------------
# Resume from checkpoint
# ----------------------------
if resume_training:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['student_state_dict'])
    teacher.load_state_dict(checkpoint['teacher_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")

# ----------------------------
# Data loading
# ----------------------------
array_stack = torch.load("final_stack.pt") if os.path.exists("final_stack.pt") else None
if array_stack is None:
    raise ValueError("Could not find `final_stack.pt`")

transform = DinoMultiCropTransform()
dataset = InMemoryDinoDataset(array_stack, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dino_collate_fn)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(start_epoch, num_epochs):
    student.train()
    teacher.eval()
    total_loss = 0.0

    for crops, originals in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        crops = [c.to(device) for c in crops]
        originals = originals.to(device)

        student_outputs = [student(c) for c in crops]
        with torch.no_grad():
            teacher_outputs = [teacher(c) for c in crops[:2]]

        loss = loss_fn(student_outputs, teacher_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        teacher_momentum = get_teacher_momentum(epoch)
        update_teacher(student, teacher, teacher_momentum)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}_loss_{avg_loss:.4f}.pt")
    torch.save({
        'epoch': epoch + 1,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    print(f"Saved checkpoint to: {save_path}")
