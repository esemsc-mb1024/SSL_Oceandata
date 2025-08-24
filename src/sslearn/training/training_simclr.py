import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# dataset (from preprocessing, but re-exported by shim)
from sslearn.datasets import InMemoryContrastiveDataset

# models
from sslearn.models.sim_arc import SimCLR, SimpleCNN

# transforms (via shim → transforms/transforms.py)
from sslearn.transformations import ContrastiveTransforms

# loss (lives in loss_fn.py)
from sslearn.loss_fn import NTXentLoss

# ---------- configuration ----------
epochs = 50
batch_size = 128
temperature = 0.4
save_every = 5
save_path = "contrastive_checkpoints"
data_path = "final_stack.npy"

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# ---------- data ----------
final_stack = np.load(data_path)
train_dataset = in_memory_contrastive_dataset(final_stack, transform=contrastive_transforms())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------- model ----------
model = simclr(simple_cnn()).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- training function ----------
def train_contrastive(model, train_loader, optimizer, epochs=50, device="cuda", save_every=5, save_path="contrastive_checkpoints"):
    model.train()
    loss_history = []

    if os.path.exists(save_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"{save_path}_run_{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    print(f"saving checkpoints to: {save_path}")

    for epoch in range(epochs):
        total_loss = 0.0
        for x_i, x_j, _ in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature=temperature)

            if torch.isnan(loss):
                print("skipping nan loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"📉 epoch [{epoch+1}/{epochs}] | loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            checkpoint_file = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_file)
            print(f"saved checkpoint: {checkpoint_file}")

    return loss_history

# ---------- run training ----------
if __name__ == "__main__":
    loss_history = train_contrastive(model, train_loader, optimizer, epochs=epochs)
