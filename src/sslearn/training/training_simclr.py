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
from sslearn.preprocessing.datasets import InMemoryContrastiveDataset

# models
from sslearn.models.sim_arc import SimCLR, SimpleCNN

# transforms (via shim → transforms/transforms.py)
from sslearn.transforms.transforms import LightContrastiveAug, ModerateContrastiveAug, HeavyContrastiveAug

# loss 
from sslearn.loss_fn import nt_xent_loss

# ---------- configuration ----------
epochs = 50
batch_size = 128
temperature = 0.5
data_path = "/path/to/sigma0_WV/processed/"

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# ---------- data ----------
final_stack = np.load(data_path)  # (N, C, H, W)
train_dataset = InMemoryContrastiveDataset(final_stack, transform=ModerateContrastiveAug())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------- model ----------
model = SimCLR(SimpleCNN()).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- training function ----------
def train_contrastive(model, train_loader, optimizer, epochs=50, device="cuda"):
    model.train()
    loss_history = []

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
        print(f"epoch [{epoch+1}/{epochs}] | loss: {avg_loss:.4f}")

    return loss_history

# ---------- run training ----------
if __name__ == "__main__":
    loss_history = train_contrastive(model, train_loader, optimizer, epochs=epochs, device=device)
