import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
sys.path.append('./data')
from fake_dataset import RealFakeDataset

# Load trained reconstructor
from train_reconstructor import Reconstructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reconstructor = Reconstructor().to(device)
reconstructor.load_state_dict(torch.load("checkpoints/reconstructor.pth"))
reconstructor.eval()

# Residual dataset
class ResidualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        with torch.no_grad():
            recon = reconstructor(img.unsqueeze(0).to(device)).squeeze(0).cpu()
        residual = (img - recon).abs()  # L1 residual
        return residual, label

# Interrogator classifier
class Interrogator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Load real/fake dataset
base_ds = RealFakeDataset("data/real", "data/fake", max_samples=1000)
residual_ds = ResidualDataset(base_ds)
loader = DataLoader(residual_ds, batch_size=32, shuffle=True)

# Train interrogator
model = Interrogator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/3 - Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "checkpoints/interrogator.pth")
print("Interrogator saved.")
