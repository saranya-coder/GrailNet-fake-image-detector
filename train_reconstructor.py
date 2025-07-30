import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ✅ Optional: only needed if import fails
import sys
sys.path.append('./data')

from fake_dataset import RealFakeDataset

# --------------------------
# Reconstructor Model
# --------------------------
class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64x64 → 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32 → 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 → 8x8
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16 → 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 32x32 → 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --------------------------
# Train Function
# --------------------------
def train_reconstructor(real_dir, image_size=64, batch_size=32, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reconstructor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    dataset = RealFakeDataset(real_dir=real_dir, fake_dir=None, image_size=image_size, max_samples=1000)
    dataset.samples = [s for s in dataset.samples if s[1] == 0]  # Only real images
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "checkpoints/reconstructor.pth")
    print("Reconstructor saved at checkpoints/reconstructor.pth")

# --------------------------
# Run It
# --------------------------
if __name__ == "__main__":
    train_reconstructor(real_dir="data/real", image_size=64)
