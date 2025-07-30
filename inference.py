import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Add data folder to path
sys.path.append('./data')

# Import your model and dataset components
from train_reconstructor import Reconstructor
from train_interrogator import Interrogator

# --------- Load Trained Models ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reconstructor = Reconstructor().to(device)
reconstructor.load_state_dict(torch.load("checkpoints/reconstructor.pth", map_location=device))
reconstructor.eval()

interrogator = Interrogator().to(device)
interrogator.load_state_dict(torch.load("checkpoints/interrogator.pth", map_location=device))
interrogator.eval()

# --------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --------- Visualization Function ----------
def show_visualization(img_tensor, recon_tensor, fname, label):
    import os
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    residual = np.abs(img - recon)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(recon)
    axs[1].set_title("Reconstruction")
    axs[2].imshow(residual, cmap='hot')
    axs[2].set_title("Residual Heatmap")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"{fname} — Predicted: {label}")
    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    save_path = os.path.join("visualizations", f"{fname}_viz.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")
# --------- Inference Loop ----------
test_folder = "data/fake"
image_files = sorted(os.listdir(test_folder))[:10]

print("\nPredictions on 10 test images:\n")

for fname in image_files:
    path = os.path.join(test_folder, fname)
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = reconstructor(x)
        residual = torch.abs(x - recon)
        logits = interrogator(residual)
        pred = torch.argmax(logits, dim=1).item()
        label = "Fake" if pred == 1 else "Real"

    print(f"{fname}: {label}")
    show_visualization(x.cpu(), recon.cpu(), fname, label)
