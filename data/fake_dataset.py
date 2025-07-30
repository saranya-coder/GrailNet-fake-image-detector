import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, image_size=128, max_samples=1000):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        real_files = sorted(os.listdir(real_dir))[:max_samples]
        fake_files = sorted(os.listdir(fake_dir))[:max_samples] if fake_dir else []

        self.samples = [(os.path.join(real_dir, f), 0) for f in real_files]
        self.samples += [(os.path.join(fake_dir, f), 1) for f in fake_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
