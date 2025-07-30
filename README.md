# GRAIL-Net: Generative Residual Artifact Interrogation Learner

GRAIL-Net is a novel deep learning model built using PyTorch to detect AI-generated fake images. It reconstructs an image using a neural autoencoder and analyzes residual artifacts to distinguish between real and fake inputs. This approach provides interpretable outputs through residual heatmaps and visual comparisons.

# Project Structure


        GRAIL-Net/
        │
        ├── checkpoints/                # Saved models
        │   ├── reconstructor.pth
        │   └── interrogator.pth
        │
        ├── data/                       # Datasets
        │   ├── real_raw/              # Original real images
        │   ├── fake_raw/              # Original fake images
        │   ├── real/                  # Resized real images
        │   ├── fake/                  # Resized fake images
        │   ├── fake_dataset.py        # Dataset loader
        │   └── __init__.py
        │
        ├── visualizations/            # Saved visualizations
        │
        ├── resize_images.py           # Script to resize raw images
        ├── train_reconstructor.py     # Train autoencoder
        ├── train_interrogator.py      # Train classifier on residuals
        ├── inference.py               # Run predictions and generate visualizations
        ├── requirements.txt           # Python dependencies
        └── README.md


# What It Does

1. Reconstruction: A shallow convolutional autoencoder reconstructs an input face.

2. Residual Learning: Subtracts reconstructed image from original to capture generation artifacts.

3. Interrogation: A lightweight CNN uses the residual image to predict whether it's real or fake.

4. Visualization: For every image, three outputs are shown:

        Original

        Reconstructed

        Residual heatmap


# Setup Instructions
To ensure a clean environment:

1. Clone the Repository

    git clone https://github.com/yourusername/GRAIL-Net.git

2. Create and Activate Virtual Environment

On Windows:

    python -m venv venv
    venv\Scripts\activate

On macOS/Linux:

    python3 -m venv venv
    source venv/bin/activate

3. Install Dependencies

    pip install -r requirements.txt

# Sample Output

Each visualization contains:

    Left: Original Image

    Center: Reconstructed Output

    Right: Residual Heatmap


# Future Work

Integrate Grad-CAM or SHAP for deeper interpretability

Extend model to detect other synthetic content (e.g., GAN videos)

Improve reconstruction fidelity with deeper autoencoder