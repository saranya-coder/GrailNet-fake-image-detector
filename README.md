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

2. Download Dataset:

    This project uses two datasets:

    Real Faces: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00005.png
    Fake Faces: https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist

    After downloading:

    Manually place the  images into this structure in the project root:

    data/
    ├── real_raw/     ← raw real images from FFHQ
    └── fake_raw/     ← raw fake images from ThisPersonDoesNotExist

    Then, run fake_dataset.py to resize and organize the training data:

        python fake_dataset.py
       
    This will create a data/ folder with the following structure:

        data/
        ├── real/          ← resized real images
        ├── fake/          ← resized fake images
        ├── real_raw/
        └── fake_raw/
        ___ fake_dataset.py
    
    Make sure the data/ folder is in the root directory of the project before training or inference.

3. Create and Activate Virtual Environment

On Windows:

    python -m venv venv
    venv\Scripts\activate

On macOS/Linux:

    python3 -m venv venv
    source venv/bin/activate

4. Install Dependencies

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