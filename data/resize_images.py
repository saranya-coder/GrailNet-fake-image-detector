from PIL import Image
import os

def resize_images(src_folder, dest_folder, size=(128, 128)):
    os.makedirs(dest_folder, exist_ok=True)
    count = 0
    for fname in os.listdir(src_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(os.path.join(src_folder, fname)).convert('RGB')
                img = img.resize(size, Image.LANCZOS)
                img.save(os.path.join(dest_folder, fname))
                count += 1
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    print(f"Resized {count} images to {size} in '{dest_folder}'")

# Resize both real and fake images
resize_images("real_raw", "real", (128, 128))
resize_images("fake_raw", "fake", (128, 128))
