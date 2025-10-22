from PIL import Image
import os

path = "path"  # Change this to your dataset path
corrupt_images = []

fnames = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
for fname in fnames: 
    fpath = os.path.join(path, fname)
    try:
        with Image.open(fpath) as img:
            img = img.convert("RGB")  # Just in case it's grayscale or RGBA
            img.resize((256, 256))  
    except Exception as e:
        print(f"Corrupt image: {fname} — {e}")
        corrupt_images.append(fpath)

        # os.remove(fpath)  # Optional: delete it
print(f"\nChecked {len(os.listdir(path))} images. Found {len(corrupt_images)} corrupt.")
