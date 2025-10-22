# EL Image Segmentation Feature Extractor + FID/KID Evaluation

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel
import pandas as pd
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
save_path = "/home/shanifi/code/pv-diffusion/FID/deeplabv3_trained.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# --- 1. Dataset Setup --- #
class ELSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        if image_list:
            self.image_files = sorted(image_list)
        else:
            self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))] )
        


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])  # assumes same filenames
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



# --- 4. Training Loop --- #
def train(model, dataloader, val_loader, epochs=20, save_path=save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    patience_counter = 0
    min_delta = 0.01
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.3f}, Avg Loss = {avg_loss:.3f}")
        
        # Validate and save the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)['out']
                val_loss += criterion(val_outputs, val_masks).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.3f}")

        # Early stopping logic
        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
            print(f"Model improved. Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    # # Save the trained model
    # torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
    # print(f"Model saved to {save_path}")
    return model



# --- 5. Feature Extractor --- #
class FeatureExtractor(nn.Module):
    def __init__(self, seg_model):
        super().__init__()
        self.encoder = seg_model.backbone  # ResNet encoder

    def forward(self, x):
        # x = self.encoder(x)['out']
        x = self.encoder(x)
        if isinstance(x, dict):  # in case model changes
            x = x.get("out", list(x.values())[-1])
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


# --- 6. Feature Extraction Function --- #
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for imgs in tqdm(dataloader):
            imgs = imgs[0].to(device) if isinstance(imgs, (list, tuple)) else imgs
            feats = model(imgs)
            features.append(feats.cpu())
    return torch.cat(features).numpy()


# --- 7. FID and KID Calculations --- #
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_kid(X, Y, degree=3, gamma=None, coef0=1):
    K_XX = polynomial_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

if __name__== "__main__":
    # --- 2. Load Dataset --- #
    train_ds = ELSegmentationDataset("/home/shanifi/code/pv-diffusion/dataset/el_images_train", "/home/shanifi/code/pv-diffusion/dataset/el_masks_train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    val_ds = ELSegmentationDataset("/home/shanifi/code/pv-diffusion/dataset/el_images_val", "/home/shanifi/code/pv-diffusion/dataset/el_masks_val", transform=transform)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    # --- 3. Model Setup --- #
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Train the model (adjust epochs as needed)
    # trained_model = train(model, train_loader, val_loader, epochs=100)
    
    # Load the saved state dictionary
    checkpoint = "/home/shanifi/code/pv-diffusion/FID/deeplabv3_trained.pth_epoch_23.pth"
    model.load_state_dict(torch.load(checkpoint))
    print("Model Loaded!")
    model.eval()
    

    feature_model = FeatureExtractor(model).to(device)
    feature_model.eval()

    # Assuming you have separate loaders for real and fake images
    syn_image_path = "/home/shanifi/code/pv-diffusion/exports/inference_multi_nopadding_withRF/image"
    syn_images_list = os.listdir(syn_image_path)
    # syn_images_list = os.listdir("/home/shanifi/code/pv-diffusion/exports/inference_mono_nopadding_withRF/image")
    cat_specific_main_path_real_data = "/home/shanifi/code/pv-diffusion/dataset/el_images_test"
    
    per_image_results = []
    
    base_names_set = set()
    for syn_image in syn_images_list:
        cat_image_base_name = syn_image.split("-")[1].rsplit("_", 1)[0]
        if cat_image_base_name.endswith("no_padding"):
            cat_image_base_name = cat_image_base_name[: -len('_no_padding')]
        base_names_set.add(cat_image_base_name)
        
    # List of images to remove (with .png extension)
    images_to_remove = [
        "CSIR_00224_r6_c4.png", "CSIR_00225_r9_c4.png", "CSIR_00232_r1_c6.png", "CSIR_00231_r6_c2.png",
        "CSIR_00224_r12_c3.png", "CSIR_00230_r9_c1.png", "CSIR_00234_r9_c6.png", "CSIR_00233_r12_c5.png",
        "CSIR_00234_r10_c4.png", "CSIR_00222_r11_c3.png", "CSIR_00231_r11_c6.png", "CSIR_00229_r12_c3.png",
        "CSIR_00225_r12_c6.png", "CSIR_00229_r1_c5.png", "CSIR_00222_r6_c2.png", "CSIR_00227_r2_c6.png",
        "CSIR_00233_r5_c2.png", "CSIR_00227_r8_c6.png", "CSIR_00221_r6_c2.png"
    ]

    # Strip '.png' and remove from set
    to_remove = {name.replace('.png', '') for name in images_to_remove}
    base_names_set.difference_update(to_remove)

    for base_name in tqdm(base_names_set, desc="per-image evaluation"):
        real_image_path = os.path.join(cat_specific_main_path_real_data, f"{base_name}.png")
        if not os.path.exists(real_image_path):
            print(f"Path does not exist: {real_image_path}")
            continue
        
        real_image = transform(Image.open(real_image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            real_feats = feature_model(real_image).cpu().numpy()
        
        # Find corresponding fake images
        matching_fake_images = [img for img in syn_images_list if base_name in img]
        if len(matching_fake_images) == 0:
            print(f"No matching fake images found for {base_name}")
            continue
        
        gen_feats = []
        for fake_image in matching_fake_images:
            fake_image_path = os.path.join(syn_image_path, fake_image)
            fake_image = transform(Image.open(fake_image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                fake_feats = feature_model(fake_image).cpu().numpy()
                gen_feats.append(fake_feats)
        
        gen_feats =np.vstack(gen_feats)

        # Compte FID and KID for this group
        mu_real = np.mean(real_feats, axis=0)
        # sigma_real = np.cov(real_feats, rowvar=False) if real_feats.shape[0] > 1 else np.eye(real_feats.shape[1])
        sigma_real = np.eye(real_feats.shape[1])  # Use identity matrix for covariance if only one sample
        mu_fake = np.mean(gen_feats, axis=0)
        sigma_fake = np.cov(gen_feats, rowvar=False)

        fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        kid = calculate_kid(real_feats, gen_feats)
        
        per_image_results.append({
            "base_name": base_name,
            "fid": fid,
            "kid": kid,
            "num_fakes": len(matching_fake_images)
        })

# --- Analysis and Reporting --- #
results_df = pd.DataFrame(per_image_results)
print(f"\nPer-Image Evaluation Summary:")
print(f"Mean FID: {results_df['fid'].mean():.3f} ± {results_df['fid'].std():.3f}")
print(f"Mean KID: {results_df['kid'].mean():.5f} ± {results_df['kid'].std():.5f}")

# Top 5 worst FID/KID examples
print("\nTop 5 images with highest FID:")
print(results_df.sort_values("fid", ascending=False).head(5)[["base_name", "fid"]])

print("\nTop 5 images with highest KID:")
print(results_df.sort_values("kid", ascending=False).head(5)[["base_name", "kid"]])

# Optional: save to CSV
results_df.to_csv("per_image_fid_kid_scores.csv", index=False)
