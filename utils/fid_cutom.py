import torch
import numpy as np
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import PIL.Image as Image
import torchvision.transforms as transforms
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import pacmap
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import lpips
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 30
BATCH_SIZE = 1
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fake_images_mono_path = "/home/shanifi/code/pv-diffusion/exports/inference_mono_resume_all161_7/image"
fake_images_mono_path = "/home/shared/sc/generated_unlabeled/mono/image"
# fake_images_multi_path = "/home/shanifi/code/pv-diffusion/exports/inference_multi_resume7_nopadding_withRF/image"
# fake_images_multi_path = "/home/shanifi/code/pv-diffusion/exports/inference_multi_resume_all161_3/image"
fake_images_multi_path = "/home/shared/sc/generated_unlabeled/multi/image"
# fake_images_multihalfcut_path = "/home/shanifi/code/pv-diffusion/exports/inference_multihalfcut_resume7_nopadding_withRF/image"
# fake_images_multihalfcut_path = "/home/shanifi/code/pv-diffusion/exports/inference_multihalfcut_resume_all161_nomulti_3/image"
fake_images_multihalfcut_path = "/home/shared/sc/generated_unlabeled/multi_halfcut/image"
# fake_images_dogbone_path = "/home/shanifi/code/pv-diffusion/exports/inference_dogbone_resume_all161_nomono_5/image"
# fake_images_dogbone_path = "/home/shanifi/code/pv-diffusion/exports/inference_dogbone_resume_all161_nomono_5/image_adjusted_brilliance"
fake_images_dogbone_path = "/home/shared/sc/generated_unlabeled/dogbone/image"

fake_images_paths = {
    "mono-c-Si": fake_images_mono_path,
    "multi-c-Si": fake_images_multi_path,
    "half-cut multi-c-Si": fake_images_multihalfcut_path,
    "IBC-dogbone": fake_images_dogbone_path,
}
real_images_path = "/home/shanifi/code/pv-diffusion/dataset/el_images_test/removed_padding"

# # Define base colors for each cell type
# cell_type_colors = {
#     "mono-c-Si": "#1f77b4",
#     "multi-c-Si": "#2ca02c",
#     "half-cut multi-c-Si": "#ff7f0e",
#     "IBC-dogbone": "#d62728",
# }

cell_type_colors = {
    "mono-c-Si": {"real": "#1f77b4", "fake": "#aec7e8"},  # Blue tones
    "multi-c-Si": {"real": "#2ca02c", "fake": "#98df8a"},  # Green tones
    "half-cut multi-c-Si": {"real": "#ff7f0e", "fake": "#ffbb78"},  # Orange tones
    "IBC-dogbone": {"real": "#d62728", "fake": "#ff9896"},  # Red tones
}

# Load model with pretrained weights
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
# Replace classifier to match your number of classes (30)
model.classifier[4] = nn.Conv2d(256, 30, kernel_size=1)
# Load your custom-trained model weights (after modifying the classifier!)
model.load_state_dict(torch.load("/home/shanifi/code/pv-diffusion/FID/deeplabv3_el_trained_earlystop.pth"))
model.eval()

# Wrap encoder
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self._out_feature = 2048 

    def forward(self, x):
        x = self.encoder(x)
        if isinstance(x, dict):
            x = x['out']
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)
    @property
    def num_features(self):  # Optional: TorchMetrics will call this if it exists
        return self._out_feature

feature_model = FeatureExtractor(model.backbone).to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # converts to float32 and scales to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalization for 3 channels
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img)

all_features = []
all_labels = []
all_types = []

for cell_type, fake_images_path in fake_images_paths.items():
    if not os.path.exists(fake_images_path):
        print(f"Path does not exist: {fake_images_path}")
    fake_images_names = sorted([f for f in os.listdir(fake_images_path) if f.endswith(".png")])
    fake_images = [load_image(os.path.join(fake_images_path, name)) for name in fake_images_names]

    real_images_names = set()
    # real_masks_list = os.listdir("/home/shanifi/code/pv-diffusion/dataset/el_masks_test/removed_padding/revised_binary_masks/multi")
    for image_name in fake_images_names:
        if image_name.endswith(".png"):
            image_name = image_name.split("-")[1]
            parts = image_name.split("_")
            real_image_name = "_".join(parts[:-1]) + ".png"
            # print(f"real image name: {real_image_name}")
            real_images_names.add(real_image_name)

    real_images_names = sorted(real_images_names)     
    real_images = [load_image(os.path.join(real_images_path, name)) for name in real_images_names if name.endswith(".png")]
    print(f"{cell_type}: {len(real_images)} real, {len(fake_images)} fake")

    # FID
    fid = FrechetInceptionDistance(feature=feature_model, normalize=True)
    fid = fid.to(DEVICE)
    fid_original = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    # KID
    num_real = len(real_images)
    num_fake = len(fake_images)
    min_samples = min(len(real_images), len(fake_images))
    subset_size = min(100, min_samples - 1) 
    kid = KernelInceptionDistance(feature=feature_model, subsets=10, subset_size=subset_size, normalize=True)
    kid = kid.to(DEVICE)
    kid_original = KernelInceptionDistance(subsets=10, subset_size=subset_size, normalize=True).to(DEVICE)


    real_tensors = torch.stack(real_images).to(DEVICE)
    fake_tensors = torch.stack(fake_images).to(DEVICE)

    real_loader = DataLoader(real_tensors, batch_size=16)
    fake_loader = DataLoader(fake_tensors, batch_size=16)

    # Extract features for t-SNE
    real_features = []
    fake_features = []

    with torch.no_grad():

        for batch in real_loader:
            fid.update(batch.to(DEVICE), real=True)
            fid_original.update(batch.to(DEVICE), real=True)
            kid.update(batch.to(DEVICE), real=True)
            kid_original.update(batch.to(DEVICE), real=True)
            features = feature_model(batch)
            real_features.append(features.cpu())
        for batch in fake_loader:
            fid.update(batch.to(DEVICE), real=False)
            fid_original.update(batch.to(DEVICE), real=False)
            kid.update(batch.to(DEVICE), real=False)
            kid_original.update(batch.to(DEVICE), real=False)
            features = feature_model(batch)
            fake_features.append(features.cpu())
        fid_score = fid.compute()
        fid_original_score = fid_original.compute()
        print("results for:", fake_images_path)
        print("FID:", fid_score.item())
        print("FID_original:", fid_original_score.item())
        kid_mean, kid_std = kid.compute()
        kid_original_mean, kid_original_std = kid_original.compute()
        print("KID mean:", kid_mean.item(), "±", kid_std.item())
        print("KID_original mean:", kid_original_mean.item(), "±", kid_original_std.item())

        print("Calculating extra metrics (Cosine, Euclidean, LPIPS)...")

        # Prepare features as numpy arrays if not already
        real_feats_np = real_features
        fake_feats_np = fake_features

        # # Cosine similarity
        # cos_sim_matrix = cosine_similarity(real_feats_np, fake_feats_np)
        # mean_cos_sim = np.mean(cos_sim_matrix)
        # print(f"Mean cosine similarity (real vs fake features): {mean_cos_sim:.4f}")

        # # Euclidean distance
        # euclid_dists = cdist(real_feats_np, fake_feats_np, metric='euclidean')
        # mean_euclid = np.mean(euclid_dists)
        # print(f"Mean Euclidean distance (real vs fake features): {mean_euclid:.4f}")

        # LPIPS perceptual distance
        lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
        lpips_model.eval()
        num_lpips_samples = min(len(real_tensors), len(fake_tensors))
        real_lpips = real_tensors[:num_lpips_samples]
        fake_lpips = fake_tensors[:num_lpips_samples]

        lpips_scores = []
        with torch.no_grad():
            for real_img, fake_img in zip(real_lpips, fake_lpips):
                real_img = real_img.unsqueeze(0).to(DEVICE)
                fake_img = fake_img.unsqueeze(0).to(DEVICE)
                dist = lpips_model(real_img, fake_img)
                lpips_scores.append(dist.item())
        mean_lpips = np.mean(lpips_scores)
        print(f"Mean LPIPS distance: {mean_lpips:.4f}")
    # PacMAP visualization
    # Stack feature tensors
    real_features = torch.cat(real_features).numpy()
    fake_features = torch.cat(fake_features).numpy()

    # Create labels
    # labels = ['Real'] * len(real_features) + ['Generated'] * len(fake_features)
    all_features.append(real_features)
    all_labels += ['Real'] * len(real_features)
    all_types += [cell_type] * len(real_features)
    all_features.append(fake_features)
    all_labels += ['Generated'] * len(fake_features)
    all_types += [cell_type] * len(fake_features)

# Reduce to 2D
all_features = np.vstack(all_features)
reducer = pacmap.PaCMAP(
    n_components=2,
    random_state=42,
    n_neighbors=17,
    MN_ratio=1,
    FP_ratio=2,
    # # num_iters=600,
    # apply_pca= True,
    )
embedding = reducer.fit_transform(all_features)

embedding = np.array(embedding)
label_array = np.array(all_labels)
type_array = np.array(all_types)

plt.figure(figsize=(9, 5), dpi=300)
for cell_type, base_color in cell_type_colors.items():
    # light = mcolors.to_rgba(base_color, alpha=0.4)
    light = base_color["real"]
    # dark = mcolors.to_rgba(base_color, alpha=1.0)
    dark = base_color["fake"]
    # darker = mcolors.to_rgba(base_color, alpha=1.0)

    # Real - Circles
    idx_real = (label_array == "Real") & (type_array == cell_type)
    plt.scatter(
        embedding[idx_real, 0],
        embedding[idx_real, 1],
        label=f"R - {cell_type}",
        c=[light],
        marker='o',
        edgecolors='k',
        s=80,
        linewidths=0.7,
        alpha=0.7,
        zorder=2
    )

    # Generated - Squares
    idx_fake = (label_array == "Generated") & (type_array == cell_type)
    plt.scatter(
        embedding[idx_fake, 0],
        embedding[idx_fake, 1],
        label=f"G - {cell_type}",
        c=[dark],
        marker='s',
        # edgecolors='k',
        s=30,
        linewidths=0.3,
        alpha=0.8,
        zorder=1
    )

# for cell_type in cell_type_colors:
#     idx = (label_array == "Real") & (type_array == cell_type)
#     if np.sum(idx) > 0:
#         x_mean = embedding[idx, 0].mean()
#         y_mean = embedding[idx, 1].mean()
#         plt.text(x_mean, y_mean, cell_type.split()[0], fontsize=9, weight='bold', ha='center')


plt.legend(loc="best", fontsize=12, markerscale=0.9, frameon=False)
# plt.title("PaCMAP: Real (Circles) vs. Generated (Squares) EL Images by Cell Type")
plt.xlabel("Component 1", fontsize=14,)
plt.ylabel("Component 2", fontsize=14,)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("FID/pacmap_all_types_styled.png", dpi=300)

# # t-SNE visualization
# real_features = []y
# avg_fake_features = []
# labels = []

# for real_name in real_images_names:
#     real_img = load_image(os.path.join(real_images_path, real_name))
#     real_tensor = real_img.unsqueeze(0).to(DEVICE)

#     # Extract real feature
#     with torch.no_grad():
#         real_feat = feature_model(real_tensor).cpu().numpy()
#         del real_tensor  # Free memory
#     real_features.append(real_feat)

#     # Get corresponding fake images (assumes real_name is part of fake image filename)
#     fake_subset = sorted([name for name in fake_images_names if real_name.replace(".png", "") in name])
#     print(f"{real_name} → {len(fake_subset)} fake images matched.")
#     fake_feats = []

#     with torch.no_grad():
#         for name in fake_subset:
#             fake_img = load_image(os.path.join(fake_images_path, name)).unsqueeze(0).to(DEVICE)
#             feat = feature_model(fake_img).squeeze(0).cpu().numpy()
#             fake_feats.append(feat)
#             del fake_img
#             torch.cuda.empty_cache()

#     avg_feat = np.mean(fake_feats, axis=0)

#     avg_fake_features.append(avg_feat)

#     labels.append(real_name)  # Label/group

# # Stack features
# all_feats = np.vstack([np.vstack(real_features), np.vstack(avg_fake_features)])
# all_labels = ['real_' + name for name in labels] + ['fake_' + name for name in labels]

# # t-SNE
# tsne = TSNE(n_components=2, perplexity=5, init='pca', learning_rate='auto', random_state=42, method='exact')
# features_2d = tsne.fit_transform(all_feats)

# # Plot
# plt.figure(figsize=(8, 6))
# real_idx = [i for i, l in enumerate(all_labels) if l.startswith('real')]
# fake_idx = [i for i, l in enumerate(all_labels) if l.startswith('fake')]

# plt.scatter(features_2d[real_idx, 0], features_2d[real_idx, 1], c='blue', label='Real', alpha=0.7, edgecolor='k')
# plt.scatter(features_2d[fake_idx, 0], features_2d[fake_idx, 1], c='red', label='Fake (avg. per condition)', alpha=0.7, edgecolor='k')

# for i in range(len(labels)):  # one per real-fake pair
#     real_point = features_2d[i]
#     fake_point = features_2d[i + len(labels)]
#     plt.plot([real_point[0], fake_point[0]], [real_point[1], fake_point[1]], 'gray', linestyle='--', alpha=0.5)

# plt.legend()
# # plt.title("t-SNE of Real vs. Avg. Mask-Conditioned Fakes per Real EL Image")
# plt.grid(True)
# plt.savefig("FID/tsne_multi_halfcut.png", dpi=300)
# print("saved t-SNE plot to FID/tsne_....png")

# # Step 1: Collect real features and all fake features
# real_features = []
# fake_features_all = []  # all individual fake features
# labels = []
# fake_group_indices = []  # stores (start, end) indices for each group of 10 fakes

# for real_name in real_images_names:
#     real_img = load_image(os.path.join(real_images_path, real_name))
#     real_tensor = real_img.unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         real_feat = feature_model(real_tensor).cpu().numpy()
#     real_features.append(real_feat)
#     labels.append(real_name)

#     # Load corresponding fake samples
#     fake_subset = sorted([name for name in fake_images_names if real_name.replace(".png", "") in name])
#     group_feats = []
#     for name in fake_subset:
#         fake_img = load_image(os.path.join(fake_images_path, name)).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             feat = feature_model(fake_img).squeeze(0).cpu().numpy()
#         group_feats.append(feat)
#         del fake_img
#         torch.cuda.empty_cache()

#     start_idx = len(fake_features_all)
#     fake_features_all.extend(group_feats)
#     end_idx = len(fake_features_all)
#     fake_group_indices.append((start_idx, end_idx))

# # Step 2: Run t-SNE on all real + fake features
# all_feats = np.vstack([np.vstack(real_features), np.vstack(fake_features_all)])
# tsne = TSNE(n_components=2, perplexity=5, init='pca', learning_rate='auto', random_state=42, method='exact')
# features_2d = tsne.fit_transform(all_feats)

# # Step 3: Split back into real and fake
# real_2d = features_2d[:len(real_features)]
# fake_2d = features_2d[len(real_features):]

# # Step 4: Plot
# plt.figure(figsize=(10, 8))
# plt.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', label='Real', alpha=0.7, edgecolor='k')

# fake_centroids_2d = []
# for i, (start, end) in enumerate(fake_group_indices):
#     group_points = fake_2d[start:end]
#     centroid = group_points.mean(axis=0)
#     fake_centroids_2d.append(centroid)

#     # Plot fake centroid
#     plt.scatter(centroid[0], centroid[1], c='red', edgecolor='k', alpha=0.8)

#     # Line connecting real to fake centroid
#     plt.plot([real_2d[i][0], centroid[0]], [real_2d[i][1], centroid[1]],
#              'gray', linestyle='--', alpha=0.5)

#     # PCA on 2D fake points to get ellipse
#     pca = PCA(n_components=2)
#     pca.fit(group_points)
#     width, height = 2 * np.sqrt(pca.explained_variance_)
#     angle = np.degrees(np.arctan2(*pca.components_[0][::-1]))
#     ellipse = Ellipse(xy=centroid,
#                       width=width,
#                       height=height,
#                       angle=angle,
#                       edgecolor='red',
#                       facecolor='none',
#                       linestyle=':',
#                       alpha=0.5)
#     plt.gca().add_patch(ellipse)

# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("FID/tsne_variance_ellipses_tsnespace.png", dpi=300)
# print("Saved t-SNE with variance ellipses in t-SNE space.")
