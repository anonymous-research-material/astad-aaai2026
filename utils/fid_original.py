import torch
import numpy as np
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import PIL.Image as Image
import torchvision.transforms as transforms
# from torchmetrics.image.scores import InceptionScore
# 
def load_image_as_uint8_tensor(path):
    img = Image.open(path).convert('RGB')  # ensure 3-channel RGB
    tensor = transforms.ToTensor()(img) * 255  # scale to [0, 255]
    tensor = tensor.to(torch.uint8)  # convert dtype
    return tensor
# 
# def load_image_as_float_tensor(path):
#     img = Image.open(path).convert('RGB')
#     tensor = transforms.ToTensor()(img)  # float32 in [0.0, 1.0]
#     return tensor

fake_images_path = "/home/shanifi/code/pv-diffusion/exports/inference_multi_resume7_nopadding_withRF/image"
fake_images_names = os.listdir(fake_images_path)
print(f"Number of fake images: {len(fake_images_names)}")
fake_images = [load_image_as_uint8_tensor(os.path.join(fake_images_path, name)) for name in fake_images_names if name.endswith(".png")]

real_images_names = set()
# real_masks_list = os.listdir("/home/shanifi/code/pv-diffusion/dataset/el_masks_test/removed_padding/revised_binary_masks/multi")
for image_name in fake_images_names:
    if image_name.endswith(".png"):
        image_name = image_name.split("-")[1]
        parts = image_name.split("_")
        real_image_name = "_".join(parts[:-1]) + ".png"
        print(f"real image name: {real_image_name}")
        real_images_names.add(real_image_name)
        
real_images_path = "/home/shanifi/code/pv-diffusion/dataset/el_images_test/removed_padding"
real_images = [load_image_as_uint8_tensor(os.path.join(real_images_path, name)) for name in real_images_names if name.endswith(".png")]
# FID
fid = FrechetInceptionDistance(feature=2048)
fid.update(torch.stack(real_images), real=True)
fid.update(torch.stack(fake_images), real=False)
fid_score = fid.compute()
print("FID:", fid_score.item())

# KID
num_real = len(real_images)
num_fake = len(fake_images)
min_samples = min(len(real_images), len(fake_images))
subset_size = min(100, min_samples - 1) 
kid = KernelInceptionDistance(subsets=100, subset_size=subset_size, feature=2048)
kid.update(torch.stack(real_images), real=True)
kid.update(torch.stack(fake_images), real=False)
kid_mean, kid_std = kid.compute()
print("KID mean:", float(kid_mean), "±", float(kid_std))