import sys
import torch
import torch.nn as nn

# Add your project root to the Python path
sys.path.append("/home/shanifi/code/pv-diffusion")

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import Unet  # Replace with actual U-Net class if different

# 1. Define the model architecture
# ⚠️ Adjust these arguments based on how your model was configured during training
denoise_model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=6,  # if input + condition are concatenated
    out_dim=3,
    with_time_emb=True,
    resnet_block_groups=8,
    # Add any additional args used during training
)

model = GaussianDiffusion(
    denoise_fn=denoise_model,
    image_size=512,
    timesteps=1000,
    loss_type='l2',  # or 'l2', depending on what you used
)

# 2. Load the state_dict
checkpoint = torch.load("/home/shanifi/code/pv-diffusion/results/train_mono_resume_all_nopadding_early_stopping_with_rf/model-best.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# 3. Wrap the model for tracing (avoids *args, **kwargs)
class ExportableModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, condition):
        return self.model(x, condition_tensors=condition)

wrapped_model = ExportableModel(model)

# 4. Dummy inputs (match what model expects)
x = torch.randn(1, 3, 512, 512)          # main image input
condition = torch.randn(1, 3, 512, 512)  # condition input (channels must match)

# 5. Export to TorchScript
traced = torch.jit.trace(wrapped_model, (x, condition))
traced.save("model_traced_from_state_dict.pt")

print("✅ Model exported as 'model_traced_from_state_dict.pt'")
