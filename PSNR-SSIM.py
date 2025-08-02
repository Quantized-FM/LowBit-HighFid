import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.model_configs import instantiate_model
from models.nn import timestep_embedding

# ──────────────
# Config
# ──────────────
SEED = 0
DATASET_NAME = "CelebA"
torch.manual_seed(SEED)
np.random.seed(SEED)

LOGFILE = "celeba_psnr_ssim.log"
def write_log(msg):
    with open(LOGFILE, 'a') as f:
        f.write(msg.strip() + '\n')
    print(msg)

# ──────────────
# Load CelebA Data
# ──────────────
data_root = "/path/to/celeba/data/"
img_dir = Path(data_root) / "img_align_celeba_64"
transform = transforms.Compose([transforms.ToTensor()])

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        img_path = self.img_dir / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label

dataset = CelebADataset(img_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# ──────────────
# PSNR / SSIM Helpers
# ──────────────
def generate_images(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            timesteps = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
            extra = {}
            out = model(batch, timesteps, extra)
            outputs.append(out.cpu())
            break  # Remove this to process ALL batches
    return torch.cat(outputs, dim=0)

def compute_psnr(ref, test):
    mse = torch.mean((ref - test) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ──────────────────────────────
# Prepare reference (Full precision)
# ──────────────────────────────
base_checkpoint_path = f"/path/to/trained/models/{DATASET_NAME}/"
args_filepath = base_checkpoint_path + f"{DATASET_NAME}-args.json"
with open(args_filepath, 'r') as f:
    args_dict = json.load(f)

device = torch.device('cuda:0')
model_key = "imagenet"  # For CelebA, use "imagenet" in your model configs

# Load Full model
full_model_path = Path(base_checkpoint_path + f"{DATASET_NAME}.pth")
full_model = instantiate_model(
    architechture=model_key,
    is_discrete='discrete_flow_matching' in args_dict and args_dict['discrete_flow_matching'],
    use_ema=args_dict['use_ema']
)
checkpoint_full = torch.load(full_model_path, map_location="cpu")
full_model.load_state_dict(checkpoint_full["model"])
full_model.to(device)

write_log("[INFO] Generating reference images from Full precision model...")
reference_images = generate_images(full_model, loader, device)
write_log("[INFO] Reference images generated.")

del full_model
torch.cuda.empty_cache()

# ──────────────────────────────
# Loop over Quantization Methods
# ──────────────────────────────
quantization_methods = ['Uniform', 'OT', 'PWL', 'LogBase2']
bit_widths = ['2', '3', '4', '5', '6', '7', '8', 'Full']
results = []

for method in quantization_methods:
    for bits in bit_widths:
        write_log(f"[INFO] Processing {method} at {bits}-bit...")
        try:
            checkpoint_path = Path(
                base_checkpoint_path + (f"{DATASET_NAME}.pth" if bits == "Full" else f"{method}-{DATASET_NAME}_q{bits}.pth")
            )

            model = instantiate_model(
                architechture=model_key,
                is_discrete='discrete_flow_matching' in args_dict and args_dict['discrete_flow_matching'],
                use_ema=args_dict['use_ema']
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(device)

            if bits == "Full":
                avg_psnr = float('nan')
                avg_ssim = float('nan')
            else:
                quantized_images = generate_images(model, loader, device)
                avg_psnr = compute_psnr(reference_images, quantized_images).item()
                avg_ssim = compute_ssim(reference_images, quantized_images).item()

            write_log(f"[RESULT] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
            results.append({"method": method, "bitwidth": bits, "PSNR": avg_psnr, "SSIM": avg_ssim})

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            write_log(f"[ERROR] {method} {bits} - {type(e).__name__}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv(f"{DATASET_NAME.lower()}_psnr_ssim_results2.csv", index=False)
write_log(f"PSNR & SSIM metrics saved to {DATASET_NAME.lower()}_psnr_ssim_results2.csv")
