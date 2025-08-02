import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------- CONFIG ---------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET = "FashionMNIST"
DATA_ROOT = "/path/to/fashionmnist/data"
BASE_CKPT_PATH = "/path/to/trained/models"
SAVE_PLOT_DIR = "latent_variance_plots_fashionMnist"
os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

quantization_methods = ['Uniform', 'OT', 'PWL', 'LogBase2']
bit_widths = ['2', '3', '4', '5', '6', '7', '8', 'Full']

LOGFILE = f"latent_shrinkage_{DATASET.lower()}.log"
def write_log(msg):
    with open(LOGFILE, 'a') as f:
        f.write(msg.strip() + '\n')
    print(msg)
def get_loader_mnist(num_samples=10000):
    transform = transforms.Compose([transforms.ToTensor()])
    fullset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=transform)
    indices = np.random.choice(len(fullset), min(num_samples, len(fullset)), replace=False)
    subset = Subset(fullset, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    batch = next(iter(loader))
    # torchvision: tuple (images, labels)
    if isinstance(batch, tuple):
        images = batch[0]
    elif torch.is_tensor(batch):
        images = batch
    elif isinstance(batch, list):
        if torch.is_tensor(batch[0]) and batch[0].ndim == 4 and batch[0].shape[0] == len(subset):
            images = batch[0]
        elif torch.is_tensor(batch[0]) and batch[0].ndim == 3:
            images = torch.stack(batch)
        else:
            raise RuntimeError(f"Unhandled list entry shape: {batch[0].shape}")
    else:
        raise RuntimeError(f"Unexpected batch type: {type(batch)}")
    if images.ndim == 5:
        images = images.squeeze(0)
    print(f"[DEBUG] Final images shape: {images.shape}")
    return images.numpy()

# --------- ANALYSIS ---------
def analyze_latent_shrinkage(latents, method, bits, dataset, save_dir=None, verbose=True):
    var_per_dim = latents.var(axis=0)
    mean_var = var_per_dim.mean()
    std_var = var_per_dim.std()
    collapsed_dims = np.sum(var_per_dim < 1e-4)
    if verbose:
        msg = (f"{dataset} | {method} @ {bits}-bit: mean_var={mean_var:.6f}, "
               f"std_var={std_var:.6f}, collapsed={collapsed_dims}/{len(var_per_dim)}")
        write_log(msg)
    plt.figure(figsize=(7,3))
    plt.boxplot(var_per_dim, vert=False, patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.5))
    plt.title(f"{dataset}: Latent Variance ({method} @ {bits}-bit)")
    plt.xlabel("Variance per latent dimension")
    plt.yticks([])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/latent_variance_{dataset}_{method}_{bits}bit.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return mean_var, std_var, collapsed_dims

# --------- LATENT EXTRACTION (replace with your own if needed) ---------
from models.model_configs import instantiate_model
from models.nn import timestep_embedding

def extract_unet_latents(model, data, device):
    net = model.model if hasattr(model, "model") else model
    net.eval()
    latents_all = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[i:i+batch_size]).float().to(device)
            timesteps = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
            emb = net.time_embed(timestep_embedding(timesteps, net.model_channels).to(device))
            x = batch
            for block in net.input_blocks:
                x = block(x, emb)
            x = net.middle_block(x, emb)
            latents = torch.flatten(x, start_dim=1)
            latents_all.append(latents.detach().cpu().numpy())
    latents_all = np.concatenate(latents_all, axis=0)
    # Dimensionality reduction (optional)
    latents_pca = PCA(n_components=20, random_state=SEED).fit_transform(latents_all)
    return latents_pca

# --------- MAIN LOOP ---------
results = []

test_images = get_loader_mnist()
args_filepath = BASE_CKPT_PATH + f"{DATASET}-args.json"
with open(args_filepath, 'r') as f:
    args_dict = json.load(f)

model_key = "fashionmnist"
device = torch.device('cuda:0')

for method in quantization_methods:
    for bits in bit_widths:
        write_log(f"\n[INFO] Processing {DATASET} {method} at {bits}-bit...")
        try:
            checkpoint_path = Path(
                BASE_CKPT_PATH + (f"{DATASET}.pth" if bits == "Full" else f"{method}-{DATASET}_q{bits}.pth")
            )
            model = instantiate_model(
                architechture=model_key,
                is_discrete='discrete_flow_matching' in args_dict and args_dict['discrete_flow_matching'],
                use_ema=args_dict['use_ema']
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(device)

            latents = extract_unet_latents(model, test_images, device)
            mean_var, std_var, collapsed_dims = analyze_latent_shrinkage(
                latents, method, bits, DATASET, save_dir=SAVE_PLOT_DIR
            )
            results.append({
                "dataset": DATASET,
                "method": method,
                "bitwidth": bits,
                "latent_mean_var": mean_var,
                "latent_std_var": std_var,
                "latent_collapsed": collapsed_dims,
            })

            del model, latents
            torch.cuda.empty_cache()
        except Exception as e:
            write_log(f"[ERROR] {DATASET} {method} {bits} - {type(e).__name__}: {e}")

# --------- SAVE ---------
df = pd.DataFrame(results)
df.to_csv(f"latent_shrinkage_summary_{DATASET.lower()}.csv", index=False)
write_log(f"All latent shrinkage stats saved to latent_shrinkage_summary_{DATASET.lower()}.csv")
