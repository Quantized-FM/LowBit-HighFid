# Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching

This repo contains code necessary to replicate, reproduce and validate results from the paper "Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching," submitted to AAAI 2026. 

## Training Flow Matching Models
All models have been trained according to the official Flow Matching repository (accessible through this [link](https://github.com/facebookresearch/flow_matching). 5 FM models have been trained in total, corresponding to each of: 
- MNIST
- FashionMNIST
- CIFAR-10
- CelebA
- ImageNet
  
(In order of increasing data complexity)

Training hyperparameters have been included in the corresponding {dataset}-args.json files (found through ```./args```). The following table includes details of the training process. 

| Dataset | Epochs | Size | FID @50,000 samples |
|---|---|---|---| 
| MNIST | 500 | 28x28 | 0.671 |
| FashionMNIST | 500 | 28x28 | 2.891 |
| CIFAR-10 | 500 | 32x32 | 3.204 |
| CelebA | 100 | 64x64 | 4.318 |
| ImageNet | 86 | 64x64 | 14.437 |

Training was done on a single A100 GPU.
### ImageNet, CIFAR-10 and CelebA
To train these three datasets, researchers are encouraged to follow the same procedure as highlighted in ```flow_matching/exmaples/image```. For each of the three, the following arguments can be used for ```train.py``` directly: 

ImageNet: 
```python
python train.py \
--data_path=path/to/resized/ImageNet/data \
--batch_size=64 \
--accum_iter=1 \
--eval_frequency=50 \
--decay_lr \
--compute_fid \
--ode_method dopri5 \
--ode_options '{"atol": 1e-5, "rtol":1e-5}'
```
CelebA: 
```python
python train.py \
--dataset=celeba \
--data_path=/path/to/resized/celeba/data \
--batch_size=64 \
--accum_iter=1 \
--eval_frequency=2 \
--decay_lr \
--ode_method dopri5 \
--ode_options '{"atol": 1e-5, "rtol":1e-5}'
```
CIFAR-10: 
```python
python train.py \
--dataset=cifar10 \
--batch_size=64 \
--nodes=1 \
--accum_iter=1 \
--eval_frequency=100 \
--epochs=3000 \
--class_drop_prob=1.0 \
--cfg_scale=0.0 \
--compute_fid \
--ode_method heun2 \
--ode_options '{"nfe": 50}' \
--use_ema \
--edm_schedule \
--skewed_timesteps
```

### MNIST and FashionMNIST 
To train these two datasets, several changes need to be made to the code-base as defined through ```flow_matching/examples/image``` in the official repo. These changes will be noted below: 
1. Update ```examples/image/models/model_configs.py```:
Add the following to ```MODEL_CONFIGS```
```python
    "mnist": {
        "in_channels": 1,
        "model_channels": 128,
        "out_channels": 1,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "fashionmnist": {
        "in_channels": 1,
        "model_channels": 128,
        "out_channels": 1,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
```
2. Update ```examples/image/train.py```:
```python
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif args.dataset == "mnist":
        dataset_train = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif args.dataset == "fashionmnist":
        dataset_train = datasets.FashionMNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")
```
3. Check/Update Transforms in ```examples/image/training/data_transform.py```:
```python
from torchvision.transforms.v2 import Lambda

def get_train_transform():
    transform_list = [
        ToImage(),
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
        Lambda(lambda x: x if x.shape[0] == 1 else x.mean(dim=0, keepdim=True)),  # Ensures 1 channel
    ]
    return Compose(transform_list)
```
4. In ```examples/image/training/eval_loop.py```:
Add this helper function at the top:
```python
def ensure_three_channels(imgs):
    # imgs: (B, C, H, W)
    if imgs.shape[1] == 1:
        return imgs.repeat(1, 3, 1, 1)
    return imgs
```
Then, in the evaluation loop, change 
```python
fid_metric.update(samples, real=True)
```
to 
```python
fid_metric.update(ensure_three_channels(samples), real=True)
```
The same will apply for the generated samples: 
```python
fid_metric.update(ensure_three_channels(synthetic_samples), real=False)
```
5. Also in ```examples/image/training/eval_loop.py```:
Change
```python
for batch_index, image_np in enumerate(images_np):
    image_dir = Path(args.output_dir) / "fid_samples"
    os.makedirs(image_dir, exist_ok=True)
    image_path = (
        image_dir
        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
    )
    PIL.Image.fromarray(image_np, "RGB").save(image_path)
```
To: 
```python
for batch_index, image_np in enumerate(images_np):
    image_dir = Path(args.output_dir) / "fid_samples"
    os.makedirs(image_dir, exist_ok=True)
    image_path = (
        image_dir
        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
    )
    # Ensure 3 channels for RGB saving
    if image_np.ndim == 2:
        image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    PIL.Image.fromarray(image_np, "RGB").save(image_path)
```

For the sake of consistency, we would recommend cloning a separate version of the flow_matching repository and making these changes in there. Then, you can train the (Fashion)MNIST model here directly without changing the original codebase for the other datasets. Further details regarding the training can be found in the official repository linked in the beginning of this document. 

## Quantization
Several quantization (per-channel) helper files have been provided in this repository through ```./quantizerFunctions```. We have used these functions on the trained ```.pth``` files to simulate quantization down to 8, 7, 6, 5, 4, 3, and 2 bits (under different quantization frameworks). The helper files will return the model in quantized/dequantized format and save as ```.pth``` files. These new files can be used for evaluation. 
1. You are able to pass these files directly through ```train.py``` in the official FM repo with the flag ```---eval_only``` to generate samples, save them and compute corresponding FIDs.
2. You can pass these files into the other files that come with **this** repo to conduct similar analyses to our paper.

## Analysis


