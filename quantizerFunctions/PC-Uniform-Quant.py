#!/usr/bin/env python3
"""
Per-channel uniform quantization of all floating-point weight tensors
in a Flow Matching model checkpoint, to n_bits precision.
"""

import argparse
import torch
import os

def uniform_quantize_tensor(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Uniformly quantize a tensor to n_bits precision.
    - For 1D tensors (e.g. biases), falls back to per-tensor quantization.
    - For ND tensors with ndim>=2, does per-channel quantization along dim=0.
    """
    if not tensor.is_floating_point() or n_bits >= 32:
        return tensor.clone()

    # total # of quantization levels
    q_levels = 2 ** n_bits - 1

    if tensor.ndim < 2:
        # per-tensor
        t_min = tensor.min()
        t_max = tensor.max()
        if t_max == t_min:
            return tensor.clone()
        scale = (t_max - t_min) / q_levels
        q = torch.round((tensor - t_min) / scale)
        return q * scale + t_min

    # per-channel along dim=0
    # compute per-channel min/max by reducing all other dims
    reduce_dims = list(range(tensor.ndim))
    reduce_dims.remove(0)
    t_min = tensor.amin(dim=reduce_dims, keepdim=True)
    t_max = tensor.amax(dim=reduce_dims, keepdim=True)

    # scale per channel
    scale = (t_max - t_min) / q_levels

    # mask out constant channels (t_max==t_min) so we don't divide by zero
    non_const = (t_max != t_min)

    # quantize + dequantize
    q = torch.round((tensor - t_min) / scale)
    tensor_q = q * scale + t_min

    # for constant channels, just copy original
    return torch.where(non_const, tensor_q, tensor)

def quantize_state_dict(state_dict: dict, n_bits: int) -> dict:
    """
    Apply per-channel uniform quantization to all float tensors in a state_dict.
    """
    q_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            q_state[k] = uniform_quantize_tensor(v, n_bits)
        else:
            q_state[k] = v
    return q_state

def main():
    parser = argparse.ArgumentParser(
        description="Per-channel uniform quantize a Flow Matching model checkpoint."
    )
    parser.add_argument(
        "model_path", type=str,
        help="Path to the .pth checkpoint (or state_dict) file."
    )
    parser.add_argument(
        "n_bits", type=int,
        help="Number of bits to quantize to (e.g. 8, 4, 2)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the original checkpoint (if omitted, saves to a new file)."
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")

    # find the state_dict
    if "model" in ckpt:
        sd = ckpt["model"]
        container = "model"
    else:
        sd = ckpt
        container = None

    print(f"Quantizing {len(sd)} tensors to {args.n_bits}-bit (per-channel)...")
    q_sd = quantize_state_dict(sd, args.n_bits)

    if container:
        ckpt["model"] = q_sd
    else:
        ckpt = q_sd

    # Save
    if args.overwrite:
        save_path = args.model_path
    else:
        base, ext = os.path.splitext(args.model_path)
        save_path = f"{base}_pcq{args.n_bits}{ext}"

    torch.save(ckpt, save_path)
    print(f"Quantized checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
