"""
Per-channel 1-D optimal-transport (equal-mass) quantization of all weight tensors
in a Flow Matching model checkpoint, saving the result to a new file.
"""

import argparse
import torch
import numpy as np
import os
import copy


def quantize_tensor_ot(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Per-channel equal-mass (optimal transport) quantization of a tensor to 'bits' precision.
    """
    # Only quantize float tensors; skip if bit-width >=32 or too few elements
    if not tensor.is_floating_point() or bits >= 32 or tensor.numel() < 2 ** bits:
        return tensor.clone()

    arr = tensor.detach().cpu().numpy()

    # Helper to quantize a 1D array
    def _quantize_flat(flat):
        K = min(2 ** bits, flat.size)
        idx = np.argsort(flat)
        flat_sorted = flat[idx]
        # split into K chunks of (approximately) equal mass
        chunks = np.array_split(flat_sorted, K)
        centroids = np.array([chunk.mean() for chunk in chunks], dtype=flat.dtype)
        quant_sorted = np.empty_like(flat_sorted)
        start = 0
        for k, chunk in enumerate(chunks):
            L = len(chunk)
            quant_sorted[start:start + L] = centroids[k]
            start += L
        inv = np.argsort(idx)
        return quant_sorted[inv]

    # If multi-dimensional, process independently over the first axis (channels)
    if arr.ndim > 1:
        out = np.zeros_like(arr)
        C = arr.shape[0]
        for c in range(C):
            flat = arr[c].reshape(-1)
            out[c] = _quantize_flat(flat).reshape(arr.shape[1:])
    else:
        out = _quantize_flat(arr.reshape(-1)).reshape(arr.shape)

    return torch.from_numpy(out).to(tensor.device)


def quantize_state_dict_ot(state_dict: dict, bits: int) -> dict:
    """
    Apply OT quantization to all float tensors in a state dict.
    """
    q_state = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            q_state[key] = quantize_tensor_ot(val, bits)
        else:
            q_state[key] = val
    return q_state


def main():
    parser = argparse.ArgumentParser(
        description="OT quantize a Flow Matching model checkpoint to a specified bit-width."
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the .pth checkpoint or state_dict file'
    )
    parser.add_argument(
        'n_bits',
        type=int,
        help='Number of bits for quantization (e.g. 8, 4, 2)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the original checkpoint file'
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location='cpu')
    if 'model' in ckpt:
        sd = ckpt['model']
        container = 'model'
    else:
        sd = ckpt
        container = None

    # Quantize
    print(f"Applying OT quantization to {len(sd)} tensors at {args.n_bits}-bit...")
    q_sd = quantize_state_dict_ot(sd, args.n_bits)

    # Repackage
    if container:
        ckpt['model'] = q_sd
    else:
        ckpt = q_sd

    # Determine save path
    if args.overwrite:
        save_path = args.model_path
    else:
        base, ext = os.path.splitext(args.model_path)
        save_path = f"{base}_ot_q{args.n_bits}{ext}"

    torch.save(ckpt, save_path)
    print(f"Quantized checkpoint saved to {save_path}")

if __name__ == '__main__':
    main()
