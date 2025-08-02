"""
Per-channel 1-D Piecewise Linear Quantization (PWLQ) of all floating-point weight tensors
in a Flow Matching model checkpoint, saving the quantized checkpoint.

This method divides the sorted tensor values into K=2^bits segments by quantile-based breakpoints
and uses the midpoint of each segment as the reconstruction level.
"""

import argparse
import torch
import numpy as np
import os


def quantize_tensor_pwl(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Per-channel piecewise linear quantization of a tensor to `bits` precision.
    """
    if not tensor.is_floating_point() or bits >= 32 or tensor.numel() < 2 ** bits:
        return tensor.clone()

    arr = tensor.detach().cpu().numpy()

    def _quantize_flat(flat):
        K = 2 ** bits
        # sort
        flat_sorted = np.sort(flat)
        N = flat_sorted.size
        # breakpoints at quantiles 0,1/K,...,1
        probs = np.linspace(0, 1, K + 1)
        # interpolate breakpoints
        breakpoints = np.interp(probs, np.linspace(0, 1, N), flat_sorted)
        # reconstruction midpoints
        mids = (breakpoints[:-1] + breakpoints[1:]) / 2
        # assign each value to its segment
        idx = np.searchsorted(breakpoints, flat, side='right') - 1
        idx = np.clip(idx, 0, K - 1)
        return mids[idx]

    # apply per-channel
    if arr.ndim > 1:
        out = np.zeros_like(arr)
        C = arr.shape[0]
        for c in range(C):
            flat = arr[c].ravel()
            out[c] = _quantize_flat(flat).reshape(arr.shape[1:])
    else:
        out = _quantize_flat(arr.ravel()).reshape(arr.shape)

    return torch.from_numpy(out).to(tensor.device)


def quantize_state_dict_pwl(state_dict: dict, bits: int) -> dict:
    """
    Apply PWL quantization to all float tensors in the state dict.
    """
    q_state = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            q_state[key] = quantize_tensor_pwl(val, bits)
        else:
            q_state[key] = val
    return q_state


def main():
    parser = argparse.ArgumentParser(
        description="PWL-quantize a Flow Matching model checkpoint to a specified bit-width."
    )
    parser.add_argument('model_path', type=str,
                        help='Path to the .pth checkpoint or state_dict file')
    parser.add_argument('n_bits', type=int,
                        help='Number of bits for quantization (e.g. 8, 4, 2)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the original checkpoint')
    args = parser.parse_args()

    # load checkpoint
    ckpt = torch.load(args.model_path, map_location='cpu')
    if 'model' in ckpt:
        sd = ckpt['model']
        container = 'model'
    else:
        sd = ckpt
        container = None

    print(f"Applying PWL quantization to {len(sd)} tensors at {args.n_bits}-bit...")
    q_sd = quantize_state_dict_pwl(sd, args.n_bits)

    if container:
        ckpt['model'] = q_sd
    else:
        ckpt = q_sd

    base, ext = os.path.splitext(args.model_path)
    save_path = args.model_path if args.overwrite else f"{base}_pwl_q{args.n_bits}{ext}"
    torch.save(ckpt, save_path)
    print(f"Quantized checkpoint saved to {save_path}")

if __name__ == '__main__':
    main()
