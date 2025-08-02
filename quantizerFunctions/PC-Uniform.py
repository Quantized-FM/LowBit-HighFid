"""
quantize_model.py

Uniformly quantize all weight tensors in a Flow Matching model checkpoint
to n_bits and save the quantized checkpoint.
"""

import argparse
import torch
import os

def uniform_quantize_tensor(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Uniformly quantize a tensor to n_bits precision.
    """
    if not tensor.is_floating_point():
        return tensor

    t_min = tensor.min()
    t_max = tensor.max()
    # Edge case: constant tensor
    if t_max == t_min:
        return tensor.clone()

    # Number of quantization levels
    q_levels = 2 ** n_bits - 1

    # Scale and zero-point
    scale = (t_max - t_min) / q_levels

    # Quantize:
    q = torch.round((tensor - t_min) / scale)
    # Dequantize:
    tensor_q = q * scale + t_min
    return tensor_q

def quantize_state_dict(state_dict: dict, n_bits: int) -> dict:
    """
    Apply uniform quantization to all float tensors in a state_dict.
    """
    q_state = {}
    for k, v in state_dict.items():
        # Only quantize float tensors (e.g., weights, biases)
        if isinstance(v, torch.Tensor):
            q_state[k] = uniform_quantize_tensor(v, n_bits)
        else:
            q_state[k] = v
    return q_state

def main():
    parser = argparse.ArgumentParser(
        description="Uniformly quantize a Flow Matching model checkpoint."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the .pth checkpoint (or state_dict) file."
    )
    parser.add_argument(
        "n_bits",
        type=int,
        help="Number of bits to quantize to (e.g. 8, 4, 2)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original checkpoint (if omitted, saves to a new file)."
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")

    # Determine where the weights live
    if "model" in ckpt:
        sd = ckpt["model"]
        container = "model"
    else:
        sd = ckpt
        container = None

    # Quantize
    print(f"Quantizing {len(sd)} tensors to {args.n_bits}-bit...")
    q_sd = quantize_state_dict(sd, args.n_bits)

    # Put back
    if container:
        ckpt["model"] = q_sd
    else:
        ckpt = q_sd

    # Save
    if args.overwrite:
        save_path = args.model_path
    else:
        base, ext = os.path.splitext(args.model_path)
        save_path = f"{base}_q{args.n_bits}{ext}"

    torch.save(ckpt, save_path)
    print(f"Quantized checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
