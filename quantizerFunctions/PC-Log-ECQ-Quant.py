"""
Per-channel quantization of model checkpoints using:
1. Non-Uniform Logarithmic Quantization
2. Entropy Constrained Quantization (ECQ)
"""

import argparse
import torch
import numpy as np
import os
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def quantize_tensor_log(tensor: torch.Tensor, bits: int, base: float = 2.0) -> torch.Tensor:
    """
    Non-uniform logarithmic quantization of a tensor.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization
        base: Base for logarithmic spacing (default: 2.0)
    """
    if not tensor.is_floating_point() or bits >= 32 or tensor.numel() < 2 ** bits:
        return tensor.clone()
    
    arr = tensor.detach().cpu().numpy()
    
    def _quantize_log_flat(flat):
        K = min(2 ** bits, flat.size)
        if K <= 1:
            return np.full_like(flat, flat.mean())
        
        # Handle signs separately
        pos_mask = flat > 0
        neg_mask = flat < 0
        zero_mask = flat == 0
        
        result = np.zeros_like(flat)
        
        # Process positive values
        if np.any(pos_mask):
            pos_vals = flat[pos_mask]
            min_pos, max_pos = pos_vals.min(), pos_vals.max()
            
            if min_pos == max_pos:
                result[pos_mask] = min_pos
            else:
                # Logarithmic spacing for positive values
                log_min = np.log(min_pos) / np.log(base)
                log_max = np.log(max_pos) / np.log(base)
                
                # Use half the quantization levels for positive values
                k_pos = K // 2 if np.any(neg_mask) else K
                log_levels = np.linspace(log_min, log_max, k_pos)
                pos_centroids = base ** log_levels
                
                # Assign each positive value to nearest centroid
                distances = cdist(pos_vals.reshape(-1, 1), pos_centroids.reshape(-1, 1))
                assignments = np.argmin(distances, axis=1)
                result[pos_mask] = pos_centroids[assignments]
        
        # Process negative values symmetrically
        if np.any(neg_mask):
            neg_vals = flat[neg_mask]
            min_neg, max_neg = neg_vals.min(), neg_vals.max()
            
            if min_neg == max_neg:
                result[neg_mask] = min_neg
            else:
                # Convert to positive, apply log quantization, convert back
                pos_neg_vals = -neg_vals
                min_pos_neg, max_pos_neg = pos_neg_vals.min(), pos_neg_vals.max()
                
                log_min = np.log(min_pos_neg) / np.log(base)
                log_max = np.log(max_pos_neg) / np.log(base)
                
                k_neg = K // 2
                log_levels = np.linspace(log_min, log_max, k_neg)
                neg_centroids = -(base ** log_levels)
                
                distances = cdist(neg_vals.reshape(-1, 1), neg_centroids.reshape(-1, 1))
                assignments = np.argmin(distances, axis=1)
                result[neg_mask] = neg_centroids[assignments]
        
        # Handle zeros (assign to smallest magnitude centroid)
        if np.any(zero_mask):
            all_centroids = result[result != 0]
            if len(all_centroids) > 0:
                closest_to_zero = all_centroids[np.argmin(np.abs(all_centroids))]
                result[zero_mask] = closest_to_zero
            else:
                result[zero_mask] = 0
        
        return result
    
    # Process per-channel for multi-dimensional tensors
    if arr.ndim > 1:
        out = np.zeros_like(arr)
        C = arr.shape[0]
        for c in range(C):
            flat = arr[c].reshape(-1)
            out[c] = _quantize_log_flat(flat).reshape(arr.shape[1:])
    else:
        out = _quantize_log_flat(arr.reshape(-1)).reshape(arr.shape)
    
    return torch.from_numpy(out).to(tensor.device)


def quantize_tensor_ecq(tensor: torch.Tensor, bits: int, max_iter: int = 100) -> torch.Tensor:
    """
    Entropy Constrained Quantization (ECQ) of a tensor.
    
    Uses Lloyd's algorithm to minimize rate-distortion cost.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization
        max_iter: Maximum iterations for Lloyd's algorithm
    """
    if not tensor.is_floating_point() or bits >= 32 or tensor.numel() < 2 ** bits:
        return tensor.clone()
    
    arr = tensor.detach().cpu().numpy()
    
    def _quantize_ecq_flat(flat):
        K = min(2 ** bits, flat.size)
        if K <= 1:
            return np.full_like(flat, flat.mean())
        
        # Initialize centroids using k-means++
        try:
            kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=max_iter, random_state=42)
            labels = kmeans.fit_predict(flat.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.flatten()
        except:
            # Fallback to uniform initialization if k-means fails
            centroids = np.linspace(flat.min(), flat.max(), K)
            labels = np.digitize(flat, centroids) - 1
            labels = np.clip(labels, 0, K - 1)
        
        # Lloyd's algorithm for ECQ
        prev_cost = float('inf')
        for iteration in range(max_iter):
            # Assignment step: assign each point to nearest centroid
            distances = np.abs(flat.reshape(-1, 1) - centroids.reshape(1, -1))
            new_labels = np.argmin(distances, axis=1)
            
            # Update step: recompute centroids
            new_centroids = np.zeros(K)
            for k in range(K):
                mask = new_labels == k
                if np.any(mask):
                    new_centroids[k] = flat[mask].mean()
                else:
                    new_centroids[k] = centroids[k]
            
            # Compute cost (MSE + entropy regularization)
            mse = np.sum((flat - new_centroids[new_labels]) ** 2) / len(flat)
            
            # Simple entropy estimate based on cluster populations
            _, counts = np.unique(new_labels, return_counts=True)
            probs = counts / len(flat)
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            
            # Combined cost (weighted sum of distortion and rate)
            lambda_reg = 0.1  # Regularization parameter
            cost = mse + lambda_reg * entropy
            
            # Check for convergence
            if abs(prev_cost - cost) < 1e-6:
                break
            
            prev_cost = cost
            centroids = new_centroids
            labels = new_labels
        
        # Final quantization
        result = centroids[labels]
        return result
    
    # Process per-channel for multi-dimensional tensors
    if arr.ndim > 1:
        out = np.zeros_like(arr)
        C = arr.shape[0]
        for c in range(C):
            flat = arr[c].reshape(-1)
            out[c] = _quantize_ecq_flat(flat).reshape(arr.shape[1:])
    else:
        out = _quantize_ecq_flat(arr.reshape(-1)).reshape(arr.shape)
    
    return torch.from_numpy(out).to(tensor.device)


def quantize_state_dict(state_dict: dict, bits: int, method: str = 'log', **kwargs) -> dict:
    """
    Apply quantization to all float tensors in a state dict.
    
    Args:
        state_dict: Model state dictionary
        bits: Number of bits for quantization
        method: Quantization method ('log' or 'ecq')
        **kwargs: Additional arguments for quantization methods
    """
    q_state = {}
    
    if method == 'log':
        quantize_fn = lambda t: quantize_tensor_log(t, bits, **kwargs)
    elif method == 'ecq':
        quantize_fn = lambda t: quantize_tensor_ecq(t, bits, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            q_state[key] = quantize_fn(val)
        else:
            q_state[key] = val
    
    return q_state


def compute_quantization_stats(original_dict: dict, quantized_dict: dict) -> dict:
    """
    Compute statistics comparing original and quantized state dictionaries.
    """
    total_mse = 0.0
    total_elements = 0
    layer_stats = {}
    
    for key in original_dict:
        if isinstance(original_dict[key], torch.Tensor) and original_dict[key].is_floating_point():
            orig = original_dict[key].float()
            quant = quantized_dict[key].float()
            
            mse = torch.mean((orig - quant) ** 2).item()
            psnr = -10 * np.log10(mse + 1e-12)
            
            layer_stats[key] = {
                'mse': mse,
                'psnr': psnr,
                'shape': list(orig.shape),
                'elements': orig.numel()
            }
            
            total_mse += mse * orig.numel()
            total_elements += orig.numel()
    
    overall_mse = total_mse / total_elements if total_elements > 0 else 0.0
    overall_psnr = -10 * np.log10(overall_mse + 1e-12)
    
    return {
        'overall_mse': overall_mse,
        'overall_psnr': overall_psnr,
        'layer_stats': layer_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Advanced quantization of model checkpoints using logarithmic or ECQ methods."
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
        '--method',
        type=str,
        choices=['log', 'ecq'],
        default='log',
        help='Quantization method: "log" for logarithmic, "ecq" for entropy constrained'
    )
    parser.add_argument(
        '--log-base',
        type=float,
        default=2.0,
        help='Base for logarithmic quantization (default: 2.0)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Maximum iterations for ECQ (default: 100)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the original checkpoint file'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute and display quantization statistics'
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location='cpu')
    
    if 'model' in ckpt:
        sd = ckpt['model']
        container = 'model'
    else:
        sd = ckpt
        container = None
    
    # Prepare quantization kwargs
    kwargs = {}
    if args.method == 'log':
        kwargs['base'] = args.log_base
    elif args.method == 'ecq':
        kwargs['max_iter'] = args.max_iter
    
    # Quantize
    print(f"Applying {args.method.upper()} quantization to {len(sd)} tensors at {args.n_bits}-bit...")
    original_sd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sd.items()} if args.stats else None
    q_sd = quantize_state_dict(sd, args.n_bits, args.method, **kwargs)
    
    # Compute statistics if requested
    if args.stats and original_sd:
        print("\nComputing quantization statistics...")
        stats = compute_quantization_stats(original_sd, q_sd)
        print(f"Overall MSE: {stats['overall_mse']:.8f}")
        print(f"Overall PSNR: {stats['overall_psnr']:.2f} dB")
        
        print("\nPer-layer statistics (top 5 by MSE):")
        sorted_layers = sorted(stats['layer_stats'].items(), key=lambda x: x[1]['mse'], reverse=True)
        for name, layer_stats in sorted_layers[:5]:
            print(f"  {name}: MSE={layer_stats['mse']:.8f}, PSNR={layer_stats['psnr']:.2f}dB, Shape={layer_stats['shape']}")
    
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
        method_suffix = f"_{args.method}"
        if args.method == 'log':
            method_suffix += f"_base{args.log_base}"
        save_path = f"{base}{method_suffix}_q{args.n_bits}{ext}"
    
    torch.save(ckpt, save_path)
    print(f"Quantized checkpoint saved to {save_path}")


if __name__ == '__main__':
    main()
