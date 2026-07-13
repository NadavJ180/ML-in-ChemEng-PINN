"""
Baseline PINN Training Script

Executes the hybrid Adam + L-BFGS training loop for all generated TGV cases.
Includes memory-safe dynamic mini-batching designed for GPUs with <= 6GB VRAM.
"""

import argparse
import json
import torch
import gc
from pathlib import Path

# Import custom architecture and loss evaluator
from src.models.pinn import BaselinePINN
from src.models.loss import LossEvaluator

def parse_args():
    """Parses command-line arguments for VRAM-safe training configuration."""
    parser = argparse.ArgumentParser(description="Train Baseline PINNs for TGV")
    
    # VRAM-safe defaults for an RTX 3050 (6GB) using Float64
    parser.add_argument("--n_int", type=int, default=4000, help="Number of interior points per step")
    parser.add_argument("--n_ic", type=int, default=1000, help="Number of IC points per step")
    parser.add_argument("--n_bc", type=int, default=500, help="Number of points per boundary per step")
    
    # Optimizer settings
    parser.add_argument("--adam_epochs", type=int, default=2000, help="Epochs for Adam optimizer")
    parser.add_argument("--lbfgs_iters", type=int, default=1000, help="Max iterations for L-BFGS")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def sample_minibatch(case_data: dict, n_int: int, n_ic: int, n_bc: int, device: str):
    """
    Randomly slices a smaller mini-batch from the full loaded case dataset 
    and pushes it to the target device to prevent CUDA Out-Of-Memory errors.
    """
    # 1. Get the total number of points available in the loaded file
    total_int = case_data["interior"].shape[0]
    total_ic = case_data["ic"].shape[0]
    total_bc = case_data["bc"]["x_bounds"][0].shape[0]
    
    # 2. Generate random indices
    idx_int = torch.randperm(total_int)[:n_int]
    idx_ic = torch.randperm(total_ic)[:n_ic]
    idx_bc = torch.randperm(total_bc)[:n_bc]
    
    # 3. Slice the data and move strictly the sliced subset to the GPU
    batch = {
        "interior": case_data["interior"][idx_int].to(device),
        "ic": case_data["ic"][idx_ic].to(device),
        "bc": {
            "x_bounds": (
                case_data["bc"]["x_bounds"][0][idx_bc].to(device), 
                case_data["bc"]["x_bounds"][1][idx_bc].to(device)
            ),
            "y_bounds": (
                case_data["bc"]["y_bounds"][0][idx_bc].to(device), 
                case_data["bc"]["y_bounds"][1][idx_bc].to(device)
            )
        }
    }
    
    # Extract the analytical true initial conditions for the sliced coordinates
    ic_true = case_data.get("ic_true", torch.zeros_like(batch["ic"])) # Update based on how ic_true is stored
    # If ic_true is not pre-calculated in the .pt file, we will calculate it here
    
    return batch