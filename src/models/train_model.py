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
import torch.optim as optim
import time
import math
import sys
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the project root
project_root = Path(__file__).parent.parent.parent

# 2. Force Python to add the project root to its search path
sys.path.append(str(project_root))

# Import custom architecture and loss evaluator
from src.models.pinn import BaselinePINN
from src.models.loss import LossEvaluator
from src.models.scaling import ResidualScaler
from src.physics.taylor_green import compute_nu, compute_T # Needed for analytical equations
from src.physics.navier_stokes import compute_residuals # Needed for evaluation

def print_vram_instructions():
    """Prints a clear banner with instructions for handling GPU Out-Of-Memory errors."""
    print("="*60)
    print("🚀 INITIALIZING PINN TRAINING PIPELINE")
    print("="*60)
    print("VRAM SAFEGUARD ALERT:")
    print("This script defaults to 4000 interior points per step to safely fit")
    print("inside a 6GB VRAM limit (like the RTX 3050).")
    print("\nIf you encounter a 'CUDA Out of Memory' (OOM) error, do NOT edit the code.")
    print("Instead, abort the script and run it again using terminal flags to lower")
    print("the batch size. For example:")
    print("\n    python src/models/train_model.py --n_int 3000 --n_ic 800 --n_bc 400")
    print("\nThis will dynamically slice smaller batches from the data without needing")
    print("to regenerate the .pt files.")
    print("="*60 + "\n")

def parse_args():
    """
    Parses command-line arguments for VRAM-safe training configuration.
    
    Inputs:
        None (Reads directly from standard terminal sys.argv inputs).
        
    Outputs:
        args (argparse.Namespace): An object containing all the parsed hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train Baseline PINNs for TGV")
    
    # VRAM-safe defaults for an RTX 3050 (6GB) using Float64
    parser.add_argument("--n_int", type=int, default=4000, help="Number of interior points per step")
    parser.add_argument("--n_ic", type=int, default=1000, help="Number of IC points per step")
    parser.add_argument("--n_bc", type=int, default=500, help="Number of points per boundary per step")
    
    # Optimizer settings
    parser.add_argument("--adam_epochs", type=int, default=5000, help="Epochs for Adam optimizer")
    parser.add_argument("--lbfgs_iters", type=int, default=5000, help="Max iterations for L-BFGS")
    parser.add_argument("--lbfgs_min_iters", type=int, default=300, help="Minimum L-BFGS iterations before early-stop check begins")
    parser.add_argument("--lbfgs_check_window", type=int, default=100, help="Window size for plateau detection")
    parser.add_argument("--lbfgs_rel_tol", type=float, default=1e-3, help="Relative loss-improvement threshold to trigger early stop")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Commands for debugging and isolated testing (reverse order and isolated cases)
    parser.add_argument("--reverse_order", action="store_true",
                     help="TEMPORARY: run cases hardest-first instead of easiest-first (for diagnostic testing)")
    parser.add_argument("--case_id", type=str, default=None,
                     help="Run only this specific case_id (for isolated diagnostic testing)")
    
    return parser.parse_args()

def sample_minibatch(case_data: dict, n_int: int, n_ic: int, n_bc: int, device: str):
    """
    Randomly slices a smaller mini-batch from the full loaded Float64 case dataset 
    and pushes it to the target device.
    
    Inputs:
        case_data (dict): The full dataset dictionary loaded from the .pt file.
        n_int (int): The target number of interior collocation points to sample.
        n_ic (int): The target number of initial condition points to sample.
        n_bc (int): The target number of boundary points to sample per edge.
        device (str): The target hardware device ('cuda' or 'cpu').
        
    Outputs:
        batch (dict): A dictionary containing only the randomly sliced tensors 
                      pushed to the target device.
        ic_true (torch.Tensor): A tensor of shape (n_ic, 2) containing the exact 
                                analytical velocities (u, v) at t=0.
    """
    total_int = case_data["interior"].shape[0]
    total_ic = case_data["ic"].shape[0]
    total_bc = case_data["bc"]["x_bounds"][0].shape[0]
    
    idx_int = torch.randperm(total_int)[:n_int]
    idx_ic = torch.randperm(total_ic)[:n_ic]
    idx_bc = torch.randperm(total_bc)[:n_bc]
    
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
    
    ic_true = case_data["ic_true"][idx_ic].to(device)
    
    return batch, ic_true

def train_adam(model, case_data, case_meta, args):
    """
    Executes Phase 1 of PINN training using the Adam optimizer.
    Utilizes dynamic mini-batching to maintain strict VRAM limits.
    
    Inputs:
        model (nn.Module): The initialized BaselinePINN.
        case_data (dict): The loaded Float64 dataset.
        case_meta (dict): The metadata containing Re, U0, and k for this case.
        args (argparse.Namespace): The parsed command-line arguments.
        
    Outputs:
        model (nn.Module): The trained model after Adam optimization.
        criterion (LossEvaluator): The initialized loss evaluator.
        history (dict): A dictionary containing the loss history for each component.
    """
    print("\n--- Phase 1: Adam Optimization ---")
    
    criterion = LossEvaluator(
        Re=case_meta["Re"], 
        U0=case_meta["U0"], 
        k=case_meta["k"]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    
    # Dictionary to track loss history
    history = {'Total': [], 'L_NS': [], 'L_div': [], 'L_IC': [], 'L_BC': []}
    
    model.train()
    
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()
        
        # 1. Slice a memory-safe mini-batch and push to GPU
        batch, ic_true = sample_minibatch(
            case_data=case_data, 
            n_int=args.n_int, 
            n_ic=args.n_ic, 
            n_bc=args.n_bc, 
            device=args.device
        )
        
        # 2. Forward pass and multi-component loss evaluation
        total_loss, metrics = criterion(model, batch, ic_true)
        
        # 3. Backward pass and weight update
        total_loss.backward()
        optimizer.step()
        
        # Log metrics
        history['Total'].append(total_loss.item())
        history['L_NS'].append(metrics['L_NS'])
        history['L_div'].append(metrics['L_div'])
        history['L_IC'].append(metrics['L_IC'])
        history['L_BC'].append(metrics['L_BC'])
        
        # 4. Progress logging
        if epoch % 100 == 0 or epoch == args.adam_epochs - 1:
            print(f"Epoch {epoch:04d}/{args.adam_epochs} | Total Loss: {total_loss.item():.4e} | "
                  f"L_NS(s): {metrics['L_NS']:.2e} | L_NS(raw): {metrics['L_NS_raw']:.2e} | L_div: {metrics['L_div']:.2e} | "
                  f"L_IC: {metrics['L_IC']:.2e} | L_BC: {metrics['L_BC']:.2e}")
                  
    return model, criterion, history

class LBFGSConverged(Exception):
    """Signals that L-BFGS has plateaued and can stop early."""
    pass

def train_lbfgs(model, criterion, case_data, case_meta, args, device, eval_interval=500):
    """
    Executes Phase 2 of PINN training using the L-BFGS optimizer.
    Uses a static mini-batch to ensure convergence of the line-search algorithm
    while maintaining the VRAM limits.
    
    Inputs:
        model (nn.Module): The Adam-trained BaselinePINN.
        criterion (LossEvaluator): The initialized loss evaluator from Phase 1.
        case_data (dict): The loaded Float64 dataset.
        args (argparse.Namespace): The parsed command-line arguments.
        
    Outputs:
        model (nn.Module): The fully trained model.
        history (dict): A dictionary containing the loss history for each component during L-BFGS.
    """
    print("\n--- Phase 2: L-BFGS Fine-Tuning ---")
    
    # 1. Slice ONE static memory-safe batch for the entire L-BFGS phase
    batch, ic_true = sample_minibatch(
        case_data=case_data, 
        n_int=args.n_int, 
        n_ic=args.n_ic, 
        n_bc=args.n_bc, 
        device=args.device
    )
    
    eval_grid = case_data["eval_grid"]   # needed for periodic checks


    # 2. Initialize L-BFGS with strong Wolfe line search for stability
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=args.lbfgs_iters,
        history_size=50,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe" 
    )
    
    # Dictionary to track loss history
    history = {'Total': [], 'L_NS': [], 'L_div': [], 'L_IC': [], 'L_BC': []}
    rel_l2_history = []   # NEW: track (iteration, RelL2) pairs
    iteration = 0
    
    # 3. Define the closure function required by L-BFGS
    def closure():
        nonlocal iteration
        optimizer.zero_grad()
        
        # Forward and backward pass
        total_loss, metrics = criterion(model, batch, ic_true)
        total_loss.backward()
        
        # Log metrics inside the closure since L-BFGS controls the step calls
        history['Total'].append(total_loss.item())
        history['L_NS'].append(metrics['L_NS'])
        history['L_div'].append(metrics['L_div'])
        history['L_IC'].append(metrics['L_IC'])
        history['L_BC'].append(metrics['L_BC'])
        
        # Logging (L-BFGS handles its own loop, so we log inside the closure)
        if iteration % 100 == 0:
            print(f"L-BFGS Iter {iteration:04d} | Total Loss: {total_loss.item():.4e} | "
                  f"L_NS(s): {metrics['L_NS']:.2e} | L_NS(raw): {metrics['L_NS_raw']:.2e} | L_div: {metrics['L_div']:.2e} | "
                  f"L_IC: {metrics['L_IC']:.2e} | L_BC: {metrics['L_BC']:.2e}")
        
        # --- NEW: periodic held-out RelL2 check ---
        if iteration > 0 and iteration % eval_interval == 0:
            _, rel_l2, mse_rc, mse_rm = evaluate_model(
                model, case_meta, eval_grid, device, chunk_size=5000, verbose=False
            )
            rel_l2_history.append((iteration, rel_l2))
            print(f"    [Periodic Eval] Iter {iteration}: RelL2={rel_l2:.4f} | MSE_Rc={mse_rc:.2e} | MSE_Rm={mse_rm:.2e}")
            model.train()   # evaluate_model sets model.eval() internally; switch back

        # --- Early stopping check, windowed to avoid noise ---
        window = args.lbfgs_check_window
        if iteration >= args.lbfgs_min_iters and iteration % window == 0 and len(history['Total']) >= 2 * window:
            recent = sum(history['Total'][-window:]) / window
            prior = sum(history['Total'][-2*window:-window]) / window
            rel_improvement = abs(prior - recent) / (abs(prior) + 1e-12)
            if rel_improvement < args.lbfgs_rel_tol:
                iteration += 1
                raise LBFGSConverged(iteration)

        iteration += 1
        return total_loss
        
    # 4. Execute the optimization step
    model.train()
    try:
        optimizer.step(closure)
    except LBFGSConverged as e:
        print(f"L-BFGS converged early at iteration {e.args[0]} (plateau detected).")
    
    return model, history, rel_l2_history

def evaluate_model(model, case_meta, eval_grid, device, chunk_size=5000, verbose=True):
    """
    Evaluates the trained model against the WP3 Acceptance Criteria using the
    81,920 point evaluation grid. Processes in chunks to prevent VRAM overflow.
    """
    if verbose:
        print("\n--- Phase 3: Acceptance Criteria Evaluation ---")
    model.eval()
    
    Re = case_meta["Re"]
    U0 = case_meta["U0"]
    k = case_meta["k"]
    nu = compute_nu(U0, Re, k)
    nu = compute_nu(U0, Re, k)
    
    # Initialize centralized scaler for consistent evaluation
    scaler = ResidualScaler(U0, k)

    total_points = eval_grid.shape[0]
    
    # Accumulators for the metrics
    sum_mse_ru, sum_mse_rv, sum_mse_rc = 0.0, 0.0, 0.0
    sum_l2_num, sum_l2_den = 0.0, 0.0
    
    # Process the grid in VRAM-safe chunks
    for i in range(0, total_points, chunk_size):
        chunk = eval_grid[i:i+chunk_size].to(dtype=torch.float64, device=device)
        
        # Track gradients for residual calculation
        x = chunk[:, 0:1].requires_grad_(True)
        y = chunk[:, 1:2].requires_grad_(True)
        t = chunk[:, 2:3].requires_grad_(True)
        coords = torch.cat([x, y, t], dim=1)
        
        # Predictions
        preds = model(coords)
        u_pred, v_pred, p_pred = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        
        # 1. Compute and scale residuals for consistent evaluation
        R_u_raw, R_v_raw, R_c_raw = compute_residuals(u_pred, v_pred, p_pred, x, y, t, nu)
        R_u, R_v, R_c = scaler.scale_residuals(R_u_raw, R_v_raw, R_c_raw)

        sum_mse_ru += torch.sum(R_u**2).item()
        sum_mse_rv += torch.sum(R_v**2).item()
        sum_mse_rc += torch.sum(R_c**2).item()
        
        # 2. Compute exact analytical velocities for RelL2
        # TGV exact decay term: exp(-2 * nu * k^2 * t)
        decay = torch.exp(-2.0 * nu * (k**2) * t)
        u_true = U0 * torch.sin(k * x) * torch.cos(k * y) * decay
        v_true = -U0 * torch.cos(k * x) * torch.sin(k * y) * decay
        
        # Accumulate L2 error components
        vel_error_sq = (u_pred - u_true)**2 + (v_pred - v_true)**2
        vel_true_sq = u_true**2 + v_true**2
        
        sum_l2_num += torch.sum(vel_error_sq).item()
        sum_l2_den += torch.sum(vel_true_sq).item()

    # Calculate final grid-wide metrics
    mse_ru = sum_mse_ru / total_points
    mse_rv = sum_mse_rv / total_points
    mse_rc = sum_mse_rc / total_points
    
    rel_l2 = math.sqrt(sum_l2_num) / math.sqrt(sum_l2_den) if sum_l2_den > 0 else float('inf')
    
    # Acceptance Evaluation
    crit_1_pass = rel_l2 < 0.10
    crit_2_pass = mse_rc < 1e-4
    crit_3_pass = (mse_ru + mse_rv) < 1e-3
    
    passed_all = crit_1_pass and crit_2_pass and crit_3_pass
    
    if verbose:
        print(f"RelL2(u,v) = {rel_l2:.4f} \t[Criteria < 0.10]: {'✅' if crit_1_pass else '❌'}")
        print(f"MSE(R_c)   = {mse_rc:.4e} \t[Criteria < 1e-4]: {'✅' if crit_2_pass else '❌'}")
        print(f"MSE(R_m)   = {(mse_ru+mse_rv):.4e} \t[Criteria < 1e-3]: {'✅' if crit_3_pass else '❌'}")
    
    return passed_all, rel_l2, mse_rc, (mse_ru + mse_rv)

def plot_case_history(case_id, adam_hist, lbfgs_hist, save_dir):
    """Generates and saves a logarithmic loss curve plot for a single case."""
    keys = ['Total', 'L_NS', 'L_div', 'L_IC', 'L_BC']
    
    # Concatenate the Adam and LBFGS histories
    combined_hist = {k: adam_hist[k] + lbfgs_hist[k] for k in keys}
    
    adam_len = len(adam_hist['Total'])
    total_len = len(combined_hist['Total'])
    x_total = np.arange(total_len)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each line
    for k in keys:
        linestyle = '--' if k == 'Total' else '-'
        linewidth = 2 if k == 'Total' else 1.5
        plt.plot(x_total, combined_hist[k], label=k, linestyle=linestyle, linewidth=linewidth)
        
    # Apply background colors to distinguish the phases
    plt.axvspan(0, adam_len, color='blue', alpha=0.05, label='Adam Phase')
    plt.axvspan(adam_len, total_len, color='orange', alpha=0.05, label='L-BFGS Phase')
    
    plt.yscale('log')
    plt.xlabel('Optimization Steps (Adam) / L-BFGS Closure Evaluations')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'Loss History for {case_id}')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', ls='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{case_id}_loss_history.png")
    plt.close()

def sort_cases_by_difficulty(cases):
    """
    Sorts a list of case dictionaries from easiest to hardest physics.
    Primary sort: Wave number (k) - lower is smoother and easier to resolve.
    Secondary sort: Reynolds number (Re) - lower is more viscous and stable.
    """
    # Sorts first by 'k', and if 'k's are tied, sorts by 'Re'
    return sorted(cases, key=lambda c: (c['k'], c['Re']))

def main():
    print_vram_instructions()
    args = parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / "data" / "cases_metadata.json"
    tensors_dir = project_root / "data" / "tensors"
    
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError("cases_metadata.json not found.")
        
    with open(metadata_path, "r") as f:
        dataset = json.load(f)
        
    # Combine all datasets into one execution list
    all_cases = dataset["train"] + dataset["validation"] + dataset["test"]
    
    # SORT BY DIFFICULTY: Easiest (low k, low Re) to Hardest (high k, high Re)
    all_cases = sort_cases_by_difficulty(all_cases)

    # TEMPORARY: for testing hardest-case behavior before full sweep — remove/disable after
    if args.reverse_order:
        all_cases = list(reversed(all_cases))
        print("⚠️  Running in REVERSED order (hardest case first) for diagnostic testing.\n")
    if args.case_id:
        all_cases = [c for c in all_cases if c["case_id"] == args.case_id]

    total_cases = len(all_cases)
    global_start_time = time.time()
    
    print(f"Starting execution for {total_cases} TGV cases on {args.device.upper()}...")
    
    for idx, case in enumerate(all_cases):
        case_id = case["case_id"]
        case_start = time.time()
        
        print(f"\n{'='*50}\n[{idx+1}/{total_cases}] Training Case: {case_id}\n"
              f"Params: Re={case['Re']}, U0={case['U0']}, k={case['k']}\n{'='*50}")
        
        # Load the Float64 tensor dictionary
        pt_path = tensors_dir / f"{case_id}.pt"
        case_data = torch.load(pt_path, map_location="cpu")
        
        # Initialize the fresh model for this case
        model = BaselinePINN().to(args.device)
        
        # Train
        model, criterion, adam_hist = train_adam(model, case_data, case, args)
        model, lbfgs_hist, rel_l2_hist = train_lbfgs(model, criterion, case_data, case, args, args.device)
        
        plot_case_history(case_id, adam_hist, lbfgs_hist, plots_dir)
        print(f"📈 Saved loss history plot to {plots_dir.name}/{case_id}_loss_history.png")
        
        if rel_l2_hist:
            iters, rel_l2_vals = zip(*rel_l2_hist)
            plt.figure(figsize=(8, 5))
            plt.plot(iters, rel_l2_vals, marker='o')
            plt.axhline(0.10, color='red', linestyle='--', label='Acceptance threshold')
            plt.xlabel('L-BFGS Iteration')
            plt.ylabel('RelL2(u,v) on eval grid')
            plt.title(f'Held-out RelL2 during L-BFGS — {case_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f"{case_id}_rel_l2_tracking.png")
            plt.close()

        # Evaluate
        eval_grid = case_data["eval_grid"]
        passed, rel_l2, mse_rc, mse_rm = evaluate_model(model, case, eval_grid, args.device)
        
       # Apply Section 12.4 Risk Mitigation Rule
        log_file = project_root / "training_summary.log"
        with open(log_file, "a") as f:
            if passed:
                f.write(f"PASSED: {case_id} | RelL2: {rel_l2:.4f}\n")
                print(f"\n✅ Case {case_id} PASSED all usability criteria.")
                # Save model
                try:
                    torch.save(model.state_dict(), models_dir / f"{case_id}_best.pth")
                except Exception as e:
                    print(f"⚠️ Failed to save model for {case_id}: {e}")
            else:
                f.write(f"FAILED: {case_id} | RelL2: {rel_l2:.4f}\n")
                print(f"\n⚠️ Case {case_id} FAILED criteria. Tagging for Risk Mitigation.")
            
        case_elapsed = time.time() - case_start
        print(f"⏳ Case {case_id} execution time: {case_elapsed/60:.2f} minutes")
        
        # Aggressive memory clearing to protect the 6GB VRAM limit between cases
        del model, case_data, eval_grid, criterion
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    global_elapsed = time.time() - global_start_time
    hours, rem = divmod(global_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n{'='*50}\n🎉 ALL CASES COMPLETE\n"
          f"Total Execution Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n{'='*50}")

if __name__ == "__main__":
    main()
