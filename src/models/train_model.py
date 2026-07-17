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
from src.physics.taylor_green import compute_nu, compute_T, generate_tgv # Needed for analytical equations
from src.physics.navier_stokes import compute_residuals # Needed for evaluation
from src.hallucinations.perturbations import apply_perturbation, PERTURBATION_NAMES, EPSILON_VALUES

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

    # Flow field distortion demo (shows perturbed vs. clean fields side by side)
    parser.add_argument("--demo_case_id", type=str, default=None,
                     help="Generate the flow field distortion demo for this specific case_id. "
                          "Defaults to the first case successfully trained in this run.")
    parser.add_argument("--skip_distortion_demo", action="store_true",
                     help="Disable generating the flow field distortion demo entirely.")

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
    history = {'Total': [], 'L_NS': [], 'L_div': [], 'L_IC': [], 'L_BC': [], 'L_p': []}
    
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
        history['L_p'].append(metrics['L_p'])   
        
        # 4. Progress logging
        if epoch % 100 == 0 or epoch == args.adam_epochs - 1:
            print(f"Epoch {epoch:04d}/{args.adam_epochs} | Total Loss: {total_loss.item():.4e} | "
                  f"L_NS(s): {metrics['L_NS']:.2e} | L_NS(raw): {metrics['L_NS_raw']:.2e} | L_div: {metrics['L_div']:.2e} | "
                  f"L_IC: {metrics['L_IC']:.2e} | L_BC: {metrics['L_BC']:.2e} | L_p: {metrics['L_p']:.2e}")
                  
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
    history = {'Total': [], 'L_NS': [], 'L_div': [], 'L_IC': [], 'L_BC': [], 'L_p': []}
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
        history['L_p'].append(metrics['L_p'])
        
        # Logging (L-BFGS handles its own loop, so we log inside the closure)
        if iteration % 100 == 0:
            print(f"L-BFGS Iter {iteration:04d} | Total Loss: {total_loss.item():.4e} | "
                  f"L_NS(s): {metrics['L_NS']:.2e} | L_NS(raw): {metrics['L_NS_raw']:.2e} | L_div: {metrics['L_div']:.2e} | "
                  f"L_IC: {metrics['L_IC']:.2e} | L_BC: {metrics['L_BC']:.2e} | L_p: {metrics['L_p']:.2e}")
        
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
    # FIX: phi_x/phi_y were never read from case_meta here -- this function was
    # silently comparing every model against phi_x=0, phi_y=0 regardless of the
    # case's actual target phase (the same bug found in verify_model.py and
    # generate_datasets.py). Since this is the function that decides
    # PASSED/FAILED for the WP3 acceptance criteria, every prior pass/fail
    # decision and every eval_interval RelL2 reading logged during L-BFGS was
    # made against the wrong ground truth.
    phi_x = case_meta["phi_x"]
    phi_y = case_meta["phi_y"]
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
        # FIX: now uses this case's actual target phase (phi_x, phi_y) via
        # generate_tgv(), instead of a hardcoded phi_x=0, phi_y=0 formula.
        u_true, v_true, _ = generate_tgv(x, y, t, U0, k, phi_x, phi_y, nu)
        
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

def plot_case_history(case_id, adam_hist, lbfgs_hist, case_dir):
    """Generates and saves a logarithmic loss curve plot for a single case, inside its own per-case folder."""
    keys = ['Total', 'L_NS', 'L_div', 'L_IC', 'L_BC', 'L_p']
    
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
    plt.savefig(case_dir / "loss_history.png")
    plt.close()

def update_loss_summary(summary_row: dict, summary_dir: Path):
    """
    Appends (or updates) one case's final-loss summary row into a cumulative
    cross-case tracking table, saved as both CSV and JSON. Since main() can be
    interrupted and resumed (existing models are skipped via the checkpoint
    logic), this reads back any existing summary first and replaces the row
    for this case_id if it already exists, rather than blindly appending
    duplicates across multiple runs.

    Inputs:
        summary_row (dict): One case's final metrics -- expected keys include
                             "case_id", "split", final loss components, "rel_l2",
                             "mse_rc", "mse_rm", "passed", and "training_minutes".
        summary_dir (Path): The general tracking folder (plots/loss_history/_summary).

    Outputs:
        None. Writes loss_summary.csv and loss_summary.json to summary_dir.
    """
    json_path = summary_dir / "loss_summary.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            rows = json.load(f)
    else:
        rows = []

    rows = [r for r in rows if r["case_id"] != summary_row["case_id"]]
    rows.append(summary_row)
    rows.sort(key=lambda r: r["case_id"])

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    csv_path = summary_dir / "loss_summary.csv"
    if rows:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def plot_loss_overlay(loss_history_root: Path, summary_dir: Path):
    """
    Rebuilds a single figure overlaying every case's Total-loss curve (read
    back from each case's own {case_id}/loss_history.json), so the whole
    30-case training sweep can be compared at a glance. Regenerated after
    every case completes, so it stays current even if the run is interrupted.

    Inputs:
        loss_history_root (Path): Root folder containing one subfolder per
                                   case_id, each with a loss_history.json.
        summary_dir (Path): The general tracking folder to save the overlay into.

    Outputs:
        None. Writes all_cases_loss_overlay.png to summary_dir. Silently does
        nothing if no per-case loss_history.json files exist yet.
    """
    case_dirs = sorted(d for d in loss_history_root.iterdir() if d.is_dir() and d.name != "_summary")
    if not case_dirs:
        return

    plt.figure(figsize=(11, 7))
    cmap = plt.get_cmap("viridis")

    for i, case_dir in enumerate(case_dirs):
        history_path = case_dir / "loss_history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r") as f:
            hist = json.load(f)
        color = cmap(i / max(len(case_dirs) - 1, 1))
        plt.plot(hist["Total"], label=case_dir.name, color=color, linewidth=1.2, alpha=0.85)

    plt.yscale('log')
    plt.xlabel('Optimization Steps (Adam) / L-BFGS Closure Evaluations')
    plt.ylabel('Total Loss (Log Scale)')
    plt.title(f'Loss History Overlay -- {len(case_dirs)} Case(s)')
    # Case count can get large across the full 30-case sweep; keep the legend
    # usable by capping how many entries it shows directly.
    if len(case_dirs) <= 15:
        plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig(summary_dir / "all_cases_loss_overlay.png", dpi=150)
    plt.close()


def generate_distortion_demo(model, case_id: str, case_meta: dict, device: str, save_dir: Path, res: int = 64, time_frac: float = 0.5):
    """
    Generates a single side-by-side figure demonstrating the flow field
    becoming visibly distorted under the Section 7 hallucination
    perturbations, using the exact same perturbation functions as the
    dedicated hallucination-generation pipeline (src/hallucinations/perturbations.py)
    so there is one source of truth for the perturbation math.

    Produces two rows in one figure:
      - Top row: ONE perturbation type ("momentum") applied at increasing
        epsilon (clean, then all of EPSILON_VALUES), showing the
        imperceptible-to-obvious distortion progression.
      - Bottom row: all 5 perturbation types applied at a single, clearly
        visible epsilon (0.05), so the different distortion signatures can
        be compared side by side.

    No gradients are needed here (this is a value-only visual demo, not a
    residual check), so the whole thing runs in inference mode on a small grid.

    Inputs:
        model (nn.Module): The just-trained (or loaded) BaselinePINN for this case.
        case_id (str): The case identifier (used in the plot title and filename).
        case_meta (dict): This case's metadata; requires "U0", "k", "Re".
        device (str): Target hardware device ('cuda' or 'cpu').
        save_dir (Path): Directory to save the figure into.
        res (int): Grid resolution (res x res points). Defaults to 64, since
                    this is a qualitative demo, not a quantitative check.
        time_frac (float): Fraction of the case's T at which the snapshot is taken.

    Outputs:
        None. Saves "{case_id}_distortion_demo.png" to save_dir.
    """
    U0, k, Re = case_meta["U0"], case_meta["k"], case_meta["Re"]
    T = compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T}
    t_val = time_frac * T

    x_lin = torch.linspace(0, 2 * math.pi, res, dtype=torch.float64)
    y_lin = torch.linspace(0, 2 * math.pi, res, dtype=torch.float64)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x = X.reshape(-1, 1).to(device)
    y = Y.reshape(-1, 1).to(device)
    t = torch.full_like(x, t_val)

    coords = {"x": x, "y": y, "t": t}

    model.eval()
    with torch.no_grad():
        preds = model(torch.cat([x, y, t], dim=1))
        clean = {"u": preds[:, 0:1], "v": preds[:, 1:2], "p": preds[:, 2:3]}
        mag_clean = torch.sqrt(clean["u"] ** 2 + clean["v"] ** 2).reshape(res, res).cpu().numpy()

        # --- Row 1: one perturbation type across increasing epsilon ---
        progression_epsilons = [0.0] + list(EPSILON_VALUES)
        progression_fields = [mag_clean]
        for epsilon in EPSILON_VALUES:
            pert = apply_perturbation("momentum", clean, coords, params, epsilon, model=None)
            mag = torch.sqrt(pert["u"] ** 2 + pert["v"] ** 2).reshape(res, res).cpu().numpy()
            progression_fields.append(mag)

        # --- Row 2: all 5 perturbation types at a fixed, clearly visible epsilon ---
        demo_epsilon = 0.05
        type_fields = [("Clean", mag_clean)]
        for perturbation_name in PERTURBATION_NAMES:
            model_arg = model if perturbation_name == "temporal_mismatch" else None
            pert = apply_perturbation(perturbation_name, clean, coords, params, demo_epsilon, model=model_arg, no_grad=True)
            mag = torch.sqrt(pert["u"] ** 2 + pert["v"] ** 2).reshape(res, res).cpu().numpy()
            type_fields.append((perturbation_name, mag))

    vmin, vmax = mag_clean.min(), mag_clean.max()
    n_cols = max(len(progression_fields), len(type_fields))
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.5))
    fig.suptitle(f"Flow Field Distortion Demo: {case_id} (Re={Re:.1f}, U0={U0:.2f}, k={k})", fontsize=14)

    for col in range(n_cols):
        ax = axes[0, col]
        if col < len(progression_fields):
            im = ax.imshow(progression_fields[col].T, origin="lower", extent=[0, 2 * math.pi, 0, 2 * math.pi],
                            cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_title(f"momentum\nε={progression_epsilons[col]}", fontsize=9)
        else:
            ax.axis("off")

    for col in range(n_cols):
        ax = axes[1, col]
        if col < len(type_fields):
            name, field = type_fields[col]
            im = ax.imshow(field.T, origin="lower", extent=[0, 2 * math.pi, 0, 2 * math.pi],
                            cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_title(name if name == "Clean" else f"{name}\nε={demo_epsilon}", fontsize=9)
        else:
            ax.axis("off")

    fig.text(0.02, 0.72, "Increasing ε\n(one type)", ha="left", va="center", fontsize=9, fontweight="bold")
    fig.text(0.02, 0.28, "Different types\n(fixed ε)", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_path = save_dir / f"{case_id}_distortion_demo.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"🌀 Saved flow field distortion demo to {save_path.relative_to(save_path.parent.parent.parent)}")


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

    # NEW: per-case loss-history folders + a general cross-case tracking folder,
    # instead of dumping every case's plots/json flat into plots_dir.
    loss_history_root = plots_dir / "loss_history"
    loss_history_root.mkdir(exist_ok=True)
    loss_summary_dir = loss_history_root / "_summary"
    loss_summary_dir.mkdir(exist_ok=True)

    # NEW: folder for the flow field distortion demo(s)
    distortion_demo_dir = plots_dir / "perturbation_demo"
    distortion_demo_dir.mkdir(exist_ok=True)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError("cases_metadata.json not found.")
        
    with open(metadata_path, "r") as f:
        dataset = json.load(f)
        
    # Combine all datasets into one execution list, tagging each case with its
    # split so it can be carried through to the loss summary table.
    all_cases = (
        [{**c, "split": "train"} for c in dataset["train"]]
        + [{**c, "split": "validation"} for c in dataset["validation"]]
        + [{**c, "split": "test"} for c in dataset["test"]]
    )
    
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
    demo_generated = False
    
    print(f"Starting execution for {total_cases} TGV cases on {args.device.upper()}...")
    
    for idx, case in enumerate(all_cases):
        case_id = case["case_id"]
        
        # --- NEW CHECKPOINT LOGIC ---
        # Check if this case has already been successfully trained and saved
        expected_model_path = project_root / "models" / f"{case_id}_best.pth"
        if expected_model_path.exists():
            print(f"\n{'='*50}\n[{idx+1}/{total_cases}] Skipping {case_id}: Model already exists.\n{'='*50}")
            continue
        # ----------------------------

        case_start = time.time()
        
        print(f"\n{'='*50}\n[{idx+1}/{total_cases}] Training Case: {case_id}\n"
              f"Params: Re={case['Re']}, U0={case['U0']}, k={case['k']}\n{'='*50}")
        
        # Load the Float64 tensor dictionary
        pt_path = tensors_dir / f"{case_id}.pt"
        case_data = torch.load(pt_path, map_location="cpu")
        
        # Initialize the fresh model for this case
        model = BaselinePINN(k=case["k"]).to(args.device)
        
        # Train
        model, criterion, adam_hist = train_adam(model, case_data, case, args)
        model, lbfgs_hist, rel_l2_hist = train_lbfgs(model, criterion, case_data, case, args, args.device)

        # NEW: everything for this case now lives in its own subfolder
        case_dir = loss_history_root / case_id
        case_dir.mkdir(exist_ok=True)

        plot_case_history(case_id, adam_hist, lbfgs_hist, case_dir)
        print(f"📈 Saved loss history plot to {case_dir.relative_to(project_root)}/loss_history.png")
        
        # Export the history log as .json for future analysis
        combined_hist = {k: adam_hist[k] + lbfgs_hist[k] for k in adam_hist.keys()}
        history_file = case_dir / "loss_history.json"
        with open(history_file, "w") as f:
            json.dump(combined_hist, f)

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
            plt.savefig(case_dir / "rel_l2_tracking.png")
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

        # NEW: update the general cross-case tracking table + overlay plot
        summary_row = {
            "case_id": case_id,
            "split": case["split"],
            "final_total_loss": combined_hist["Total"][-1],
            "final_L_NS": combined_hist["L_NS"][-1],
            "final_L_div": combined_hist["L_div"][-1],
            "final_L_IC": combined_hist["L_IC"][-1],
            "final_L_BC": combined_hist["L_BC"][-1],
            "final_L_p": combined_hist["L_p"][-1],
            "rel_l2": rel_l2,
            "mse_rc": mse_rc,
            "mse_rm": mse_rm,
            "passed": passed,
            "training_minutes": case_elapsed / 60,
        }
        update_loss_summary(summary_row, loss_summary_dir)
        plot_loss_overlay(loss_history_root, loss_summary_dir)

        # NEW: generate the flow field distortion demo for at least one case
        # (either the user-requested --demo_case_id, or the first case that
        # passes in this run if no specific case was requested).
        should_generate_demo = (not args.skip_distortion_demo) and passed and (
            args.demo_case_id == case_id or (args.demo_case_id is None and not demo_generated)
        )
        if should_generate_demo:
            generate_distortion_demo(model, case_id, case, args.device, distortion_demo_dir)
            demo_generated = True

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