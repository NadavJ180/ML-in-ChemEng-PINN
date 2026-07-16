"""
Hallucination Verification Script (Section 7 sanity checks)

This script answers the two DIFFERENT questions the hallucination benchmark
depends on, and deliberately checks them over DIFFERENT epsilon ranges:

  1. "Do small perturbations actually look clean?"
     -> Visual check, restricted to epsilon in {0.01, 0.02}. This is the
        specific claim under test: that the deceptive regime is deceptive.
        Checking large epsilon here wouldn't add information, since large
        epsilon is *expected* to look wrong.

  2. "Does each perturbation type actually activate its intended physical
     violation, and does the violation scale sensibly with epsilon?"
     -> Quantitative check (PDE residuals, periodic-boundary mismatch,
        deviation from the analytical solution), run across the FULL
        epsilon sweep {0.005, 0.01, 0.02, 0.05, 0.1} and all 5 perturbation
        types. This is a different claim from (1) and needs the full range
        to be credible -- a monotonic, non-negligible violation curve is
        much stronger evidence than two isolated points.

Outputs (per case, saved under plots/hallucination_verification/{case_id}/):
  - contour_eps_0.01_0.02.png   : Required deliverable -- clean vs. perturbed
                                   side-by-side contours at eps in {0.01, 0.02},
                                   plus a difference panel, for all 5 perturbations.
  - contour_full_sweep.png      : Extra robustness figure -- difference-from-clean
                                   maps across all 5 epsilon values, so the
                                   imperceptible -> obvious transition is visible.
  - residual_curves.png         : Extra robustness figure -- each perturbation's
                                   dominant violation metric vs. epsilon (log scale).
  - residual_summary.csv/.json  : Full quantitative table: every
                                   (perturbation, epsilon) combination x every
                                   violation metric, for direct comparison
                                   against the project's Table 1 mapping.

Usage:
    python src/hallucinations/verify_hallucinations.py
    python src/hallucinations/verify_hallucinations.py --case_id case_00
    python src/hallucinations/verify_hallucinations.py --all_cases
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the project root (mirrors the convention used elsewhere in the repo)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.models.scaling import ResidualScaler
from src.physics.navier_stokes import compute_residuals
from src.physics.taylor_green import compute_nu, compute_T, generate_tgv
from src.hallucinations.perturbations import (
    apply_perturbation,
    EPSILON_VALUES,
    PERTURBATION_NAMES,
)
from src.hallucinations.generate_hallucinations import load_case_metadata

# The two epsilon values the visual imperceptibility check is restricted to,
# per the task specification.
VISUAL_CHECK_EPSILONS = [0.01, 0.02]


def parse_args():
    """
    Parses command-line arguments controlling which case(s) to verify, the
    resolution and time-slice of the differentiable verification grid, and
    where outputs are written.

    Inputs:
        None (reads directly from sys.argv).

    Outputs:
        args (argparse.Namespace): Parsed arguments with fields
            case_id (str | None), all_cases (bool), device (str),
            res (int), time_frac (float), n_bc (int), output_dir (str).
    """
    parser = argparse.ArgumentParser(description="Verify the physical hallucination dataset (Section 7).")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Case to generate full plots + table for. Defaults to the first case with a trained model.")
    parser.add_argument("--all_cases", action="store_true",
                        help="Also compute (but not plot) the quantitative residual table for every trained case, "
                             "to confirm violation patterns hold across the dataset, not just one case.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--res", type=int, default=96,
                        help="Resolution of the square differentiable verification grid (res x res points).")
    parser.add_argument("--time_frac", type=float, default=0.5,
                        help="Fraction of the case's final time T at which the verification snapshot is taken.")
    parser.add_argument("--n_bc", type=int, default=200,
                        help="Number of paired points sampled along the periodic x-boundary for the BC mismatch check.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory. Defaults to <project_root>/plots/hallucination_verification.")
    return parser.parse_args()


def load_model(case_id: str, k: float, device: str):
    """
    Loads a trained BaselinePINN checkpoint for a single case.

    Inputs:
        case_id (str): The case identifier (e.g. "case_00").
        k (float): The case's wavenumber, required by the BaselinePINN constructor.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        model (nn.Module): The loaded, float64, eval-mode model on `device`.
    """
    model_path = project_root / "models" / f"{case_id}_best.pth"
    model = BaselinePINN(k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.to(torch.float64)
    model.eval()
    return model


def build_interior_grid(T: float, res: int, time_frac: float, device: str):
    """
    Builds a square, gradient-tracked (x, y) grid at a single fixed time
    slice, suitable for both contour plotting and PDE residual computation
    (since residuals require x, y, t to carry requires_grad=True).

    Inputs:
        T (float): The case's final simulation time.
        res (int): Grid resolution (res x res points).
        time_frac (float): Fraction of T at which the snapshot is taken.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        x, y, t (torch.Tensor): Flattened leaf tensors of shape (res*res, 1),
                                 each with requires_grad=True, dtype float64.
    """
    t_val = time_frac * T
    x_lin = torch.linspace(0, 2 * torch.pi, res, dtype=torch.float64)
    y_lin = torch.linspace(0, 2 * torch.pi, res, dtype=torch.float64)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")

    x = X.reshape(-1, 1).to(device).clone().requires_grad_(True)
    y = Y.reshape(-1, 1).to(device).clone().requires_grad_(True)
    t = torch.full_like(x, t_val).clone().requires_grad_(True)

    return x, y, t


def build_boundary_pair_grid(T: float, n_points: int, time_frac: float, device: str):
    """
    Builds paired coordinate sets along the periodic x-boundaries (x=0 and
    x=2*pi), sharing the same y and t values, so left/right predictions can
    be compared directly to check periodic boundary consistency.

    Inputs:
        T (float): The case's final simulation time.
        n_points (int): Number of paired boundary points to sample.
        time_frac (float): Fraction of T at which the snapshot is taken.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        x_left, x_right, y_b, t_b (torch.Tensor): Shape (n_points, 1) each,
                                                    dtype float64. x_left is
                                                    all zeros, x_right is all
                                                    2*pi; y_b and t_b are
                                                    shared between the two sides.
    """
    t_val = time_frac * T
    y_lin = torch.linspace(0, 2 * torch.pi, n_points, dtype=torch.float64).reshape(-1, 1)

    x_left = torch.zeros_like(y_lin).to(device)
    x_right = torch.full_like(y_lin, 2 * torch.pi).to(device)
    y_b = y_lin.to(device)
    t_b = torch.full_like(y_lin, t_val).to(device)

    return x_left, x_right, y_b, t_b


def compute_clean_and_perturbed(model, x, y, t, params, perturbation_name: str, epsilon: float, track_gradients: bool = True):
    """
    Runs the model once to obtain the clean (u, v, p) prediction, then
    applies the requested Section 7 perturbation on top of it, optionally
    preserving the autograd graph so PDE residuals can be computed downstream.

    Inputs:
        model (nn.Module): The trained BaselinePINN.
        x, y, t (torch.Tensor): Coordinate tensors, shape (N, 1). Must have
                                 requires_grad=True if track_gradients=True.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES.
        epsilon (float): Perturbation strength.
        track_gradients (bool): If True, keeps the computation differentiable
                                 w.r.t. x, y, t (needed for residual checks).
                                 If False, wraps the temporal_mismatch re-query
                                 in torch.no_grad() for speed.

    Outputs:
        clean (dict): {"u", "v", "p"} clean predictions, each (N, 1).
        perturbed (dict): {"u", "v", "p"} hallucinated predictions, each (N, 1).
    """
    coords_tensor = torch.cat([x, y, t], dim=1)
    preds = model(coords_tensor)
    clean = {"u": preds[:, 0:1], "v": preds[:, 1:2], "p": preds[:, 2:3]}
    coords = {"x": x, "y": y, "t": t}

    model_arg = model if perturbation_name == "temporal_mismatch" else None
    perturbed = apply_perturbation(
        perturbation_name, clean, coords, params, epsilon,
        model=model_arg, no_grad=not track_gradients,
    )
    return clean, perturbed


def residual_stats(u, v, p, x, y, t, nu: float, scaler: ResidualScaler, band_sigma: float = 0.2, band_width_factor: float = 3.0):
    """
    Computes the scaled Navier-Stokes PDE residual MSEs for a given
    (u, v, p) field, both globally and split into a thin band near the
    periodic x-boundaries vs. the rest of the domain.

    The near-boundary split exists because the "boundary" perturbation's
    m(x) = exp(-x^2/sigma^2) + exp(-(2*pi-x)^2/sigma^2) is symmetric by
    construction: m(0) == m(2*pi) exactly, so it shifts both edges by the
    same amount and is invisible to a naive x=0 vs x=2*pi value comparison
    (see bc_violation_stats). Its actual signature is a spurious, spatially
    localized structure near BOTH edges that a translation-invariant
    periodic flow should not have -- which only shows up as an elevated
    residual concentrated in a thin band near x=0 and x=2*pi, diluted away
    in a whole-domain average. Splitting the residual by region surfaces it.

    Inputs:
        u, v, p (torch.Tensor): Velocity/pressure predictions, shape (N, 1),
                                 differentiable w.r.t. x, y, t.
        x, y, t (torch.Tensor): Coordinates with requires_grad=True, shape (N, 1).
        nu (float): Kinematic viscosity for this case.
        scaler (ResidualScaler): The case's residual scaler.
        band_sigma (float): The sigma used by the boundary perturbation's
                             Gaussian bumps; defines the band width below.
        band_width_factor (float): The near-boundary band extends
                                    band_width_factor * band_sigma in from
                                    each edge (default 3 sigma, i.e. where
                                    the boundary perturbation's Gaussian
                                    bumps have effectively decayed to ~0).

    Outputs:
        dict: {
            "mse_Ru", "mse_Rv", "mse_Rc": whole-domain scaled residual MSEs,
            "mse_Ru_near_boundary", "mse_Rv_near_boundary", "mse_Rc_near_boundary":
                scaled residual MSEs restricted to points within
                band_width_factor * band_sigma of x=0 or x=2*pi,
            "mse_Ru_far_from_boundary", "mse_Rv_far_from_boundary", "mse_Rc_far_from_boundary":
                the same, restricted to the remaining (interior) points.
        }
    """
    R_u, R_v, R_c = compute_residuals(u, v, p, x, y, t, nu)
    R_u_s, R_v_s, R_c_s = scaler.scale_residuals(R_u, R_v, R_c)

    with torch.no_grad():
        band_width = band_width_factor * band_sigma
        near_boundary_mask = ((x < band_width) | (x > (2 * torch.pi - band_width))).squeeze(-1)
        far_mask = ~near_boundary_mask

        def masked_mse(tensor_flat, mask):
            if mask.sum().item() == 0:
                return float("nan")
            return torch.mean(tensor_flat[mask] ** 2).item()

        Ru_flat, Rv_flat, Rc_flat = R_u_s.squeeze(-1), R_v_s.squeeze(-1), R_c_s.squeeze(-1)

        stats = {
            "mse_Ru": torch.mean(R_u_s ** 2).item(),
            "mse_Rv": torch.mean(R_v_s ** 2).item(),
            "mse_Rc": torch.mean(R_c_s ** 2).item(),
            "mse_Ru_near_boundary": masked_mse(Ru_flat, near_boundary_mask),
            "mse_Rv_near_boundary": masked_mse(Rv_flat, near_boundary_mask),
            "mse_Rc_near_boundary": masked_mse(Rc_flat, near_boundary_mask),
            "mse_Ru_far_from_boundary": masked_mse(Ru_flat, far_mask),
            "mse_Rv_far_from_boundary": masked_mse(Rv_flat, far_mask),
            "mse_Rc_far_from_boundary": masked_mse(Rc_flat, far_mask),
        }
    return stats


def bc_violation_stats(model, x_left, x_right, y_b, t_b, params, perturbation_name: str, epsilon: float):
    """
    Checks periodic boundary consistency of a perturbed field by comparing
    predictions at x=0 vs x=2*pi (same y, t) after applying the requested
    perturbation independently on each side.

    CAVEAT: this check is blind to the "boundary" perturbation itself. Its
    m(x) = exp(-x^2/sigma^2) + exp(-(2*pi-x)^2/sigma^2) satisfies
    m(0) == m(2*pi) exactly by construction, so both edges are shifted by
    the identical amount and a raw value comparison sees no mismatch. Use
    the near-boundary-band residual stats in residual_stats() instead to
    detect that perturbation's actual (spatially localized) violation.
    This check remains useful for the other 4 perturbation types, and for
    confirming the boundary perturbation's symmetric-by-design blind spot
    empirically rather than just asserting it.

    Inputs:
        model (nn.Module): The trained BaselinePINN.
        x_left, x_right, y_b, t_b (torch.Tensor): Paired boundary coordinates
                                                    from build_boundary_pair_grid,
                                                    shape (n_points, 1) each.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES.
        epsilon (float): Perturbation strength.

    Outputs:
        dict: {"bc_u_mismatch": float, "bc_v_mismatch": float,
               "bc_p_mismatch": float} mean squared differences between the
              left- and right-boundary hallucinated predictions.
    """
    with torch.no_grad():
        clean_left = model(torch.cat([x_left, y_b, t_b], dim=1))
        clean_right = model(torch.cat([x_right, y_b, t_b], dim=1))

    fields_left = {"u": clean_left[:, 0:1], "v": clean_left[:, 1:2], "p": clean_left[:, 2:3]}
    fields_right = {"u": clean_right[:, 0:1], "v": clean_right[:, 1:2], "p": clean_right[:, 2:3]}
    coords_left = {"x": x_left, "y": y_b, "t": t_b}
    coords_right = {"x": x_right, "y": y_b, "t": t_b}

    model_arg = model if perturbation_name == "temporal_mismatch" else None

    with torch.no_grad():
        pert_left = apply_perturbation(perturbation_name, fields_left, coords_left, params, epsilon, model=model_arg, no_grad=True)
        pert_right = apply_perturbation(perturbation_name, fields_right, coords_right, params, epsilon, model=model_arg, no_grad=True)

        u_mismatch = torch.mean((pert_left["u"] - pert_right["u"]) ** 2).item()
        v_mismatch = torch.mean((pert_left["v"] - pert_right["v"]) ** 2).item()
        p_mismatch = torch.mean((pert_left["p"] - pert_right["p"]) ** 2).item()

    return {"bc_u_mismatch": u_mismatch, "bc_v_mismatch": v_mismatch, "bc_p_mismatch": p_mismatch}


def analytical_deviation(u, v, x, y, t, case_meta: dict, nu: float):
    """
    Computes the relative L2 deviation of a hallucinated (u, v) field from
    the exact analytical Taylor-Green Vortex solution at the SAME (true,
    unshifted) coordinates. This is a general "how wrong is this vs. ground
    truth" check, most relevant for temporal_mismatch (which is expected to
    look like the correct solution at a different time) but computed for
    every perturbation type for completeness.

    Inputs:
        u, v (torch.Tensor): Hallucinated velocity predictions, shape (N, 1).
        x, y, t (torch.Tensor): Coordinates, shape (N, 1) (t is the TRUE,
                                 unshifted grid time, not epsilon-shifted).
        case_meta (dict): Case metadata; requires "U0", "k", "phi_x", "phi_y".
        nu (float): Kinematic viscosity for this case.

    Outputs:
        float: Relative L2 error between the hallucinated field and the
               analytical solution at (x, y, t).
    """
    U0, k, phi_x, phi_y = case_meta["U0"], case_meta["k"], case_meta["phi_x"], case_meta["phi_y"]
    with torch.no_grad():
        u_true, v_true, _ = generate_tgv(x, y, t, U0, k, phi_x, phi_y, nu)
        num = torch.sum((u - u_true) ** 2 + (v - v_true) ** 2)
        den = torch.sum(u_true ** 2 + v_true ** 2)
        rel_l2 = torch.sqrt(num / den).item() if den.item() > 0 else float("inf")
    return rel_l2


def visual_deviation(u_tilde, v_tilde, u_clean, v_clean):
    """
    Computes the relative L2 magnitude of the change a perturbation makes to
    the velocity field, as a numeric backing for the "imperceptible at low
    epsilon" claim (a small number here means the perturbed field is close
    to the clean field in an L2 sense, consistent with visual similarity).

    Inputs:
        u_tilde, v_tilde (torch.Tensor): Hallucinated velocity, shape (N, 1).
        u_clean, v_clean (torch.Tensor): Clean velocity, shape (N, 1).

    Outputs:
        float: Relative L2 norm of (u_tilde - u_clean, v_tilde - v_clean)
               against (u_clean, v_clean).
    """
    with torch.no_grad():
        num = torch.sum((u_tilde - u_clean) ** 2 + (v_tilde - v_clean) ** 2)
        den = torch.sum(u_clean ** 2 + v_clean ** 2)
        rel_l2 = torch.sqrt(num / den).item() if den.item() > 0 else float("inf")
    return rel_l2


def build_residual_table(model, case_id: str, case_meta: dict, args):
    """
    Runs the full quantitative verification sweep for a single case: every
    (perturbation, epsilon) combination against every violation metric
    (interior PDE residuals, boundary mismatch, analytical deviation, visual
    deviation), plus one "clean" reference row.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k", "phi_x", "phi_y").
        args (argparse.Namespace): Parsed CLI arguments (uses device, res,
                                    time_frac, n_bc).

    Outputs:
        list[dict]: One row per (perturbation, epsilon) combination (plus one
                    clean row), each containing case_id, perturbation_type,
                    epsilon, every computed violation metric (global and
                    near-boundary-band residuals, bc mismatch, analytical and
                    visual deviation), and "boundary_localization_ratio" --
                    the ratio of near-boundary to far-from-boundary residual,
                    which is the metric that actually detects the "boundary"
                    perturbation (see the caveat in residual_stats()).
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    nu = compute_nu(U0, Re, k)
    T = compute_T(U0, Re, k)
    scaler = ResidualScaler(U0, k)
    params = {"U0": U0, "k": k, "T": T}

    x, y, t = build_interior_grid(T, args.res, args.time_frac, args.device)
    x_left, x_right, y_b, t_b = build_boundary_pair_grid(T, args.n_bc, args.time_frac, args.device)

    rows = []

    # --- Clean reference row ---
    with torch.no_grad():
        clean_preds = model(torch.cat([x, y, t], dim=1))
    # Residuals of the clean field need their own gradient-tracked pass
    clean_u, clean_v, clean_p = compute_clean_and_perturbed(model, x, y, t, params, "velocity_divergence", 0.0)[0].values()
    clean_res = residual_stats(clean_u, clean_v, clean_p, x, y, t, nu, scaler)
    clean_bc = bc_violation_stats(model, x_left, x_right, y_b, t_b, params, "velocity_divergence", 0.0)
    clean_analytical = analytical_deviation(clean_u, clean_v, x, y, t, case_meta, nu)
    clean_localization_ratio = clean_res["mse_Ru_near_boundary"] / (clean_res["mse_Ru_far_from_boundary"] + 1e-30)

    rows.append({
        "case_id": case_id, "perturbation_type": "none", "epsilon": 0.0,
        "Re": Re, "U0": U0, "k": k, "T": T,
        **clean_res, **clean_bc,
        "rel_l2_vs_analytical": clean_analytical,
        "rel_l2_vs_clean": 0.0,
        "boundary_localization_ratio": clean_localization_ratio,
    })

    # --- Perturbed rows ---
    for perturbation_name in PERTURBATION_NAMES:
        for epsilon in EPSILON_VALUES:
            clean, perturbed = compute_clean_and_perturbed(model, x, y, t, params, perturbation_name, epsilon)
            res = residual_stats(perturbed["u"], perturbed["v"], perturbed["p"], x, y, t, nu, scaler)
            bc = bc_violation_stats(model, x_left, x_right, y_b, t_b, params, perturbation_name, epsilon)
            deviation_vs_true = analytical_deviation(perturbed["u"], perturbed["v"], x, y, t, case_meta, nu)
            deviation_vs_clean = visual_deviation(perturbed["u"], perturbed["v"], clean["u"], clean["v"])
            localization_ratio = res["mse_Ru_near_boundary"] / (res["mse_Ru_far_from_boundary"] + 1e-30)

            rows.append({
                "case_id": case_id, "perturbation_type": perturbation_name, "epsilon": epsilon,
                "Re": Re, "U0": U0, "k": k, "T": T,
                **res, **bc,
                "rel_l2_vs_analytical": deviation_vs_true,
                "rel_l2_vs_clean": deviation_vs_clean,
                "boundary_localization_ratio": localization_ratio,
            })

    return rows


def plot_visual_check(model, case_id: str, case_meta: dict, args, output_dir: Path):
    """
    Generates the required deliverable: side-by-side velocity-magnitude
    contour comparisons of the clean field vs. perturbed fields at
    epsilon in {0.01, 0.02}, for all 5 perturbation types, plus a
    difference panel to make subtle artifacts inspectable.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k").
        args (argparse.Namespace): Parsed CLI arguments (uses device, res, time_frac).
        output_dir (Path): Directory to save the figure into.

    Outputs:
        None. Saves "contour_eps_0.01_0.02.png" to output_dir.
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    T = compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T}
    res = args.res

    x, y, t = build_interior_grid(T, res, args.time_frac, args.device)

    fig, axes = plt.subplots(len(PERTURBATION_NAMES), 4, figsize=(16, 4 * len(PERTURBATION_NAMES)))
    fig.suptitle(f"Visual Imperceptibility Check: {case_id} (ε ∈ {{0.01, 0.02}})", fontsize=16)

    for row_idx, perturbation_name in enumerate(PERTURBATION_NAMES):
        clean, pert_001 = compute_clean_and_perturbed(model, x, y, t, params, perturbation_name, 0.01, track_gradients=False)
        _, pert_002 = compute_clean_and_perturbed(model, x, y, t, params, perturbation_name, 0.02, track_gradients=False)

        with torch.no_grad():
            mag_clean = torch.sqrt(clean["u"] ** 2 + clean["v"] ** 2).detach().reshape(res, res).cpu().numpy()
            mag_001 = torch.sqrt(pert_001["u"] ** 2 + pert_001["v"] ** 2).detach().reshape(res, res).cpu().numpy()
            mag_002 = torch.sqrt(pert_002["u"] ** 2 + pert_002["v"] ** 2).detach().reshape(res, res).cpu().numpy()
            diff_002 = np.abs(mag_002 - mag_clean)

        vmin, vmax = mag_clean.min(), mag_clean.max()

        panels = [
            (mag_clean, "Clean", vmin, vmax),
            (mag_001, "Perturbed ε=0.01", vmin, vmax),
            (mag_002, "Perturbed ε=0.02", vmin, vmax),
            (diff_002, "|Diff| ε=0.02", None, None),
        ]

        for col_idx, (data, title, pmin, pmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            cmap = "magma" if col_idx == 3 else "RdBu_r"
            im = ax.imshow(data.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                            cmap=cmap, vmin=pmin, vmax=pmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(perturbation_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = output_dir / "contour_eps_0.01_0.02.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def plot_full_sweep(model, case_id: str, case_meta: dict, args, output_dir: Path):
    """
    Generates the extra robustness figure: for every perturbation type, the
    |velocity magnitude difference from clean| across ALL 5 epsilon values,
    using a fixed per-row color scale so the imperceptible-to-obvious
    transition as epsilon grows is directly visible.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k").
        args (argparse.Namespace): Parsed CLI arguments (uses device, res, time_frac).
        output_dir (Path): Directory to save the figure into.

    Outputs:
        None. Saves "contour_full_sweep.png" to output_dir.
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    T = compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T}
    res = args.res

    x, y, t = build_interior_grid(T, res, args.time_frac, args.device)

    fig, axes = plt.subplots(len(PERTURBATION_NAMES), len(EPSILON_VALUES),
                              figsize=(3.2 * len(EPSILON_VALUES), 3.2 * len(PERTURBATION_NAMES)))
    fig.suptitle(f"Full Epsilon Sweep — |Velocity Magnitude Diff from Clean|: {case_id}", fontsize=16)

    for row_idx, perturbation_name in enumerate(PERTURBATION_NAMES):
        clean, _ = compute_clean_and_perturbed(model, x, y, t, params, perturbation_name, EPSILON_VALUES[0], track_gradients=False)
        with torch.no_grad():
            mag_clean = torch.sqrt(clean["u"] ** 2 + clean["v"] ** 2).detach()

        diffs = []
        for epsilon in EPSILON_VALUES:
            _, pert = compute_clean_and_perturbed(model, x, y, t, params, perturbation_name, epsilon, track_gradients=False)
            with torch.no_grad():
                mag_pert = torch.sqrt(pert["u"] ** 2 + pert["v"] ** 2).detach()
                diffs.append((mag_pert - mag_clean).abs().reshape(res, res).cpu().numpy())

        # Fixed color scale across this row's epsilons, so intensity growth is directly comparable.
        row_vmax = max(d.max() for d in diffs) + 1e-12

        for col_idx, (epsilon, diff) in enumerate(zip(EPSILON_VALUES, diffs)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(diff.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                            cmap="magma", vmin=0, vmax=row_vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx == 0:
                ax.set_title(f"ε={epsilon}")
            if col_idx == 0:
                ax.set_ylabel(perturbation_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = output_dir / "contour_full_sweep.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def plot_residual_curves(rows: list, case_id: str, output_dir: Path):
    """
    Plots each perturbation type's dominant violation metric against
    epsilon on a log-log scale, to visualize monotonic activation trends
    and compare relative severity across perturbation types.

    Inputs:
        rows (list[dict]): The residual summary rows for a single case, as
                            produced by build_residual_table().
        case_id (str): The case identifier (used in the plot title).
        output_dir (Path): Directory to save the figure into.

    Outputs:
        None. Saves "residual_curves.png" to output_dir.
    """
    # The metric most relevant to each perturbation's intended violation.
    # NOTE: "boundary" uses the near-boundary-band residual, not bc_u_mismatch --
    # see the caveat in bc_violation_stats(): the boundary perturbation's m(x)
    # is symmetric (m(0) == m(2*pi)) by construction, so a raw left/right value
    # comparison cannot see it; its violation only shows up as a spatially
    # localized residual spike near the edges.
    dominant_metric = {
        "velocity_divergence": "mse_Rc",
        "momentum": "mse_Ru",
        "pressure": "mse_Ru",
        "temporal_mismatch": "rel_l2_vs_analytical",
        "boundary": "mse_Ru_near_boundary",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for perturbation_name in PERTURBATION_NAMES:
        metric_key = dominant_metric[perturbation_name]
        eps_vals, metric_vals = [], []
        for row in rows:
            if row["perturbation_type"] == perturbation_name:
                eps_vals.append(row["epsilon"])
                metric_vals.append(row[metric_key])
        ax.plot(eps_vals, metric_vals, marker="o", label=f"{perturbation_name} ({metric_key})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epsilon (perturbation strength)")
    ax.set_ylabel("Violation metric (log scale)")
    ax.set_title(f"Violation Activation vs. Epsilon: {case_id}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "residual_curves.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def save_summary_table(rows: list, output_dir: Path, filename_stem: str):
    """
    Writes the residual/violation summary rows to both CSV and JSON.

    Inputs:
        rows (list[dict]): Summary rows to persist.
        output_dir (Path): Directory to save the files into.
        filename_stem (str): Base filename (without extension) for the two files.

    Outputs:
        None. Writes {filename_stem}.csv and {filename_stem}.json to output_dir.
    """
    if not rows:
        return

    csv_path = output_dir / f"{filename_stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"{filename_stem}.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"📊 Saved {csv_path.relative_to(project_root)}")
    print(f"📊 Saved {json_path.relative_to(project_root)}")


def main():
    """
    Entry point. Verifies one representative case with full plots + table
    (the required visual imperceptibility check plus the extra robustness
    figures), and optionally sweeps every trained case's quantitative
    violation table (--all_cases) to confirm the activation patterns hold
    across the whole dataset, not just one case.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes plots and summary tables under
        plots/hallucination_verification/, and prints progress to stdout.
    """
    args = parse_args()

    metadata_path = project_root / "data" / "cases_metadata.json"
    case_meta_by_id = load_case_metadata(metadata_path)
    models_dir = project_root / "models"

    output_root = Path(args.output_dir) if args.output_dir else project_root / "plots" / "hallucination_verification"
    output_root.mkdir(parents=True, exist_ok=True)

    trained_case_ids = sorted(
        cid for cid in case_meta_by_id
        if (models_dir / f"{cid}_best.pth").exists()
    )
    if not trained_case_ids:
        raise FileNotFoundError(f"No trained models found in {models_dir}")

    primary_case_id = args.case_id if args.case_id else trained_case_ids[0]
    if primary_case_id not in trained_case_ids:
        raise ValueError(f"{primary_case_id} has no trained model in {models_dir}")

    print("=" * 60)
    print("🔬 VERIFYING PHYSICAL HALLUCINATION DATASET (Section 7 sanity checks)")
    print(f"Primary case (plots + table): {primary_case_id}")
    print(f"Visual imperceptibility check restricted to ε ∈ {VISUAL_CHECK_EPSILONS}")
    print(f"Quantitative residual checks run across ε ∈ {EPSILON_VALUES}")
    print("=" * 60)

    case_output_dir = output_root / primary_case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    case_meta = case_meta_by_id[primary_case_id]
    model = load_model(primary_case_id, case_meta["k"], args.device)

    print(f"\n[{primary_case_id}] Generating required visual imperceptibility plot (ε=0.01, 0.02)...")
    plot_visual_check(model, primary_case_id, case_meta, args, case_output_dir)

    print(f"[{primary_case_id}] Generating extra full-epsilon-sweep robustness plot...")
    plot_full_sweep(model, primary_case_id, case_meta, args, case_output_dir)

    print(f"[{primary_case_id}] Running quantitative residual / violation checks across all ε...")
    primary_rows = build_residual_table(model, primary_case_id, case_meta, args)
    save_summary_table(primary_rows, case_output_dir, "residual_summary")
    plot_residual_curves(primary_rows, primary_case_id, case_output_dir)

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if args.all_cases:
        print("\n" + "=" * 60)
        print(f"🔁 Sweeping quantitative checks across all {len(trained_case_ids)} trained cases...")
        print("=" * 60)

        all_rows = list(primary_rows)
        for case_id in trained_case_ids:
            if case_id == primary_case_id:
                continue
            case_meta = case_meta_by_id[case_id]
            model = load_model(case_id, case_meta["k"], args.device)
            print(f"[{case_id}] Running quantitative checks...")
            rows = build_residual_table(model, case_id, case_meta, args)
            all_rows.extend(rows)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

        save_summary_table(all_rows, output_root, "residual_summary_all_cases")

    print("\n" + "=" * 60)
    print("✅ Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
