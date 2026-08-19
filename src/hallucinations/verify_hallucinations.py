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


def phase_amplitude_fit_diagnostic(model, x, y, t, case_meta: dict, nu: float, n_phase_steps: int = 36):
    """
    Empirically tests whether a model's clean prediction is a phase-shifted
    (rather than genuinely wrong) member of the Taylor-Green Vortex family.

    WHY THIS EXISTS: the governing PDE + periodic boundary condition are
    translation-invariant, so u(x,y,t; phi_x, phi_y) satisfies the same
    physics for ANY (phi_x, phi_y) -- the PDE residual and periodic-BC loss
    are structurally blind to phase. Only the initial-condition loss anchors
    a trained model to the SPECIFIC target phase. If IC loss is under-weighted
    or under-converged, a model can have near-zero PDE/BC residuals while
    sitting on the wrong member of that family -- which shows up as a large
    rel_l2_vs_analytical even though the model is physically self-consistent.

    This function searches over a grid of candidate phases (phi_x', phi_y')
    and, for each, finds the closed-form optimal amplitude scale that best
    matches the model's clean output to that phase-shifted analytical
    solution. If the best achievable residual after this search is small,
    it confirms the "phase mismatch" explanation empirically. If it remains
    large even at the best-fit phase, something else is going on (e.g. a
    genuine training failure) and the phase-symmetry explanation does not apply.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        x, y, t (torch.Tensor): Coordinates, shape (N, 1) (t is the TRUE,
                                 unshifted grid time).
        case_meta (dict): Case metadata; requires "U0", "k", "phi_x", "phi_y"
                           (the target phase, for comparison against the fit).
        nu (float): Kinematic viscosity for this case.
        n_phase_steps (int): Resolution of the (phi_x, phi_y) search grid
                              (n_phase_steps x n_phase_steps combinations).

    Outputs:
        dict: {
            "target_phi_x", "target_phi_y": the case's true target phase,
            "best_fit_phi_x", "best_fit_phi_y": the phase that best matches
                the model's clean output,
            "best_fit_amplitude_scale": the optimal amplitude scale factor
                (relative to case U0) at that best-fit phase,
            "rel_l2_at_target_phase": the original rel_l2_vs_analytical,
                i.e. how wrong the model looks WITHOUT allowing phase to vary,
            "rel_l2_at_best_fit_phase": how wrong the model looks AFTER
                allowing phase (and amplitude) to vary -- small here + large
                above confirms the phase-mismatch explanation.
        }
    """
    U0, k, target_phi_x, target_phi_y = case_meta["U0"], case_meta["k"], case_meta["phi_x"], case_meta["phi_y"]

    with torch.no_grad():
        model_preds = model(torch.cat([x, y, t], dim=1))
        u_model, v_model = model_preds[:, 0:1], model_preds[:, 1:2]

        rel_l2_at_target = analytical_deviation(u_model, v_model, x, y, t, case_meta, nu)

        phase_candidates = torch.linspace(0, 2 * torch.pi, n_phase_steps + 1)[:-1]
        best_rel_l2 = float("inf")
        best_phi_x, best_phi_y, best_scale = target_phi_x, target_phi_y, 1.0

        for phi_x_c in phase_candidates:
            for phi_y_c in phase_candidates:
                u_ref, v_ref, _ = generate_tgv(x, y, t, U0, k, phi_x_c.item(), phi_y_c.item(), nu)

                # Closed-form optimal amplitude scale c minimizing ||model - c*ref||^2:
                # c* = <model, ref> / <ref, ref>
                numerator = torch.sum(u_model * u_ref + v_model * v_ref)
                denominator = torch.sum(u_ref ** 2 + v_ref ** 2)
                scale = (numerator / denominator).item() if denominator.item() > 0 else 0.0

                resid = torch.sum((u_model - scale * u_ref) ** 2 + (v_model - scale * v_ref) ** 2)
                total = torch.sum(u_model ** 2 + v_model ** 2)
                rel_l2 = torch.sqrt(resid / total).item() if total.item() > 0 else float("inf")

                if rel_l2 < best_rel_l2:
                    best_rel_l2 = rel_l2
                    best_phi_x, best_phi_y, best_scale = phi_x_c.item(), phi_y_c.item(), scale

    return {
        "target_phi_x": target_phi_x, "target_phi_y": target_phi_y,
        "best_fit_phi_x": best_phi_x, "best_fit_phi_y": best_phi_y,
        "best_fit_amplitude_scale": best_scale,
        "rel_l2_at_target_phase": rel_l2_at_target,
        "rel_l2_at_best_fit_phase": best_rel_l2,
    }


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
    split = case_meta.get("split", "unknown")
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
        "case_id": case_id, "split": split, "perturbation_type": "none", "epsilon": 0.0,
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
                "case_id": case_id, "split": split, "perturbation_type": perturbation_name, "epsilon": epsilon,
                "Re": Re, "U0": U0, "k": k, "T": T,
                **res, **bc,
                "rel_l2_vs_analytical": deviation_vs_true,
                "rel_l2_vs_clean": deviation_vs_clean,
                "boundary_localization_ratio": localization_ratio,
            })

    return rows


def safe_diff_vlim(diff_array: np.ndarray, floor: float = 1e-12):
    """
    Computes a non-degenerate (vmin, vmax) pair for imshow'ing a
    difference-from-clean array. A perturbation that doesn't touch the
    plotted field (e.g. "pressure" leaves u, v exactly unchanged) produces
    an all-zero diff array; imshow's default auto-scaling (vmin=vmax=None)
    degenerates on a constant array and renders an arbitrary flat color
    instead of a genuinely "nothing changed" panel. Forcing a small,
    explicit positive vmax avoids that and reads correctly as "near zero."

    Inputs:
        diff_array (np.ndarray): The |perturbed - clean| array to be plotted.
        floor (float): Minimum vmax to use when the array is exactly (or
                        nearly) constant, so the color scale never degenerates.

    Outputs:
        tuple: (vmin, vmax) = (0.0, max(diff_array.max(), floor)).
    """
    return 0.0, max(float(diff_array.max()), floor)


def plot_visual_check(model, case_id: str, case_meta: dict, args, output_dir: Path):
    """
    Generates the required deliverable: side-by-side velocity-magnitude
    contour comparisons of the clean field vs. perturbed fields at
    epsilon in {0.01, 0.02}, for all 5 perturbation types, plus a
    difference panel to make subtle artifacts inspectable.

    NOTE: the "pressure" perturbation only redefines p and leaves (u, v)
    exactly unchanged (see perturb_pressure() in perturbations.py), so its
    row's velocity diff panel is mathematically guaranteed to be all-zero
    -- this is expected, not a bug, and is annotated as such rather than
    left as an unexplained blank/flat panel. A companion pressure-field
    contour is saved separately for that perturbation via
    plot_pressure_field_check(), since a velocity-only view cannot show it.

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
        diff_vmin, diff_vmax = safe_diff_vlim(diff_002)
        is_degenerate = diff_002.max() < 1e-10

        panels = [
            (mag_clean, "Clean", vmin, vmax),
            (mag_001, "Perturbed ε=0.01", vmin, vmax),
            (mag_002, "Perturbed ε=0.02", vmin, vmax),
            (diff_002, "|Diff| ε=0.02", diff_vmin, diff_vmax),
        ]

        for col_idx, (data, title, pmin, pmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            cmap = "magma" if col_idx == 3 else "RdBu_r"
            im = ax.imshow(data.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                            cmap=cmap, vmin=pmin, vmax=pmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if col_idx == 3 and is_degenerate:
                ax.text(np.pi, np.pi, "u, v unaffected\n(by design)",
                        ha="center", va="center", color="white", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(perturbation_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = output_dir / "contour_eps_0.01_0.02.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def plot_pressure_field_check(model, case_id: str, case_meta: dict, args, output_dir: Path):
    """
    Generates a companion contour comparison in PRESSURE space (Clean p vs.
    Perturbed p at eps in {0.01, 0.02}) for the "pressure" perturbation
    specifically. Since that perturbation leaves (u, v) exactly unchanged,
    the velocity-based plot_visual_check() panel is guaranteed to show
    nothing for it; this figure confirms the perturbation is actually
    happening, just in the field it's designed to affect.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k").
        args (argparse.Namespace): Parsed CLI arguments (uses device, res, time_frac).
        output_dir (Path): Directory to save the figure into.

    Outputs:
        None. Saves "contour_pressure_eps_0.01_0.02.png" to output_dir.
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    T = compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T}
    res = args.res

    x, y, t = build_interior_grid(T, res, args.time_frac, args.device)

    clean, pert_001 = compute_clean_and_perturbed(model, x, y, t, params, "pressure", 0.01, track_gradients=False)
    _, pert_002 = compute_clean_and_perturbed(model, x, y, t, params, "pressure", 0.02, track_gradients=False)

    with torch.no_grad():
        p_clean = clean["p"].detach().reshape(res, res).cpu().numpy()
        p_001 = pert_001["p"].detach().reshape(res, res).cpu().numpy()
        p_002 = pert_002["p"].detach().reshape(res, res).cpu().numpy()
        diff_002 = np.abs(p_002 - p_clean)

    vmin, vmax = p_clean.min(), p_clean.max()
    diff_vmin, diff_vmax = safe_diff_vlim(diff_002)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Pressure-Space Check (velocity is unaffected by design): {case_id}", fontsize=14)

    panels = [
        (p_clean, "Clean p", vmin, vmax),
        (p_001, "Perturbed p, ε=0.01", vmin, vmax),
        (p_002, "Perturbed p, ε=0.02", vmin, vmax),
        (diff_002, "|Diff| p, ε=0.02", diff_vmin, diff_vmax),
    ]
    for col_idx, (data, title, pmin, pmax) in enumerate(panels):
        ax = axes[col_idx]
        cmap = "magma" if col_idx == 3 else "RdBu_r"
        im = ax.imshow(data.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap=cmap, vmin=pmin, vmax=pmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    plt.tight_layout()
    save_path = output_dir / "contour_pressure_eps_0.01_0.02.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def plot_vector_field_check(model, case_id: str, case_meta: dict, args, output_dir: Path,
                              vector_res: int = 18, demo_epsilon: float = 0.1):
    """
    Generates quiver (vector) plots showing how each Section 7 perturbation
    affects the DIRECTION of the velocity field, not just its magnitude.

    WHY THIS IS A DIFFERENT CHECK FROM THE CONTOUR PLOTS: velocity-magnitude
    contours (plot_visual_check, plot_full_sweep) show sqrt(u^2+v^2), which is
    blind to pure rotation -- a perturbation could rotate every vector and
    leave the magnitude field completely unchanged. This function makes the
    directional effect visible directly, and the five perturbation types have
    genuinely different signatures here:
      - velocity_divergence only modifies u, so it TILTS vectors (breaks the
        u/v balance continuity depends on), not just rescales them.
      - momentum perturbs u and v independently, producing a more complex,
        spatially varying rotation pattern.
      - pressure should show EXACTLY zero vector change (u, v untouched by
        design), same caveat as the pressure-space contour check.
      - temporal_mismatch, if the model tracks the analytical TGV decay
        reasonably, should mostly RESCALE vectors uniformly rather than
        rotate them (the decay factor multiplies u and v identically), unlike
        velocity_divergence/momentum which genuinely change direction.
      - boundary tilts vectors, but only in a thin band near the periodic
        x-edges, leaving the interior essentially untouched.

    NOTE: unlike the imperceptibility check, this uses a large, clearly
    visible epsilon (default 0.1) by design -- the goal here is showing the
    MECHANISM of each perturbation, not re-testing whether it's invisible
    (that's already covered by plot_visual_check at eps in {0.01, 0.02}).

    Uses a coarse grid (vector_res x vector_res, default 18x18) since dense
    quiver plots become unreadable clutter. Clean and perturbed columns
    within each row share an identical arrow-length scale and color range
    (derived from the clean field) so a fair visual comparison is possible;
    the difference column gets its own scale since it's a much smaller quantity.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k").
        args (argparse.Namespace): Parsed CLI arguments (uses device, time_frac).
        output_dir (Path): Directory to save the figure into.
        vector_res (int): Quiver grid resolution (vector_res x vector_res arrows).
        demo_epsilon (float): Perturbation strength used for this demonstration.

    Outputs:
        None. Saves "vector_field_eps_{demo_epsilon}.png" to output_dir.
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    T = compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T}

    x, y, t = build_interior_grid(T, vector_res, args.time_frac, args.device)

    fig, axes = plt.subplots(len(PERTURBATION_NAMES), 3, figsize=(12, 4 * len(PERTURBATION_NAMES)))
    fig.suptitle(f"Vector Field Distortion Check: {case_id} (ε={demo_epsilon})", fontsize=16)

    X = x.detach().reshape(vector_res, vector_res).cpu().numpy()
    Y = y.detach().reshape(vector_res, vector_res).cpu().numpy()

    for row_idx, perturbation_name in enumerate(PERTURBATION_NAMES):
        clean, perturbed = compute_clean_and_perturbed(
            model, x, y, t, params, perturbation_name, demo_epsilon, track_gradients=False
        )
        with torch.no_grad():
            U_clean = clean["u"].detach().reshape(vector_res, vector_res).cpu().numpy()
            V_clean = clean["v"].detach().reshape(vector_res, vector_res).cpu().numpy()
            U_pert = perturbed["u"].detach().reshape(vector_res, vector_res).cpu().numpy()
            V_pert = perturbed["v"].detach().reshape(vector_res, vector_res).cpu().numpy()
            speed_clean = np.sqrt(U_clean ** 2 + V_clean ** 2)
            speed_pert = np.sqrt(U_pert ** 2 + V_pert ** 2)
            U_diff = U_pert - U_clean
            V_diff = V_pert - V_clean
            speed_diff = np.sqrt(U_diff ** 2 + V_diff ** 2)

        vmin, vmax = speed_clean.min(), speed_clean.max()

        ax_clean = axes[row_idx, 0]
        Q_clean = ax_clean.quiver(X, Y, U_clean, V_clean, speed_clean, cmap="viridis", clim=(vmin, vmax))
        shared_scale = Q_clean.scale

        ax_pert = axes[row_idx, 1]
        is_degenerate = speed_diff.max() < 1e-10
        if is_degenerate:
            # Perturbed vectors are identical to clean (e.g. "pressure"); plot
            # them anyway so the panel isn't blank, but flag it explicitly
            # rather than leaving an unexplained duplicate of column 1.
            ax_pert.quiver(X, Y, U_pert, V_pert, speed_pert, cmap="viridis", clim=(vmin, vmax),
                            scale=shared_scale, scale_units=Q_clean.scale_units)
            ax_pert.text(np.pi, np.pi, "u, v unaffected\n(by design)", ha="center", va="center",
                         fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            ax_pert.quiver(X, Y, U_pert, V_pert, speed_pert, cmap="viridis", clim=(vmin, vmax),
                            scale=shared_scale, scale_units=Q_clean.scale_units)

        ax_diff = axes[row_idx, 2]
        if is_degenerate:
            ax_diff.text(np.pi, np.pi, "no difference\n(by design)", ha="center", va="center",
                         fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax_diff.set_xlim(0, 2 * np.pi)
            ax_diff.set_ylim(0, 2 * np.pi)
        else:
            # Explicit scale rather than matplotlib's auto-heuristic: some
            # perturbations (e.g. "boundary") are spatially sparse/localized,
            # and auto-scaling calibrated against a field of near-zero values
            # with a few real outliers renders those outliers as misleadingly
            # long streaks. Pinning the longest arrow to ~0.8 grid-cells keeps
            # every row's diff panel honestly proportioned to where the
            # perturbation actually has an effect.
            grid_spacing = (2 * np.pi) / max(vector_res - 1, 1)
            max_diff_speed = speed_diff.max()
            diff_scale = max_diff_speed / (0.8 * grid_spacing) if max_diff_speed > 1e-12 else 1.0
            ax_diff.quiver(X, Y, U_diff, V_diff, speed_diff, cmap="magma", scale=diff_scale, scale_units="xy")

        for ax in (ax_clean, ax_pert, ax_diff):
            ax.set_aspect("equal")
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)

        axes[row_idx, 0].set_ylabel(perturbation_name, fontsize=11, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Clean")
            axes[row_idx, 1].set_title(f"Perturbed ε={demo_epsilon}")
            axes[row_idx, 2].set_title("Difference vectors")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = output_dir / f"vector_field_eps_{demo_epsilon}.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"🖼️  Saved {save_path.relative_to(project_root)}")


def plot_full_sweep(model, case_id: str, case_meta: dict, args, output_dir: Path):
    """
    Generates the extra robustness figure: for every perturbation type, the
    |velocity magnitude difference from clean| across ALL 5 epsilon values,
    using a fixed per-row color scale so the imperceptible-to-obvious
    transition as epsilon grows is directly visible.

    NOTE: the "pressure" perturbation only redefines p and leaves (u, v)
    exactly unchanged, so its row is expected to render as solid black
    (zero velocity diff) at every epsilon -- that row is annotated
    accordingly rather than left looking like a stalled computation. See
    plot_pressure_field_check() for the corresponding pressure-space view.

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
        is_degenerate_row = max(d.max() for d in diffs) < 1e-10

        for col_idx, (epsilon, diff) in enumerate(zip(EPSILON_VALUES, diffs)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(diff.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                            cmap="magma", vmin=0, vmax=row_vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if is_degenerate_row and col_idx == len(EPSILON_VALUES) // 2:
                ax.text(np.pi, np.pi, "u, v unaffected\n(by design)",
                        ha="center", va="center", color="white", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))
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
    #
    # NOTE: "temporal_mismatch" uses rel_l2_vs_clean (self-consistency: how much
    # the model's own output changes under the time shift), not
    # rel_l2_vs_analytical (deviation from ground truth). The latter is
    # confounded by the model's own baseline training quality -- if the clean
    # prediction is already far from the analytical solution (e.g. an
    # under-trained model), that baseline error dominates the ratio and
    # swamps the epsilon-dependent signal, making the curve look falsely
    # flat. rel_l2_vs_clean isolates the perturbation's own effect and scales
    # with epsilon regardless of model quality. rel_l2_vs_analytical is still
    # plotted as a dashed reference line so that saturation is visible rather
    # than silently discarded.
    dominant_metric = {
        "velocity_divergence": "mse_Rc",
        "momentum": "mse_Ru",
        "pressure": "mse_Ru",
        "temporal_mismatch": "rel_l2_vs_clean",
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

    # Secondary reference line: shows whether rel_l2_vs_analytical is
    # saturated by baseline model error (flat) or tracking epsilon (rising).
    eps_vals_temporal, analytical_vals = [], []
    for row in rows:
        if row["perturbation_type"] == "temporal_mismatch":
            eps_vals_temporal.append(row["epsilon"])
            analytical_vals.append(row["rel_l2_vs_analytical"])
    ax.plot(eps_vals_temporal, analytical_vals, marker="x", linestyle="--", color="gray", alpha=0.7,
            label="temporal_mismatch (rel_l2_vs_analytical, reference)")

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


def check_phase_mismatch(model, case_id: str, case_meta: dict, rows: list, args, output_dir: Path, threshold: float = 0.1):
    """
    Runs phase_amplitude_fit_diagnostic() if (and only if) this case's clean
    rel_l2_vs_analytical exceeds `threshold`, prints an interpretable verdict,
    and saves the result. Factored out so it can be applied uniformly to the
    primary case and to every case in an --all_cases sweep, rather than only
    the one case picked for full plotting.

    Inputs:
        model (nn.Module): The trained BaselinePINN for this case.
        case_id (str): The case identifier.
        case_meta (dict): This case's metadata ("Re", "U0", "k", "phi_x", "phi_y").
        rows (list[dict]): This case's residual summary rows, as produced by
                            build_residual_table() (used to read the clean
                            baseline's rel_l2_vs_analytical without recomputing it).
        args (argparse.Namespace): Parsed CLI arguments (uses device, res, time_frac).
        output_dir (Path): Directory to save "{case_id}_phase_diagnostic.csv/.json" into.
        threshold (float): Only runs the (relatively expensive) phase search
                            if the clean rel_l2_vs_analytical exceeds this value.

    Outputs:
        dict | None: The diagnostic result dict (see phase_amplitude_fit_diagnostic()),
                     with "case_id" added, or None if the threshold wasn't exceeded.
    """
    clean_row = next(r for r in rows if r["perturbation_type"] == "none")
    if clean_row["rel_l2_vs_analytical"] <= threshold:
        return None

    print(f"\n[{case_id}] Clean rel_l2_vs_analytical = {clean_row['rel_l2_vs_analytical']:.3f} is large -- "
          f"running phase/amplitude-fit diagnostic to check whether this is a phase mismatch "
          f"(PDE + periodic BC are translation-invariant, so residual loss alone can't fix phase)...")

    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    nu = compute_nu(U0, Re, k)
    T = compute_T(U0, Re, k)
    x_diag, y_diag, t_diag = build_interior_grid(T, args.res, args.time_frac, args.device)
    diagnostic = phase_amplitude_fit_diagnostic(model, x_diag, y_diag, t_diag, case_meta, nu)
    diagnostic_row = {"case_id": case_id, "split": case_meta.get("split", "unknown"), **diagnostic}
    save_summary_table([diagnostic_row], output_dir, f"{case_id}_phase_diagnostic")

    print(f"  Target phase:    (phi_x={diagnostic['target_phi_x']:.3f}, phi_y={diagnostic['target_phi_y']:.3f})")
    print(f"  Best-fit phase:  (phi_x={diagnostic['best_fit_phi_x']:.3f}, phi_y={diagnostic['best_fit_phi_y']:.3f}), "
          f"amplitude scale={diagnostic['best_fit_amplitude_scale']:.3f}")
    print(f"  rel_l2 at target phase:   {diagnostic['rel_l2_at_target_phase']:.3f}")
    print(f"  rel_l2 at best-fit phase: {diagnostic['rel_l2_at_best_fit_phase']:.3f}")
    if diagnostic["rel_l2_at_best_fit_phase"] < threshold:
        print("  ✅ VERDICT: phase mismatch confirmed -- model found a physically valid but wrong-phase TGV "
              "solution. Check IC loss weighting for this case's training run.")
    else:
        print("  ⚠️  VERDICT: NOT a simple phase mismatch -- error remains large even at the best-fit phase. "
              "This points to a genuine model quality issue, not just an IC/phase symmetry artifact.")

    return diagnostic_row


def main():
    """
    Entry point. Verifies one representative case with full plots + table
    (the required visual imperceptibility check plus the extra robustness
    figures), and optionally sweeps every trained case's quantitative
    violation table (--all_cases) to confirm the activation patterns hold
    across the whole dataset, not just one case.

    If any case's clean rel_l2_vs_analytical is large (>0.1), also runs
    check_phase_mismatch() / phase_amplitude_fit_diagnostic() on that case
    to empirically test whether it's explained by the PDE's translation
    invariance (a phase-mismatched but otherwise physically valid solution)
    rather than a genuine model quality issue -- see phase_amplitude_fit_diagnostic()'s
    docstring for the underlying reasoning. Runs for the primary case always,
    and for every case when --all_cases is set, so a systemic IC-loss-weighting
    issue (vs. a one-off) can be told apart from the aggregated results.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes plots and summary tables (and, when triggered, one
        {case_id}_phase_diagnostic.csv/.json per flagged case, plus
        phase_diagnostic_all_cases.csv/.json when --all_cases is set) under
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

    print(f"[{primary_case_id}] Generating companion pressure-field check (pressure perturbation doesn't touch velocity)...")
    plot_pressure_field_check(model, primary_case_id, case_meta, args, case_output_dir)

    print(f"[{primary_case_id}] Generating vector field distortion check...")
    plot_vector_field_check(model, primary_case_id, case_meta, args, case_output_dir, demo_epsilon=1.0)

    print(f"[{primary_case_id}] Generating extra full-epsilon-sweep robustness plot...")
    plot_full_sweep(model, primary_case_id, case_meta, args, case_output_dir)

    print(f"[{primary_case_id}] Running quantitative residual / violation checks across all ε...")
    primary_rows = build_residual_table(model, primary_case_id, case_meta, args)
    save_summary_table(primary_rows, case_output_dir, "residual_summary")
    plot_residual_curves(primary_rows, primary_case_id, case_output_dir)

    phase_diagnostic_rows = []
    primary_diagnostic = check_phase_mismatch(model, primary_case_id, case_meta, primary_rows, args, case_output_dir)
    if primary_diagnostic:
        phase_diagnostic_rows.append(primary_diagnostic)

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

            diagnostic = check_phase_mismatch(model, case_id, case_meta, rows, args, output_root)
            if diagnostic:
                phase_diagnostic_rows.append(diagnostic)

            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

        save_summary_table(all_rows, output_root, "residual_summary_all_cases")

        split_counts = {"train": 0, "validation": 0, "test": 0}
        for cid in trained_case_ids:
            s = case_meta_by_id[cid].get("split", "unknown")
            if s in split_counts:
                split_counts[s] += 1
        print(f"\n📁 Trained cases by split: {split_counts['train']} train / "
              f"{split_counts['validation']} validation / {split_counts['test']} test "
              f"(per src/data/sampler.py's 20/5/5 partition)")

        if phase_diagnostic_rows:
            save_summary_table(phase_diagnostic_rows, output_root, "phase_diagnostic_all_cases")
            n_confirmed = sum(1 for d in phase_diagnostic_rows if d["rel_l2_at_best_fit_phase"] < 0.1)
            print(f"\n📋 Phase mismatch flagged in {len(phase_diagnostic_rows)}/{len(trained_case_ids)} cases "
                  f"({n_confirmed} confirmed as pure phase mismatch, "
                  f"{len(phase_diagnostic_rows) - n_confirmed} likely genuine model quality issues). "
                  f"See phase_diagnostic_all_cases.csv for the full breakdown.")
            flagged_by_split = {"train": 0, "validation": 0, "test": 0}
            for d in phase_diagnostic_rows:
                s = d.get("split", "unknown")
                if s in flagged_by_split:
                    flagged_by_split[s] += 1
            print(f"   Flagged by split: {flagged_by_split['train']}/{split_counts['train']} train, "
                  f"{flagged_by_split['validation']}/{split_counts['validation']} validation, "
                  f"{flagged_by_split['test']}/{split_counts['test']} test")

    print("\n" + "=" * 60)
    print("✅ Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()