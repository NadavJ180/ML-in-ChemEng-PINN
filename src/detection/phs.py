"""
Physical Hallucination Score (PHS) -- Core Formula Module (Section 8, WP5)

Implements the 4 raw violation components and the normalization/scoring
pipeline defined in the project write-up's Section 8:

    Smom = MSE(Ru) + MSE(Rv)                          (momentum violation)
    Sdiv = MSE(ux + vy)                                (divergence violation,
                                                         i.e. MSE(Rc))
    Sbc  = MSE(s|x=0 - s|x=2pi) + MSE(s|y=0 - s|y=2pi)  (boundary violation,
                                                         s = (u, v, p))
    E(t)     = mean_{x,y}[ (u^2 + v^2) / 2 ]            (kinetic energy)
    Ephys(t) = E(0) * exp(-4 * nu * k^2 * t)            (TGV's analytical
                                                         decay)
    SE   = MSE_t( E(t) - Ephys(t) )                     (energy violation)

    S_bar_j = Sj / (mean(Sj over clean VALIDATION-split fields) + 1e-12)
    PHS     = S_bar_mom + S_bar_div + S_bar_bc + S_bar_E
    tau     = percentile95(PHS over clean VALIDATION-split fields)
    hallucinated  <=>  PHS > tau

WP5 also asks for 2 residual-only baselines PHS is benchmarked against:
    Score1 = S_bar_mom
    Score2 = S_bar_mom + S_bar_div
    Score3 = PHS (all four terms)
(see BASELINE_DEFINITIONS below). The write-up's Score1/Score2 are written
in terms of the raw Smom/Sdiv, but every score here is built from the SAME
normalized components PHS uses -- otherwise Score1/Score2 would inherit
the raw cross-case scale problem described below, making the baseline
comparison meaningless. This module always operates on S_bar_j.

TWO DELIBERATE DEVIATIONS FROM THE LITERAL SECTION 8 NOTATION (both
documented again at their point of use below):

  1. Smom, Sdiv are computed from residuals that are FIRST non-dimension-
     alized via ResidualScaler (src.models.scaling), exactly as loss.py and
     verify_hallucinations.py already do -- not left in raw physical units.
     Without this, a case with a large sampled U0 would trivially have a
     much larger Smom/Sdiv than a low-U0 case even with an equally
     well-trained model, which would corrupt the cross-case pooled
     normalization below (mean(Sj_valid) would be dominated by whichever
     validation cases happen to have large U0).

  2. Sbc's u, v differences are divided by U0 and its p difference by
     scale_p = U0^2 before squaring, mirroring loss.py's compute_bc_loss.
     Same cross-case-comparability reason as (1).

KNOWN BLIND SPOT (confirmed empirically during Issue #9's verification
audit, see verify_hallucinations.py's bc_violation_stats docstring): the
"boundary" perturbation's m(x) = exp(-x^2/sigma^2) + exp(-(2*pi-x)^2/sigma^2)
satisfies m(0) == m(2*pi) by construction, so BOTH x-edges are shifted by
the identical amount and Sbc's raw value comparison cannot see it -- Sbc
does NOT rise for the "boundary" perturbation type specifically. This does
not break PHS as a whole: Smom/Sdiv DO rise for "boundary" (confirmed in
Issue #9's residual_summary.json -- whole-domain mse_Ru rises from ~3e-6 to
~1.7e-3 across the epsilon sweep for case_00), since the added near-edge
bump still shows up in u_x, u_xx feeding the momentum/divergence residuals.
PHS's sum therefore still increases for every one of the 5 perturbation
types; it just isn't Sbc doing the work for this particular one.

Every function below takes an already-loaded, eval-mode BaselinePINN and
computes one thing at a time, mirroring the split used throughout
src/hallucinations/verify_hallucinations.py (small, single-purpose,
independently testable functions; looping over cases/perturbations/epsilons
is left to the calling script, evaluate_phs.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.physics.navier_stokes import compute_residuals
from src.physics.taylor_green import generate_tgv
from src.data.point_samplers import sample_interior_points, sample_periodic_boundaries
from src.hallucinations.perturbations import apply_perturbation

# The four raw PHS components, per Section 8.
PHS_COMPONENT_NAMES = ["mom", "div", "bc", "E"]

# WP5's three detection scores, expressed as which normalized (S_bar)
# components each one sums. Score3_PHS_full IS the Physical Hallucination
# Score; Score1/Score2 are the residual-only baselines it is compared
# against for the AUC(PHS) > AUC(Smom + Sdiv) acceptance criterion.
BASELINE_DEFINITIONS = {
    "Score1_momentum_only": ["mom"],
    "Score2_momentum_divergence": ["mom", "div"],
    "Score3_PHS_full": ["mom", "div", "bc", "E"],
}


def _leaf(coords_tensor: torch.Tensor, col: int, device: str) -> torch.Tensor:
    """
    Slices one column out of a plain (N, 3) coordinate tensor (as returned
    by src.data.point_samplers) and returns it as an INDEPENDENT,
    gradient-tracked leaf tensor.

    WHY THIS EXISTS: computing Ru, Rv (and their x/y/t derivatives) via
    autograd requires x, y, t to each be their own leaf tensor with
    requires_grad=True -- a view/slice of one shared tensor would not give
    independent d/dx, d/dy, d/dt gradients.

    Inputs:
        coords_tensor (torch.Tensor): Shape (N, 3), columns ordered (x, y, t).
        col (int): Which column to extract (0=x, 1=y, 2=t).
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        torch.Tensor: Shape (N, 1), dtype float64, requires_grad=True, on `device`.
    """
    return coords_tensor[:, col:col + 1].clone().to(device=device, dtype=torch.float64).requires_grad_(True)


def _field_at(model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
              params: dict, perturbation_name: str, epsilon: float, no_grad: bool) -> dict:
    """
    Runs the model at (x, y, t) and applies the requested perturbation on
    top, or returns the clean prediction unchanged if perturbation_name is
    "none" (the clean-baseline sentinel used throughout this module --
    NOT a key in PERTURBATION_REGISTRY, so it must be special-cased here
    rather than forwarded to apply_perturbation).

    Inputs:
        model (nn.Module): The trained BaselinePINN.
        x, y, t (torch.Tensor): Coordinates, shape (N, 1).
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none" for
                                  the clean baseline.
        epsilon (float): Perturbation strength (ignored if "none").
        no_grad (bool): If True, wraps everything in torch.no_grad() (use
                         for Sbc/SE, which are value comparisons); if
                         False, keeps the graph alive (required for Smom/
                         Sdiv, which need x, y, t to already have
                         requires_grad=True).

    Outputs:
        dict: {"u", "v", "p"}, each (N, 1).
    """
    def _run():
        coords_tensor = torch.cat([x, y, t], dim=1)
        preds = model(coords_tensor)
        clean = {"u": preds[:, 0:1], "v": preds[:, 1:2], "p": preds[:, 2:3]}
        if perturbation_name == "none":
            return clean
        coords = {"x": x, "y": y, "t": t}
        model_arg = model if perturbation_name == "temporal_mismatch" else None
        return apply_perturbation(perturbation_name, clean, coords, params, epsilon,
                                   model=model_arg, no_grad=no_grad)

    if no_grad:
        with torch.no_grad():
            return _run()
    return _run()


def compute_momentum_divergence_violation(model, T: float, params: dict, perturbation_name: str,
                                           epsilon: float, nu: float, scaler,
                                           n_interior: int = 20000, chunk_size: int = 8000,
                                           device: str = "cpu") -> tuple[float, float]:
    """
    Computes Smom = MSE(Ru) + MSE(Rv) and Sdiv = MSE(Rc) [Section 8] for one
    (perturbation, epsilon) field, on a fresh interior collocation sample
    (reusing src.data.point_samplers.sample_interior_points -- the SAME
    distribution training's own PDE loss term is evaluated on), processed
    in VRAM-safe chunks since Ru, Rv require second-order autograd. See the
    module docstring for why residuals are scaled before squaring.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): This case's final simulation time.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Perturbation strength.
        nu (float): Kinematic viscosity for this case.
        scaler (ResidualScaler): This case's residual scaler.
        n_interior (int): Total number of interior points to sample.
        chunk_size (int): Points per forward/backward pass (VRAM safety).
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        (Smom, Sdiv): both float.
    """
    interior = sample_interior_points(T, n_interior)

    sum_Ru2, sum_Rv2, sum_Rc2, n_done = 0.0, 0.0, 0.0, 0
    for i in range(0, n_interior, chunk_size):
        chunk = interior[i:i + chunk_size]
        x, y, t = _leaf(chunk, 0, device), _leaf(chunk, 1, device), _leaf(chunk, 2, device)

        field = _field_at(model, x, y, t, params, perturbation_name, epsilon, no_grad=False)
        R_u, R_v, R_c = compute_residuals(field["u"], field["v"], field["p"], x, y, t, nu)
        R_u_s, R_v_s, R_c_s = scaler.scale_residuals(R_u, R_v, R_c)

        sum_Ru2 += torch.sum(R_u_s ** 2).item()
        sum_Rv2 += torch.sum(R_v_s ** 2).item()
        sum_Rc2 += torch.sum(R_c_s ** 2).item()
        n_done += chunk.shape[0]

    Smom = (sum_Ru2 / n_done) + (sum_Rv2 / n_done)
    Sdiv = sum_Rc2 / n_done
    return Smom, Sdiv


def _boundary_pair_mismatch(model, side_a: torch.Tensor, side_b: torch.Tensor, params: dict,
                             perturbation_name: str, epsilon: float, U0: float, scale_p: float,
                             device: str) -> float:
    """
    Computes MSE(u_a - u_b) + MSE(v_a - v_b) + MSE(p_a - p_b) (each divided
    by U0, U0, and scale_p respectively before squaring -- see module
    docstring) between two paired boundary sides of a hallucinated field.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        side_a, side_b (torch.Tensor): Paired boundary coordinates from
            src.data.point_samplers.sample_periodic_boundaries, shape
            (n_points, 3) each, columns (x, y, t).
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Perturbation strength.
        U0 (float): This case's velocity scale (non-dimensionalizes u, v).
        scale_p (float): This case's pressure scale, U0^2 (non-
                          dimensionalizes p). Pass scaler.scale_p.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        float: The summed, non-dimensionalized MSE for this boundary pair.
    """
    with torch.no_grad():
        xa = side_a.to(device=device, dtype=torch.float64)
        xb = side_b.to(device=device, dtype=torch.float64)
        x_a, y_a, t_a = xa[:, 0:1], xa[:, 1:2], xa[:, 2:3]
        x_b, y_b, t_b = xb[:, 0:1], xb[:, 1:2], xb[:, 2:3]

        field_a = _field_at(model, x_a, y_a, t_a, params, perturbation_name, epsilon, no_grad=True)
        field_b = _field_at(model, x_b, y_b, t_b, params, perturbation_name, epsilon, no_grad=True)

        mse_u = torch.mean(((field_a["u"] - field_b["u"]) / U0) ** 2).item()
        mse_v = torch.mean(((field_a["v"] - field_b["v"]) / U0) ** 2).item()
        mse_p = torch.mean(((field_a["p"] - field_b["p"]) / scale_p) ** 2).item()
    return mse_u + mse_v + mse_p


def compute_boundary_violation(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                                U0: float, scale_p: float, n_bc_per_axis: int = 1000,
                                device: str = "cpu") -> float:
    """
    Computes Sbc = MSE(s|x=0 - s|x=2pi) + MSE(s|y=0 - s|y=2pi) [Section 8]
    for one (perturbation, epsilon) field, on paired periodic-boundary
    points (reusing src.data.point_samplers.sample_periodic_boundaries --
    the same sampler training's own BC loss term uses). No gradient
    tracking needed; this is a plain value comparison, not a PDE residual.
    See the module docstring for this metric's known blind spot re: the
    "boundary" perturbation type.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): This case's final simulation time.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Perturbation strength.
        U0 (float): This case's velocity scale.
        scale_p (float): This case's pressure scale (scaler.scale_p).
        n_bc_per_axis (int): Number of paired points sampled per boundary axis.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        float: Sbc.
    """
    bounds = sample_periodic_boundaries(T, n_bc_per_axis)
    x_left, x_right = bounds["x_bounds"]
    y_bottom, y_top = bounds["y_bounds"]

    Sbc_x = _boundary_pair_mismatch(model, x_left, x_right, params, perturbation_name, epsilon, U0, scale_p, device)
    Sbc_y = _boundary_pair_mismatch(model, y_bottom, y_top, params, perturbation_name, epsilon, U0, scale_p, device)
    return Sbc_x + Sbc_y


def compute_energy_violation(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                              U0: float, k: int, phi_x: float, phi_y: float, nu: float,
                              n_time: int = 20, energy_res: int = 32, device: str = "cpu") -> float:
    """
    Computes SE = MSE_t( E(t) - Ephys(t) ) [Section 8] for one
    (perturbation, epsilon) field, where E(t) = mean_{x,y}[(u^2+v^2)/2] is
    evaluated at n_time slices spanning [0, T] on an energy_res x energy_res
    spatial grid, and Ephys(t) is obtained by evaluating generate_tgv() on
    THE SAME grid (rather than trusting a hand-derived closed form), so any
    implicit grid-quadrature bias cancels between the measured and
    analytical curves.

    Kinetic energy is phase-invariant for the TGV solution (spatial
    averages of sin^2/cos^2 over a full period don't depend on the phase
    offset), so this remains a correct energy-decay reference even for the
    currently phase-mismatched trained models identified in Issue #9's
    audit -- any phi_x, phi_y works identically here; the case's own
    values are used for consistency with the rest of the pipeline.

    No gradient tracking needed; this is a value comparison, not a PDE residual.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): This case's final simulation time.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Perturbation strength.
        U0, k, phi_x, phi_y (float/int): This case's TGV parameters.
        nu (float): Kinematic viscosity for this case.
        n_time (int): Number of time slices spanning [0, T].
        energy_res (int): Spatial grid resolution per time slice
                           (energy_res x energy_res points).
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        float: SE.
    """
    x_lin = torch.linspace(0, 2 * torch.pi, energy_res, dtype=torch.float64)
    y_lin = torch.linspace(0, 2 * torch.pi, energy_res, dtype=torch.float64)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    x = X.reshape(-1, 1).to(device)
    y = Y.reshape(-1, 1).to(device)

    squared_errors = []
    for t_val in torch.linspace(0, T, n_time, dtype=torch.float64):
        t = torch.full_like(x, float(t_val))

        with torch.no_grad():
            field = _field_at(model, x, y, t, params, perturbation_name, epsilon, no_grad=True)
            E_meas = 0.5 * torch.mean(field["u"] ** 2 + field["v"] ** 2).item()

        u_a, v_a, _ = generate_tgv(x, y, t, U0, k, phi_x, phi_y, nu)
        E_phys = 0.5 * torch.mean(u_a ** 2 + v_a ** 2).item()

        squared_errors.append((E_meas - E_phys) ** 2)

    return float(np.mean(squared_errors))


def compute_phs_components(model, case_meta: dict, nu: float, T: float, scaler,
                            perturbation_name: str, epsilon: float,
                            n_interior: int = 20000, n_bc_per_axis: int = 1000,
                            n_time: int = 20, energy_res: int = 32, chunk_size: int = 8000,
                            device: str = "cpu") -> dict:
    """
    Computes all 4 raw PHS components for ONE (case, perturbation, epsilon)
    field. This is the single entry point evaluate_phs.py calls per row of
    the hallucination index.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN for this case.
        case_meta (dict): This case's metadata ("U0", "k", "phi_x", "phi_y", ...).
        nu (float): Kinematic viscosity for this case.
        T (float): This case's final simulation time.
        scaler (ResidualScaler): This case's residual scaler.
        perturbation_name (str): One of PERTURBATION_NAMES, or "none" for
                                  the clean baseline.
        epsilon (float): Perturbation strength (ignored if "none").
        n_interior, n_bc_per_axis, n_time, energy_res, chunk_size (int):
            Resolution/sampling knobs, forwarded to the 3 component
            functions above. Defaults match the project's other evaluation
            grids where a direct equivalent exists (64x64x20 total points
            informed n_interior/n_time; see evaluate_phs.py's parse_args).
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        dict: {"mom": Smom, "div": Sdiv, "bc": Sbc, "E": SE}, all float.
    """
    params = {"U0": case_meta["U0"], "k": case_meta["k"], "T": T}

    Smom, Sdiv = compute_momentum_divergence_violation(
        model, T, params, perturbation_name, epsilon, nu, scaler, n_interior, chunk_size, device,
    )
    Sbc = compute_boundary_violation(
        model, T, params, perturbation_name, epsilon, case_meta["U0"], scaler.scale_p, n_bc_per_axis, device,
    )
    SE = compute_energy_violation(
        model, T, params, perturbation_name, epsilon,
        case_meta["U0"], case_meta["k"], case_meta["phi_x"], case_meta["phi_y"], nu,
        n_time, energy_res, device,
    )
    return {"mom": Smom, "div": Sdiv, "bc": Sbc, "E": SE}


def compute_normalizers(valid_validation_rows: pd.DataFrame) -> dict:
    """
    Computes the 4 per-component normalizers: mean(Sj) over clean
    (label == "clean") fields from the VALIDATION split only [Section 8:
    "normalized using valid validation fields"].

    Inputs:
        valid_validation_rows (pd.DataFrame): Rows already filtered to
            split == "validation" and label == "clean", with columns
            "mom", "div", "bc", "E".

    Outputs:
        dict: {component_name: normalizer (float)}.
    """
    if len(valid_validation_rows) == 0:
        raise ValueError("No clean validation-split rows provided -- cannot compute normalizers.")
    return {c: float(valid_validation_rows[c].mean()) for c in PHS_COMPONENT_NAMES}


def normalize_components(df: pd.DataFrame, normalizers: dict, eps: float = 1e-12) -> pd.DataFrame:
    """
    Adds one "{component}_bar" column per PHS_COMPONENT_NAMES entry:
    S_bar_j = Sj / (mean(Sj_valid) + eps) [Section 8].

    Inputs:
        df (pd.DataFrame): Must contain raw "mom", "div", "bc", "E" columns.
        normalizers (dict): Output of compute_normalizers().
        eps (float): Numerical floor preventing division by zero.

    Outputs:
        pd.DataFrame: `df` with 4 new columns appended (copy, not in-place).
    """
    out = df.copy()
    for c in PHS_COMPONENT_NAMES:
        out[f"{c}_bar"] = out[c] / (normalizers[c] + eps)
    return out


def compute_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column per BASELINE_DEFINITIONS entry (Score1_momentum_only,
    Score2_momentum_divergence, Score3_PHS_full): the sum of that
    baseline's normalized ("_bar") components.

    Inputs:
        df (pd.DataFrame): Must already have the 4 "*_bar" columns from
                            normalize_components().

    Outputs:
        pd.DataFrame: `df` with 3 new score columns appended (copy).
    """
    out = df.copy()
    for score_name, components in BASELINE_DEFINITIONS.items():
        out[score_name] = sum(out[f"{c}_bar"] for c in components)
    return out


def select_threshold(validation_clean_scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Selects tau = percentile95(score over clean VALIDATION-split fields)
    [Section 8: "Select the threshold tau from validation fields"].

    Inputs:
        validation_clean_scores (array-like): One score's values over the
            clean, validation-split fields only.
        percentile (float): Which percentile to use. Defaults to 95, per
                             Section 8.

    Outputs:
        float: tau.
    """
    return float(np.percentile(np.asarray(validation_clean_scores), percentile))