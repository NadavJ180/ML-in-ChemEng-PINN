"""
Physical Hallucination Score (PHS): the four raw components and the scoring.

    Smom = MSE(Ru) + MSE(Rv)          momentum residual
    Sdiv = MSE(Rc)                    continuity residual
    Sbc  = near-edge / far-from-edge momentum residual (compute_boundary_localization_violation)
    SE   = MSE_t( E(t) - Ephys(t) )   kinetic energy against the exact Taylor-Green decay

The residuals are made dimensionless with ResidualScaler (src.models.scaling) before they
are squared, as in training. Then:

    S_bar_j = Sj / (normalizer_j + 1e-12)
    a score = a sum of S_bar_j (SCORE_DEFINITIONS)
    flagged  <=>  score > tau

The first iteration fitted the normalizers (means over clean fields) and tau (their 95th
percentile) on the clean validation fields, with the three scores of BASELINE_DEFINITIONS
(evaluate_phs.py). The final iteration adds two ablations (SCORE_DEFINITIONS) and freezes
the normalizers and tau on the 30 old clean fields (frozen_calibration.py, PROTOCOL.md).

Each function computes one thing for one field; looping over cases, damage types and
strengths is left to the callers. Why the components are defined this way: DESIGN.md,
section 5.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from src.physics.navier_stokes import compute_residuals
from src.physics.taylor_green import generate_tgv, compute_decay_timescale
from src.data.point_samplers import sample_interior_points
from src.hallucinations.perturbations import apply_perturbation

# The 4 raw PHS components. "bc" is computed via compute_boundary_localization_violation
# (a near/far residual ratio), NOT a literal periodicity comparison -- see this
# module's docstring ("WHY Sbc IS A NEAR/FAR RATIO") for why, and the README's Findings
# section for the fuller investigation.
PHS_COMPONENT_NAMES = ["mom", "div", "bc", "E"]

# 2 residual-only baselines PHS is compared against, plus PHS itself as Score3.
BASELINE_DEFINITIONS = {
    "Score1_momentum_only": ["mom"],
    "Score2_momentum_divergence": ["mom", "div"],
    "Score3_PHS_full": ["mom", "div", "bc", "E"],
}

# The five scores of the final iteration (#45): the three above, plus PHS without its boundary
# term and PHS without its energy term. Kept apart from BASELINE_DEFINITIONS so the default
# evaluate_phs.py run keeps writing exactly the three old scores.
SCORE_DEFINITIONS = {
    **BASELINE_DEFINITIONS,
    "Score4_PHS_no_bc": ["mom", "div", "E"],
    "Score5_PHS_no_E": ["mom", "div", "bc"],
}


@contextlib.contextmanager
def _seeded(seed: int | None):
    """
    Context manager: if seed is not None, sets torch's global random generator to seed for
    the duration of the block, and restores its earlier state on exit. With seed None it does
    nothing.

    The samplers of src.data.point_samplers draw from the global generator, so this is how
    every field of a case (clean and damaged) is scored at the same points (DESIGN.md,
    section 5).
    """
    if seed is None:
        yield
        return
    prior_state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(prior_state)


def _leaf(coords_tensor: torch.Tensor, col: int, device: str) -> torch.Tensor:
    """
    Returns one column of an (N, 3) coordinate tensor as its own float64 leaf tensor with
    requires_grad=True, so that autograd gives separate derivatives with respect to x, y
    and t.

    Inputs:
        coords_tensor (torch.Tensor): Shape (N, 3), columns (x, y, t).
        col (int): The column (0 = x, 1 = y, 2 = t).
        device (str): 'cuda' or 'cpu'.

    Outputs:
        torch.Tensor: Shape (N, 1), float64, requires_grad=True, on `device`.
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
                                           device: str = "cpu", seed: int = None) -> tuple[float, float]:
    """
    Computes Smom = MSE(Ru) + MSE(Rv) and Sdiv = MSE(Rc) for one
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
        seed (int | None): If provided, all sampled points are drawn
            reproducibly from this seed via _seeded() -- pass the SAME
            seed (e.g. derived from case_id) across a case's clean field
            and every one of its perturbed variants so they share
            identical (x, y, t) points (see _seeded's docstring).

    Outputs:
        (Smom, Sdiv): both float.
    """
    with _seeded(seed):
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


def compute_boundary_localization_violation(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                                             nu: float, scaler, band_width_fraction: float = 0.05,
                                             n_points: int = 20000, chunk_size: int = 8000,
                                             device: str = "cpu", seed: int = None) -> tuple[float, int]:
    """
    Sbc of one field: the mean squared scaled momentum residual at the points near an edge
    (within band_width_fraction * 2*pi of any of the four edges), divided by the same at the
    other points. Both are computed on the same field, from one sample of n_points points.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): The case's final time.
        params (dict): "U0", "k", "T" and "tau_decay".
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Damage strength.
        nu (float): Kinematic viscosity of the case.
        scaler (ResidualScaler): The case's residual scaler.
        band_width_fraction (float): Width of the near-edge band, as a fraction of 2*pi.
        n_points (int): Interior points sampled before the near/far split.
        chunk_size (int): Points per forward/backward pass.
        device (str): 'cuda' or 'cpu'.
        seed (int | None): See _seeded.

    Outputs:
        (Sbc, n_near): the ratio (float), and how many points fell in the near band (int).
                       Sbc is 0.0 if either band is empty.
    """
    two_pi = 2 * np.pi
    band_width = band_width_fraction * two_pi
    with _seeded(seed):
        interior = sample_interior_points(T, n_points)
    x_all, y_all = interior[:, 0], interior[:, 1]
    near_mask = ((x_all < band_width) | (x_all > two_pi - band_width) |
                 (y_all < band_width) | (y_all > two_pi - band_width))
    near_points = interior[near_mask]
    far_points = interior[~near_mask]
    n_near = near_points.shape[0]
    n_far = far_points.shape[0]
    if n_near == 0 or n_far == 0:
        return 0.0, n_near

    def _mean_scaled_residual_sq(points):
        sum_Ru2, sum_Rv2, n_done = 0.0, 0.0, 0
        for i in range(0, points.shape[0], chunk_size):
            chunk = points[i:i + chunk_size]
            x, y, t = _leaf(chunk, 0, device), _leaf(chunk, 1, device), _leaf(chunk, 2, device)

            field = _field_at(model, x, y, t, params, perturbation_name, epsilon, no_grad=False)
            R_u, R_v, R_c = compute_residuals(field["u"], field["v"], field["p"], x, y, t, nu)
            R_u_s, R_v_s, _ = scaler.scale_residuals(R_u, R_v, R_c)  # R_c's scaled form is unused here

            sum_Ru2 += torch.sum(R_u_s ** 2).item()
            sum_Rv2 += torch.sum(R_v_s ** 2).item()
            n_done += chunk.shape[0]
        return (sum_Ru2 / n_done) + (sum_Rv2 / n_done)

    near_residual = _mean_scaled_residual_sq(near_points)
    far_residual = _mean_scaled_residual_sq(far_points)
    Sbc = near_residual / (far_residual + 1e-15)
    return Sbc, n_near


def compute_energy_violation(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                              U0: float, k: int, phi_x: float, phi_y: float, nu: float,
                              n_time: int = 20, energy_res: int = 32, device: str = "cpu") -> float:
    """
    SE of one field: the mean over n_time times in [0, T] of (E(t) - Ephys(t))^2, where
    E(t) = mean over an energy_res x energy_res grid of (u^2 + v^2) / 2, and Ephys(t) is the
    same average of the exact solution (generate_tgv) on the same grid. No gradients are
    needed.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): The case's final time.
        params (dict): "U0", "k", "T" and "tau_decay".
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Damage strength.
        U0, k, phi_x, phi_y (float/int): The case's Taylor-Green parameters.
        nu (float): Kinematic viscosity of the case.
        n_time (int): Number of times in [0, T].
        energy_res (int): Grid points per side at each time.
        device (str): 'cuda' or 'cpu'.

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


def compute_relative_error(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                            n_points: int = 5000, seed: int = None, device: str = "cpu") -> float:
    """
    The relative L2 size of the change a damage makes to the whole field:
    sqrt(sum (du^2 + dv^2 + dp^2) / sum (u^2 + v^2 + p^2)) over n_points interior points, with
    u and v divided by U0 and p by U0^2. Pressure is included because the pressure damage
    changes no velocity. Not a PHS component; used by detection_sensitivity.py.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): The case's final time.
        params (dict): "U0", "k", "T" (and "tau_decay" for "temporal_mismatch").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none" (returns 0.0).
        epsilon (float): Damage strength.
        n_points (int): Interior points to sample.
        seed (int | None): See _seeded.
        device (str): 'cuda' or 'cpu'.

    Outputs:
        float: The relative change, or inf if the clean field is zero at every point.
    """
    if perturbation_name == "none":
        return 0.0

    U0 = params["U0"]
    scale_p = U0 ** 2

    with _seeded(seed):
        interior = sample_interior_points(T, n_points)
    x = interior[:, 0:1].to(device=device, dtype=torch.float64)
    y = interior[:, 1:2].to(device=device, dtype=torch.float64)
    t = interior[:, 2:3].to(device=device, dtype=torch.float64)

    with torch.no_grad():
        clean_field = _field_at(model, x, y, t, params, "none", 0.0, no_grad=True)
        perturbed_field = _field_at(model, x, y, t, params, perturbation_name, epsilon, no_grad=True)

        du = (perturbed_field["u"] - clean_field["u"]) / U0
        dv = (perturbed_field["v"] - clean_field["v"]) / U0
        dp = (perturbed_field["p"] - clean_field["p"]) / scale_p
        num = torch.sum(du ** 2 + dv ** 2 + dp ** 2)

        u_ref = clean_field["u"] / U0
        v_ref = clean_field["v"] / U0
        p_ref = clean_field["p"] / scale_p
        den = torch.sum(u_ref ** 2 + v_ref ** 2 + p_ref ** 2)

        rel_l2 = torch.sqrt(num / den).item() if den.item() > 0 else float("inf")
    return rel_l2


def compute_phs_components(model, case_meta: dict, nu: float, T: float, scaler,
                            perturbation_name: str, epsilon: float,
                            n_interior: int = 20000, n_bc_per_axis: int = 1000,
                            n_time: int = 20, energy_res: int = 32, chunk_size: int = 8000,
                            device: str = "cpu", bc_local_band_width_fraction: float = 0.05,
                            bc_local_n_points: int = 20000, seed: int = None) -> dict:
    """
    The four raw components of one (case, damage type, strength) field. This is the function
    evaluate_phs.py and natural_errors.py call for every field.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN of the case.
        case_meta (dict): The case's entry ("U0", "k", "phi_x", "phi_y", ...).
        nu (float): Kinematic viscosity of the case.
        T (float): The case's final time.
        scaler (ResidualScaler): The case's residual scaler.
        perturbation_name (str): One of PERTURBATION_NAMES, or "none" for the clean field.
        epsilon (float): Damage strength (ignored for "none").
        n_interior (int): Points for Smom and Sdiv.
        n_bc_per_axis (int): Not used; kept so evaluate_phs.py's --n_bc still passes through.
        n_time, energy_res (int): Times and grid size for SE.
        chunk_size (int): Points per forward/backward pass.
        device (str): 'cuda' or 'cpu'.
        bc_local_band_width_fraction, bc_local_n_points: For Sbc
            (compute_boundary_localization_violation).
        seed (int | None): Passed to every component; pass the same seed (from the case
            number) for a case's clean field and all its damaged ones.

    Outputs:
        dict: {"mom": Smom, "div": Sdiv, "bc": Sbc, "E": SE}, floats.
    """
    params = {"U0": case_meta["U0"], "k": case_meta["k"], "T": T,
              "tau_decay": compute_decay_timescale(nu, case_meta["k"])}

    Smom, Sdiv = compute_momentum_divergence_violation(
        model, T, params, perturbation_name, epsilon, nu, scaler, n_interior, chunk_size, device, seed,
    )
    Sbc, _ = compute_boundary_localization_violation(
        model, T, params, perturbation_name, epsilon, nu, scaler,
        bc_local_band_width_fraction, bc_local_n_points, chunk_size, device, seed,
    )
    SE = compute_energy_violation(
        model, T, params, perturbation_name, epsilon,
        case_meta["U0"], case_meta["k"], case_meta["phi_x"], case_meta["phi_y"], nu,
        n_time, energy_res, device,
    )
    return {"mom": Smom, "div": Sdiv, "bc": Sbc, "E": SE}


def compute_normalizers(valid_validation_rows: pd.DataFrame, component_names: list = None) -> dict:
    """
    Computes the per-component normalizers: mean(Sj) over clean
    (label == "clean") fields from the VALIDATION split only.

    Inputs:
        valid_validation_rows (pd.DataFrame): Rows already filtered to
            split == "validation" and label == "clean", with a column for
            each name in component_names.
        component_names (list[str] | None): Which raw columns to
            normalize. Defaults to PHS_COMPONENT_NAMES (all 4); pass a
            narrower list only for a specific ablation.

    Outputs:
        dict: {component_name: normalizer (float)}.
    """
    component_names = component_names or PHS_COMPONENT_NAMES
    if len(valid_validation_rows) == 0:
        raise ValueError("No clean validation-split rows provided -- cannot compute normalizers.")
    return {c: float(valid_validation_rows[c].mean()) for c in component_names}


def normalize_components(df: pd.DataFrame, normalizers: dict, eps: float = 1e-12,
                          component_names: list = None) -> pd.DataFrame:
    """
    Adds one "{component}_bar" column per requested component:
    S_bar_j = Sj / (mean(Sj_valid) + eps).

    Inputs:
        df (pd.DataFrame): Must contain a raw column for each name in
            component_names.
        normalizers (dict): Output of compute_normalizers().
        eps (float): Numerical floor preventing division by zero.
        component_names (list[str] | None): Defaults to PHS_COMPONENT_NAMES (all 4).

    Outputs:
        pd.DataFrame: `df` with new columns appended (copy, not in-place).
    """
    component_names = component_names or PHS_COMPONENT_NAMES
    out = df.copy()
    for c in component_names:
        out[f"{c}_bar"] = out[c] / (normalizers[c] + eps)
    return out


def compute_baseline_scores(df: pd.DataFrame, baseline_definitions: dict = None) -> pd.DataFrame:
    """
    Adds one column per baseline_definitions entry: the sum of that
    baseline's normalized ("_bar") components.

    Inputs:
        df (pd.DataFrame): Must already have the relevant "*_bar" columns
                            from normalize_components().
        baseline_definitions (dict | None): Defaults to BASELINE_DEFINITIONS
            (Score1/Score2/Score3 -- Score3_PHS_full is the official PHS).

    Outputs:
        pd.DataFrame: `df` with new score columns appended (copy).
    """
    baseline_definitions = baseline_definitions or BASELINE_DEFINITIONS
    out = df.copy()
    for score_name, components in baseline_definitions.items():
        out[score_name] = sum(out[f"{c}_bar"] for c in components)
    return out


def select_threshold(validation_clean_scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Selects tau = percentile95(score over clean VALIDATION-split fields).

    Inputs:
        validation_clean_scores (array-like): One score's values over the
            clean, validation-split fields only.
        percentile (float): Which percentile to use. Defaults to 95.

    Outputs:
        float: tau.
    """
    return float(np.percentile(np.asarray(validation_clean_scores), percentile))