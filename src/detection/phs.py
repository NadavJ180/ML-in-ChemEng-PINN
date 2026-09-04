"""
Physical Hallucination Score (PHS) -- Core Formula Module

Implements the 4 raw violation components and the normalization/scoring
pipeline this project uses:

    Smom = MSE(Ru) + MSE(Rv)                          (momentum violation)
    Sdiv = MSE(ux + vy)                                (divergence violation,
                                                         i.e. MSE(Rc))
    Sbc  = near-boundary / far-from-boundary ratio of  (boundary violation --
           the scaled momentum residual (see            see "WHY Sbc IS A
           compute_boundary_localization_violation)      NEAR/FAR RATIO" below)
    E(t)     = mean_{x,y}[ (u^2 + v^2) / 2 ]            (kinetic energy)
    Ephys(t) = E(0) * exp(-4 * nu * k^2 * t)            (TGV's analytical
                                                         decay)
    SE   = MSE_t( E(t) - Ephys(t) )                     (energy violation)

    S_bar_j = Sj / (mean(Sj over clean VALIDATION-split fields) + 1e-12)
    PHS     = S_bar_mom + S_bar_div + S_bar_bc + S_bar_E
    tau     = percentile95(PHS over clean VALIDATION-split fields)
    hallucinated  <=>  PHS > tau

This module also defines 2 residual-only baselines PHS is benchmarked
against (see BASELINE_DEFINITIONS below):
    Score1_momentum_only       = S_bar_mom
    Score2_momentum_divergence = S_bar_mom + S_bar_div
    Score3_PHS_full            = S_bar_mom + S_bar_div + S_bar_bc + S_bar_E
                                  (= PHS, the official, currently-adopted score)
Score1/Score2 could be written in terms of the raw Smom/Sdiv, but every
score here is built from the SAME normalized components PHS uses --
otherwise Score1/Score2 would inherit the raw cross-case scale problem
described below, making the baseline comparison meaningless. This module
always operates on S_bar_j.

WHY Sbc IS A NEAR/FAR RATIO, NOT A LITERAL PERIODICITY COMPARISON:
the "boundary" perturbation's m(x) = exp(-x^2/sigma^2) + exp(-(2*pi-x)^2/sigma^2)
satisfies m(0) == m(2*pi) by construction, so a check that only compares
VALUES at the boundary pair (s|x=0 vs s|x=2*pi, or any derivative of it) is
structurally blind to this perturbation type -- this follows for every
derivative order, not just the raw value, since m(x) = f(x) + f(2*pi-x)
for a smooth f, and every derivative of that sum matches at x=0 vs x=2*pi
by the same symmetry. A spatially-LOCALIZED check catches it instead: does
the field behave anomalously in a neighborhood of the edges, not just
exactly on them (see compute_boundary_localization_violation below). This
also turned out to be necessary for the OTHER 4 perturbation types too,
not just "boundary" -- checked directly, a literal periodicity-comparison
Sbc was essentially flat (~3.3e-7) and indistinguishable from its own
clean baseline across every perturbation type in this benchmark, because
the other perturbations are all built from integer-frequency trig
functions that are exactly periodic on [0, 2*pi] by construction.
Confirmed to work: the near/far ratio computed here climbs ~170x across
the epsilon sweep for "boundary" specifically, while the original
literal-Sbc formula stayed flat. See the README's Findings section for
the fuller investigation and history of this replacement.

TWO DELIBERATE NON-DIMENSIONALIZATION CHOICES (documented again at their
point of use below):

  1. Smom, Sdiv, and Sbc are all computed from residuals that are FIRST
     non-dimensionalized via ResidualScaler (src.models.scaling), exactly as
     loss.py and verify_hallucinations.py already do -- not left in raw
     physical units. Without this, a case with a large sampled U0 would
     trivially have a much larger Smom/Sdiv/Sbc than a low-U0 case even with
     an equally well-trained model, which would corrupt the cross-case
     pooled normalization below (mean(Sj_valid) would be dominated by
     whichever validation cases happen to have large U0).

  2. Wavenumber k was found to correlate with a case's natural (clean-field)
     residual scale even after (1)'s scaling -- see the README's Findings
     section for the full mechanism (raw fitting error increases with k, but
     ResidualScaler's U0^2*k divisor over-corrects for it). Not fixed here;
     documented as a known source of calibration sensitivity to how a
     train/validation/test split happens to distribute k across its splits.

Every function below takes an already-loaded, eval-mode BaselinePINN and
computes one thing at a time, mirroring the split used throughout
src/hallucinations/verify_hallucinations.py (small, single-purpose,
independently testable functions; looping over cases/perturbations/epsilons
is left to the calling script, evaluate_phs.py).
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


@contextlib.contextmanager
def _seeded(seed: int | None):
    """
    Context manager: if seed is not None, temporarily sets torch's GLOBAL
    RNG to `seed` for the duration of the `with` block, restoring whatever
    state it had beforehand on exit. If seed is None, does nothing (the
    original, unseeded behavior).

    WHY THIS EXISTS: sample_interior_points (src.data.point_samplers) draws
    from torch's global RNG with no seed
    argument of their own, so two separate calls -- e.g. one for a case's
    clean field and one for the same case's epsilon=0.1 "momentum" variant
    -- land on completely different random (x, y, t) points. That's an
    unnecessary extra source of noise on top of genuine small-sample
    calibration noise: comparing "did the residual go up" between two
    DIFFERENT point clouds is noisier than comparing it at the SAME
    points. evaluate_phs.py passes the SAME seed (derived from case_id
    alone, not perturbation/epsilon) into every field of a given case, so
    all of that case's fields -- clean and every hallucinated variant --
    are evaluated at identical coordinates. Restoring the prior RNG state
    on exit keeps this fully side-effect-free for anything else in the
    process that relies on unseeded randomness.
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
    Sbc, as a RATIO of near-boundary to far-from-boundary momentum-residual
    magnitude, both computed on the SAME field (whichever field this is
    called on -- clean or a specific perturbed variant), not compared
    against a separately-stored clean baseline. That clean-baseline
    normalization still happens afterward, in the usual pipeline (see
    normalize_components) -- exactly like every other component; this
    function's own job is to produce one raw number per field, and a ratio
    computed WITHIN that field is what gives it the property below.

    WHY A RATIO, NOT THE ABSOLUTE NEAR-BOUNDARY VALUE (an earlier version of
    this function computed only the numerator): checked directly, that
    absolute-value version was found to be highly correlated with Smom for
    globally-applied perturbations (momentum, pressure, velocity_divergence,
    temporal_mismatch all raise residual roughly everywhere, including near
    the edges, so the near-boundary-only value rises right along with Smom,
    contributing a mostly-redundant copy of the same signal to PHS's sum --
    confirmed empirically: a hallucination-uniformly-applied perturbation
    inflated both by similar relative amounts). A RATIO cancels that: if a
    perturbation raises residual UNIFORMLY (near and far rise together),
    the ratio stays close to whatever it was for a clean field, regardless
    of how large that uniform rise is. If a perturbation concentrates near
    the edges specifically (like "boundary"), only the numerator rises,
    so the ratio spikes. This directly restores the original design intent
    from the boundary_localization_ratio diagnostic in
    verify_hallucinations.py (which this was adapted from, and which
    already used a near/far ratio) -- the ratio was lost when this was
    first adapted for PHS scoring, keeping only the numerator; this
    restores it.

    This REPLACES a literal periodicity/value comparison (s|x=0 vs s|x=2pi)
    entirely -- it is not an addition alongside it.
    Checked directly and found the original formula was essentially blind
    to EVERY perturbation type in this benchmark (not just "boundary"):
    velocity_divergence/momentum/pressure all add perturbation terms built
    from integer-frequency trig functions, which are exactly periodic on
    [0, 2*pi] by construction, so they can never move s(0) away from
    s(2*pi) regardless of epsilon. That is a property of how the
    perturbations happen to be built, not a flaw in checking periodicity
    per se (periodicity genuinely is the correct constraint here, for any
    field regardless of internal symmetry) -- but the practical result is
    the same either way: the original formula contributed no discriminative
    signal against this benchmark's actual perturbations, so it was
    removed rather than kept alongside a working replacement.

    Cost note: computing BOTH the near-band and far-band residual (rather
    than only the near band, as the previous version did) means this now
    needs the same expensive double-backward residual computation on
    close to the FULL n_points sample, not just the ~19% falling in the
    near-boundary band at the default 5% band_width_fraction. This roughly
    doubles this component's own cost versus the previous, numerator-only
    version (see the README's Findings section on why evaluate_phs.py
    takes the time it does).

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): This case's final simulation time.
        params (dict): Case-specific physical constants ("U0", "k", "T").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none".
        epsilon (float): Perturbation strength.
        nu (float): Kinematic viscosity for this case.
        scaler (ResidualScaler): This case's residual scaler.
        band_width_fraction (float): Distance from an edge (in x or y),
            as a FRACTION of the domain length (2*pi), counted as
            "near-boundary". Default 0.05 (5%).
        n_points (int): Total interior points sampled before the
            near/far split; with band_width_fraction=0.05, roughly a
            fifth of uniformly sampled points fall in the near band, the
            rest in the far band.
        chunk_size (int): Points per forward/backward pass (VRAM safety).
        device (str): Target hardware device ('cuda' or 'cpu').
        seed (int | None): See compute_momentum_divergence_violation's docstring.

    Outputs:
        (Sbc, n_near): Sbc (float, the near/far residual ratio -- this IS
            "bc" in PHS_COMPONENT_NAMES / compute_phs_components' output),
            n_near (int, how many sampled points fell in the near band --
            callers can use this to sanity-check band_width_fraction isn't
            too narrow/wide; not needed for scoring itself).
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
    Computes SE = MSE_t( E(t) - Ephys(t) ) for one
    (perturbation, epsilon) field, where E(t) = mean_{x,y}[(u^2+v^2)/2] is
    evaluated at n_time slices spanning [0, T] on an energy_res x energy_res
    spatial grid, and Ephys(t) is obtained by evaluating generate_tgv() on
    THE SAME grid (rather than trusting a hand-derived closed form), so any
    implicit grid-quadrature bias cancels between the measured and
    analytical curves.

    Kinetic energy is phase-invariant for the TGV solution (spatial
    averages of sin^2/cos^2 over a full period don't depend on the phase
    offset), so this remains a correct energy-decay reference even for the
    currently phase-mismatched trained models identified in the
    hallucination-verification audit -- any phi_x, phi_y works identically here; the case's own
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


def compute_relative_error(model, T: float, params: dict, perturbation_name: str, epsilon: float,
                            n_points: int = 5000, seed: int = None, device: str = "cpu") -> float:
    """
    Computes the relative L2 magnitude of the change a perturbation makes
    to the FULL field (u, v, p) -- non-dimensionalized consistently with
    the rest of this module (u, v divided by U0; p divided by scale_p =
    U0^2, the same convention ResidualScaler uses) before
    pooling, so the three quantities contribute on a comparable footing
    rather than whichever has the largest raw magnitude dominating the sum.

    DELIBERATELY DIFFERENT FROM verify_hallucinations.py's visual_deviation,
    which compares (u, v) ONLY -- that metric's purpose is "does this look
    different in a velocity contour plot," matching what its own plots
    actually show. This function's purpose is different: "how
    much did the full field change, in a way that's comparable across
    every perturbation type" -- and "pressure" only modifies p, leaving
    (u, v) completely untouched, so a velocity-only version of this metric
    would be EXACTLY 0 for every epsilon of that one perturbation type
    (confirmed: this was tried first and broke the log-space interpolation
    in detection_sensitivity.py's find_boundary_crossings with a literal
    zero). Including p fixes that and is arguably the more honest
    "how much did this change" measure for this function's purpose anyway.

    Unlike Smom/Sdiv/Sbc/SE, this needs no gradients (it's a plain value
    comparison), and it is NOT one of PHS's 4 official
    components -- it exists purely to express "how much does epsilon
    actually change the field" as a single, physically interpretable
    number that's comparable ACROSS perturbation types. Epsilon itself is
    not directly comparable between perturbation types (a fraction of U0
    for "momentum", a fraction of a domain-edge bump amplitude for
    "boundary", a fraction of a decay timescale for "temporal_mismatch"),
    so "at epsilon=0.002 detection starts to fail" is a claim specific to
    whichever perturbation type generated it, while "at ~X% relative
    error detection starts to fail" is not.

    Inputs:
        model (nn.Module): The trained, eval-mode BaselinePINN.
        T (float): This case's final simulation time.
        params (dict): Case-specific physical constants; requires "U0",
                        "k", "T" (and "tau_decay" if perturbation_name is
                        "temporal_mismatch").
        perturbation_name (str): One of PERTURBATION_NAMES, or "none"
                                  (returns 0.0 trivially).
        epsilon (float): Perturbation strength.
        n_points (int): Interior points to sample.
        seed (int | None): Passed to _seeded() for reproducibility (see
            its docstring) -- pass the same per-case seed used elsewhere
            for this case to compare against the identical point cloud;
            None leaves sampling unseeded.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        float: Relative L2 error, or float("inf") if the clean field has
        zero norm on the sampled points (a degenerate case not expected
        in practice).
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
    Computes the 4 raw PHS components for ONE (case, perturbation, epsilon)
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
        n_interior, n_time, energy_res, chunk_size (int):
            Resolution/sampling knobs, forwarded to the component functions
            above. Defaults match the project's other evaluation grids
            where a direct equivalent exists (64x64x20 total points
            informed n_interior/n_time; see evaluate_phs.py's parse_args).
        n_bc_per_axis (int): Currently UNUSED here -- kept in the signature
            for call-site compatibility (evaluate_phs.py's --n_bc flag still
            forwards to it), since "bc" is now computed entirely via
            compute_boundary_localization_violation's near/far ratio, which
            uses n_points instead. The original boundary-pair sampling this
            parameter fed has been deleted (see this module's docstring).
        device (str): Target hardware device ('cuda' or 'cpu').
        bc_local_band_width_fraction, bc_local_n_points: Forwarded to
            compute_boundary_localization_violation (see its docstring),
            which is what "bc" is now computed from.
        seed (int | None): If provided, passed through to every component
            function so this field's random points are reproducible. Pass
            the SAME seed (derived from case_id, not perturbation/epsilon)
            for a case's clean field and all its perturbed variants so
            they are compared at identical (x, y, t) points -- see
            _seeded's docstring for why this matters.

    Outputs:
        dict: {"mom": Smom, "div": Sdiv, "bc": Sbc (computed via the
               boundary-LOCALIZED method -- see this module's docstring
               for why the original periodicity-comparison Sbc was
               replaced, not kept alongside), "E": SE}, all float.
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