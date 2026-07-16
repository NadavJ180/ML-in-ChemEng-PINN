"""
Physical Hallucination Perturbation Engine (Section 7)

Implements the 5 mathematical perturbation functions used to inject visually
plausible, physically inconsistent hallucinations into valid PINN-generated
flow fields (u, v, p).

Each perturbation function has the identical signature pattern:

    perturb_<name>(fields, coords, params, epsilon, model=None) -> dict

- `fields`  : dict with keys "u", "v", "p" holding the CLEAN model predictions
              on the evaluation grid (torch.Tensor, shape (N, 1) each).
- `coords`  : dict with keys "x", "y", "t" holding the evaluation grid
              coordinates (torch.Tensor, shape (N, 1) each, requires_grad not needed).
- `params`  : dict with keys "U0", "k", "T" (case-specific physical constants).
- `epsilon` : perturbation strength (float).
- `model`   : the trained BaselinePINN, only required for the temporal
              mismatch perturbation (it must re-query the network at a
              shifted time).

Every function returns a NEW dict {"u": ..., "v": ..., "p": ...} — the clean
`fields` dict passed in is never mutated in place, so the same baseline can be
reused across all epsilon values and perturbation types.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Registry populated at the bottom of this file so callers can do:
#     from src.hallucinations.perturbations import PERTURBATION_REGISTRY
# ---------------------------------------------------------------------------
PERTURBATION_REGISTRY = {}


def register(name):
    """
    Decorator factory that registers a perturbation function under a public
    string name, so it can be looked up later via apply_perturbation() or
    iterated over via PERTURBATION_REGISTRY / PERTURBATION_NAMES.

    Args:
        name (str): The public name the perturbation will be registered under
                     (e.g. "momentum", "boundary").

    Returns:
        Callable: A decorator that, when applied to a perturbation function,
                  stores that function in PERTURBATION_REGISTRY[name] and
                  returns the function unchanged.
    """
    def _wrap(fn):
        PERTURBATION_REGISTRY[name] = fn
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# 1. Velocity-Divergence Perturbation
#    ũ = u + ε U0 sin(3x + 0.7) sin(2y)
#    (v, p left untouched -> deliberately injects a non-zero divergence,
#     since only the u-component of an otherwise divergence-free field changes)
# ---------------------------------------------------------------------------
@register("velocity_divergence")
def perturb_velocity_divergence(fields, coords, params, epsilon, model=None, **kwargs):
    """
    Injects a divergence-causing hallucination by perturbing only the
    u-component of an otherwise clean, (approximately) divergence-free
    velocity field: ũ = u + ε U0 sin(3x + 0.7) sin(2y). Leaving v and p
    untouched means continuity (u_x + v_y = 0) is deliberately violated.

    Args:
        fields (dict): Clean predictions with keys "u", "v", "p", each a
                        torch.Tensor of shape (N, 1).
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; only "U0" is used here.
        epsilon (float): Perturbation strength.
        model: Unused for this perturbation; accepted only to keep a uniform
               function signature across all registered perturbations.
        **kwargs: Accepted and ignored, so all registered perturbations share
                   a uniform call signature via apply_perturbation().

    Returns:
        dict: {"u": hallucinated u (N, 1), "v": clean v (N, 1) unchanged,
               "p": clean p (N, 1) unchanged}.
    """
    U0 = params["U0"]
    x, y = coords["x"], coords["y"]

    u_tilde = fields["u"] + epsilon * U0 * torch.sin(3 * x + 0.7) * torch.sin(2 * y)

    return {
        "u": u_tilde,
        "v": fields["v"].clone(),
        "p": fields["p"].clone(),
    }


# ---------------------------------------------------------------------------
# 2. Momentum Perturbation
#    ũ = u + ε U0 sin(4x) cos(3y) cos(2t)
#    ṽ = v + ε U0 cos(2x) sin(5y) sin(t)
# ---------------------------------------------------------------------------
@register("momentum")
def perturb_momentum(fields, coords, params, epsilon, model=None, **kwargs):
    """
    Injects a momentum-conservation hallucination by adding independent,
    space-and-time-varying noise to both velocity components:
    ũ = u + ε U0 sin(4x) cos(3y) cos(2t), ṽ = v + ε U0 cos(2x) sin(5y) sin(t).
    This is designed to unbalance the x- and y-momentum PDE residuals while
    still looking like a plausible flow field. Pressure is left unchanged.

    Args:
        fields (dict): Clean predictions with keys "u", "v", "p", each a
                        torch.Tensor of shape (N, 1).
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; only "U0" is used here.
        epsilon (float): Perturbation strength.
        model: Unused for this perturbation; accepted only to keep a uniform
               function signature across all registered perturbations.
        **kwargs: Accepted and ignored, so all registered perturbations share
                   a uniform call signature via apply_perturbation().

    Returns:
        dict: {"u": hallucinated u (N, 1), "v": hallucinated v (N, 1),
               "p": clean p (N, 1) unchanged}.
    """
    U0 = params["U0"]
    x, y, t = coords["x"], coords["y"], coords["t"]

    u_tilde = fields["u"] + epsilon * U0 * torch.sin(4 * x) * torch.cos(3 * y) * torch.cos(2 * t)
    v_tilde = fields["v"] + epsilon * U0 * torch.cos(2 * x) * torch.sin(5 * y) * torch.sin(t)

    return {
        "u": u_tilde,
        "v": v_tilde,
        "p": fields["p"].clone(),
    }


# ---------------------------------------------------------------------------
# 3. Pressure Perturbation
#    p̃ = p + ε U0^2 cos(5x) cos(4y)
# ---------------------------------------------------------------------------
@register("pressure")
def perturb_pressure(fields, coords, params, epsilon, model=None, **kwargs):
    """
    Injects a pressure-field hallucination by adding a high-wavenumber
    spatial ripple to the clean pressure prediction:
    p̃ = p + ε U0^2 cos(5x) cos(4y). Velocity components are left unchanged,
    so this isolates inconsistencies between the perturbed pressure gradient
    and the (still-clean) momentum residuals.

    Args:
        fields (dict): Clean predictions with keys "u", "v", "p", each a
                        torch.Tensor of shape (N, 1).
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; only "U0" is used here.
        epsilon (float): Perturbation strength.
        model: Unused for this perturbation; accepted only to keep a uniform
               function signature across all registered perturbations.
        **kwargs: Accepted and ignored, so all registered perturbations share
                   a uniform call signature via apply_perturbation().

    Returns:
        dict: {"u": clean u (N, 1) unchanged, "v": clean v (N, 1) unchanged,
               "p": hallucinated p (N, 1)}.
    """
    U0 = params["U0"]
    x, y = coords["x"], coords["y"]

    p_tilde = fields["p"] + (epsilon * U0 ** 2) * torch.cos(5 * x) * torch.cos(4 * y)

    return {
        "u": fields["u"].clone(),
        "v": fields["v"].clone(),
        "p": p_tilde,
    }


# ---------------------------------------------------------------------------
# 4. Temporal Mismatch Perturbation
#    ũ(t) = u(t + εT), ṽ(t) = v(t + εT)
#    This one is NOT a value-space perturbation: it re-queries the trained
#    model at a shifted time and reports that as the "prediction" at the
#    original grid time t. Requires the model itself.
# ---------------------------------------------------------------------------
@register("temporal_mismatch")
def perturb_temporal_mismatch(fields, coords, params, epsilon, model=None, no_grad=True, **kwargs):
    """
    Injects a temporal-consistency hallucination by re-querying the trained
    model at a shifted time and reporting that as the prediction at the
    original grid time: ũ(t) = u(t + εT), ṽ(t) = v(t + εT). Unlike the other
    4 perturbations, this is NOT a value-space transformation of `fields` —
    it requires a second forward pass through `model`, since the shifted-time
    prediction cannot be derived from the clean (u, v) values alone.

    Args:
        fields (dict): Clean predictions with keys "u", "v", "p", each a
                        torch.Tensor of shape (N, 1). Only "p" is reused
                        here (returned unchanged); "u"/"v" are not used since
                        they are regenerated from `model` at the shifted time.
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; requires "T", the
                        case's final simulation time.
        epsilon (float): Perturbation strength; the time shift is epsilon * T.
        model (nn.Module): The trained PINN to re-query at the shifted time.
                            Required for this perturbation (raises otherwise).
        no_grad (bool): If True (default, used during dataset generation),
                        wraps the re-query in torch.no_grad() for speed/memory.
                        Set to False when the caller needs to backprop through
                        this perturbation (e.g. to compute PDE residuals of
                        the hallucinated field during verification) — in that
                        case `x`, `y`, `t` must already have requires_grad=True.
        **kwargs: Accepted and ignored, so all registered perturbations share
                   a uniform call signature via apply_perturbation().

    Returns:
        dict: {"u": model prediction at t + epsilon*T (N, 1),
               "v": model prediction at t + epsilon*T (N, 1),
               "p": clean p (N, 1) unchanged, since Section 7 only
               redefines (u, v) for this perturbation}.

    Raises:
        ValueError: If `model` is None, since the shifted-time query cannot
                    be computed without it.
    """
    if model is None:
        raise ValueError("perturb_temporal_mismatch requires the trained model to re-query at t + epsilon*T.")

    T = params["T"]
    x, y, t = coords["x"], coords["y"], coords["t"]

    t_shifted = t + epsilon * T
    shifted_coords = torch.cat([x, y, t_shifted], dim=1)

    if no_grad:
        with torch.no_grad():
            preds_shifted = model(shifted_coords)
    else:
        preds_shifted = model(shifted_coords)

    u_tilde = preds_shifted[:, 0:1]
    v_tilde = preds_shifted[:, 1:2]

    return {
        "u": u_tilde,
        "v": v_tilde,
        # Pressure is not part of the Section 7 spec for this perturbation;
        # keep the clean pressure prediction unchanged.
        "p": fields["p"].clone(),
    }


# ---------------------------------------------------------------------------
# 5. Boundary Perturbation
#    ũ = u + ε U0 m(x) sin(3y), where
#    m(x) = exp(-x^2/sigma^2) + exp(-(2π - x)^2/sigma^2), sigma = 0.2
#    (localizes the hallucination near the x=0 / x=2π periodic boundaries)
# ---------------------------------------------------------------------------
@register("boundary")
def perturb_boundary(fields, coords, params, epsilon, model=None, sigma: float = 0.2, **kwargs):
    """
    Injects a boundary-consistency hallucination by adding a localized bump
    to the u-component near the periodic x-boundaries (x=0 and x=2*pi):
    ũ = u + ε U0 m(x) sin(3y), where
    m(x) = exp(-x^2/sigma^2) + exp(-(2*pi - x)^2/sigma^2).
    Because m(x) decays to ~0 away from the boundaries, this perturbation is
    visually subtle in the domain interior but breaks periodic boundary
    consistency (u(0, y, t) != u(2*pi, y, t)) at the edges.

    Args:
        fields (dict): Clean predictions with keys "u", "v", "p", each a
                        torch.Tensor of shape (N, 1).
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; only "U0" is used here.
        epsilon (float): Perturbation strength.
        model: Unused for this perturbation; accepted only to keep a uniform
               function signature across all registered perturbations.
        sigma (float): Width of the Gaussian bumps localizing the
                       perturbation near the x-boundaries. Defaults to 0.2,
                       per the Section 7 specification.
        **kwargs: Accepted and ignored, so all registered perturbations share
                   a uniform call signature via apply_perturbation().

    Returns:
        dict: {"u": hallucinated u (N, 1), "v": clean v (N, 1) unchanged,
               "p": clean p (N, 1) unchanged}.
    """
    U0 = params["U0"]
    x, y = coords["x"], coords["y"]

    two_pi = 2 * torch.pi
    m_x = torch.exp(-(x ** 2) / sigma ** 2) + torch.exp(-((two_pi - x) ** 2) / sigma ** 2)

    u_tilde = fields["u"] + epsilon * U0 * m_x * torch.sin(3 * y)

    return {
        "u": u_tilde,
        "v": fields["v"].clone(),
        "p": fields["p"].clone(),
    }


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------
def apply_perturbation(name, fields, coords, params, epsilon, model=None, **kwargs):
    """
    Looks up and applies a registered perturbation by name.

    Args:
        name    : One of PERTURBATION_REGISTRY.keys().
        fields  : Clean {"u", "v", "p"} tensors on the evaluation grid.
        coords  : {"x", "y", "t"} evaluation grid coordinates.
        params  : {"U0", "k", "T"} case-specific physical constants.
        epsilon : Perturbation strength.
        model   : Trained PINN (only needed for "temporal_mismatch").
        **kwargs: Forwarded to the underlying perturbation function, e.g.
                   sigma for "boundary" or no_grad for "temporal_mismatch".

    Returns:
        dict: {"u", "v", "p"} hallucinated fields.
    """
    if name not in PERTURBATION_REGISTRY:
        raise KeyError(f"Unknown perturbation '{name}'. Available: {list(PERTURBATION_REGISTRY)}")
    return PERTURBATION_REGISTRY[name](fields, coords, params, epsilon, model=model, **kwargs)


# Canonical strengths & ordering used throughout the hallucination sweep (Section 7)
EPSILON_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
PERTURBATION_NAMES = [
    "velocity_divergence",
    "momentum",
    "pressure",
    "temporal_mismatch",
    "boundary",
]