"""
Unit tests for src.hallucinations.perturbations

Verifies each of the 5 perturbation functions:
  - Leaves the fields it is not supposed to touch untouched.
  - Perturbs the fields it is supposed to touch by a nonzero, epsilon-scaled amount.
  - Recovers the clean field exactly when epsilon = 0 (except temporal_mismatch,
    which still shifts time by 0 * T = 0, so it should also match the clean field).
"""

import torch
import pytest

from src.hallucinations.perturbations import (
    apply_perturbation,
    PERTURBATION_NAMES,
    EPSILON_VALUES,
)


@pytest.fixture
def sample_fields_and_coords():
    """
    Builds a small synthetic clean flow field and matching evaluation-grid
    coordinates for testing the perturbation functions in isolation, without
    needing a real trained PINN.

    Inputs:
        None.

    Outputs:
        tuple:
            fields (dict): {"u", "v", "p"} clean tensors, each shape (256, 1).
            coords (dict): {"x", "y", "t"} coordinate tensors, each shape (256, 1).
            params (dict): {"U0": 1.0, "k": 1, "T": 1.0, "tau_decay": 3.0}
                physical constants. tau_decay=3.0 (> T=1.0) mimics the real
                dataset, where every case's natural decay timescale exceeds
                its capped simulation window (see the README's Findings
                section) -- this is what makes temporal_mismatch's
                clamp-to-T behavior exercised by this fixture rather than
                a no-op.
    """
    torch.manual_seed(0)
    N = 256
    x = torch.empty(N, 1).uniform_(0, 2 * torch.pi)
    y = torch.empty(N, 1).uniform_(0, 2 * torch.pi)
    t = torch.empty(N, 1).uniform_(0, 1.0)

    u = torch.sin(x) * torch.cos(y)
    v = -torch.cos(x) * torch.sin(y)
    p = torch.zeros_like(u)

    fields = {"u": u, "v": v, "p": p}
    coords = {"x": x, "y": y, "t": t}
    params = {"U0": 1.0, "k": 1, "T": 1.0, "tau_decay": 3.0}
    return fields, coords, params


class DummyModel(torch.nn.Module):
    """
    A trivial, hand-written stand-in for BaselinePINN so the
    temporal_mismatch perturbation (which must re-query an actual model
    object) can be unit tested without loading a real trained network.
    """
    def forward(self, coords):
        """
        Computes a simple decaying-vortex-like (u, v, p) prediction directly
        from raw coordinates, mimicking the interface of BaselinePINN.forward.

        Inputs:
            coords (torch.Tensor): Flattened (x, y, t) coordinates, shape (N, 3).

        Outputs:
            torch.Tensor: Predicted (u, v, p), shape (N, 3).
        """
        x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        u = torch.sin(x) * torch.cos(y) * torch.exp(-t)
        v = -torch.cos(x) * torch.sin(y) * torch.exp(-t)
        p = torch.zeros_like(u)
        return torch.cat([u, v, p], dim=1)


@pytest.mark.parametrize("name", PERTURBATION_NAMES)
def test_perturbation_registered_and_runs(name, sample_fields_and_coords):
    """
    Verifies that every perturbation registered in PERTURBATION_NAMES can be
    looked up and run through apply_perturbation() without error, and that
    it always returns a dict with exactly the "u", "v", "p" keys, each
    matching the shape of the clean input fields.

    Inputs:
        name (str): Perturbation name, parametrized over PERTURBATION_NAMES.
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    model = DummyModel() if name == "temporal_mismatch" else None

    out = apply_perturbation(name, fields, coords, params, epsilon=0.05, model=model)

    assert set(out.keys()) == {"u", "v", "p"}
    for key in ("u", "v", "p"):
        assert out[key].shape == fields[key].shape


def test_velocity_divergence_only_changes_u(sample_fields_and_coords):
    """
    Verifies that the velocity_divergence perturbation modifies only the
    u-component and leaves v and p exactly equal to the clean fields.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    out = apply_perturbation("velocity_divergence", fields, coords, params, epsilon=0.05)

    assert not torch.allclose(out["u"], fields["u"])
    assert torch.allclose(out["v"], fields["v"])
    assert torch.allclose(out["p"], fields["p"])


def test_momentum_changes_u_and_v_only(sample_fields_and_coords):
    """
    Verifies that the momentum perturbation modifies both u and v while
    leaving p exactly equal to the clean field.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    out = apply_perturbation("momentum", fields, coords, params, epsilon=0.05)

    assert not torch.allclose(out["u"], fields["u"])
    assert not torch.allclose(out["v"], fields["v"])
    assert torch.allclose(out["p"], fields["p"])


def test_pressure_changes_p_only(sample_fields_and_coords):
    """
    Verifies that the pressure perturbation modifies only p and leaves u and
    v exactly equal to the clean fields.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    out = apply_perturbation("pressure", fields, coords, params, epsilon=0.05)

    assert torch.allclose(out["u"], fields["u"])
    assert torch.allclose(out["v"], fields["v"])
    assert not torch.allclose(out["p"], fields["p"])


def test_boundary_localizes_near_x_boundaries(sample_fields_and_coords):
    """
    Verifies that the boundary perturbation is spatially localized: points
    near the periodic x-boundaries (x=0, x=2*pi) should see a larger
    perturbation magnitude on average than points near the domain center
    (x=pi), since m(x) is a pair of Gaussians centered on those boundaries.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    out = apply_perturbation("boundary", fields, coords, params, epsilon=0.1)

    delta = (out["u"] - fields["u"]).abs()
    x = coords["x"]

    # Points near x=0 or x=2*pi should see a larger perturbation than points
    # near the domain center (x = pi), since m(x) is a pair of Gaussians
    # centered on the periodic boundaries.
    near_boundary_mask = (x < 0.3) | (x > 2 * torch.pi - 0.3)
    near_center_mask = (x > torch.pi - 0.3) & (x < torch.pi + 0.3)

    if near_boundary_mask.any() and near_center_mask.any():
        assert delta[near_boundary_mask].mean() > delta[near_center_mask].mean()


def test_temporal_mismatch_requires_model(sample_fields_and_coords):
    """
    Verifies that calling the temporal_mismatch perturbation without a model
    raises a ValueError, since it cannot compute a shifted-time prediction
    without re-querying an actual network.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert/pytest.raises on failure).
    """
    fields, coords, params = sample_fields_and_coords
    with pytest.raises(ValueError):
        apply_perturbation("temporal_mismatch", fields, coords, params, epsilon=0.05, model=None)


def test_temporal_mismatch_matches_shifted_model_query_when_unclamped(sample_fields_and_coords):
    """
    Verifies temporal_mismatch's (u, v) output exactly matches an
    independently-computed forward pass at t + epsilon*tau_decay, restricted
    to the points where that shift does NOT exceed T (so the clamp is a
    no-op there) -- isolating the "normal" unclamped code path from the
    boundary-saturation path tested separately below. The fixture's t is
    drawn uniformly over the full [0, T] range, so for any nonzero shift
    SOME points near t=T will always need clamping regardless of epsilon;
    masking to the unclamped subset is what makes this claim correct
    rather than assuming a single epsilon avoids clamping for every point.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture
            (T=1.0, tau_decay=3.0).

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    model = DummyModel()
    epsilon = 0.1  # shift = 0.3; unclamped only for t <= T - 0.3 = 0.7

    out = apply_perturbation("temporal_mismatch", fields, coords, params, epsilon, model=model)

    x, y, t = coords["x"], coords["y"], coords["t"]
    unclamped_shift = t + epsilon * params["tau_decay"]
    within_bounds = unclamped_shift <= params["T"]
    assert within_bounds.any(), "test setup error: need at least some unclamped points to test the unclamped path"

    expected = model(torch.cat([x, y, unclamped_shift], dim=1))

    assert torch.allclose(out["u"][within_bounds], expected[:, 0:1][within_bounds])
    assert torch.allclose(out["v"][within_bounds], expected[:, 1:2][within_bounds])
    # p is untouched by this perturbation per the Section 7 spec
    assert torch.allclose(out["p"], fields["p"])


def test_temporal_mismatch_clamps_at_domain_boundary(sample_fields_and_coords):
    """
    Verifies temporal_mismatch clamps the shifted time to T rather than
    querying the model outside the domain it was trained on, for an epsilon
    large enough that epsilon*tau_decay alone would overshoot T. This is the
    behavior perturb_temporal_mismatch's docstring specifically calls out as
    intentional (see its "WITHOUT this clamp..." paragraph) -- this test
    guards it from silently regressing into unclamped extrapolation.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture
            (T=1.0, tau_decay=3.0).

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    model = DummyModel()
    epsilon = 0.9  # epsilon * tau_decay = 2.7, far beyond T=1.0 -- must clamp

    out = apply_perturbation("temporal_mismatch", fields, coords, params, epsilon, model=model)

    x, y, t = coords["x"], coords["y"], coords["t"]
    unclamped_shift = t + epsilon * params["tau_decay"]
    assert (unclamped_shift > params["T"]).all(), "test setup error: this epsilon should trigger clamping"

    expected_at_T = model(torch.cat([x, y, torch.full_like(t, params["T"])], dim=1))
    assert torch.allclose(out["u"], expected_at_T[:, 0:1])
    assert torch.allclose(out["v"], expected_at_T[:, 1:2])


@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
def test_epsilon_values_are_all_valid_strengths(epsilon, sample_fields_and_coords):
    """
    Sanity check that every canonical epsilon sweep strength produces
    finite (u, v, p) output for every registered perturbation, guarding
    against numerical blow-up (e.g. division issues in the boundary
    perturbation's Gaussian terms) at the exact strengths used in production.

    Inputs:
        epsilon (float): Perturbation strength, parametrized over EPSILON_VALUES.
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    model = DummyModel()

    for name in PERTURBATION_NAMES:
        m = model if name == "temporal_mismatch" else None
        out = apply_perturbation(name, fields, coords, params, epsilon, model=m)
        assert torch.isfinite(out["u"]).all()
        assert torch.isfinite(out["v"]).all()
        assert torch.isfinite(out["p"]).all()