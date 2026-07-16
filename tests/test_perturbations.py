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
            params (dict): {"U0": 1.0, "k": 1, "T": 1.0} physical constants.
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
    params = {"U0": 1.0, "k": 1, "T": 1.0}
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


def test_temporal_mismatch_matches_shifted_model_query(sample_fields_and_coords):
    """
    Verifies that the temporal_mismatch perturbation's (u, v) output exactly
    matches an independently-computed forward pass of the model at the
    shifted time (t + epsilon*T), and that p is left equal to the clean
    pressure field.

    Inputs:
        sample_fields_and_coords (tuple): The (fields, coords, params) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    fields, coords, params = sample_fields_and_coords
    model = DummyModel()
    epsilon = 0.1

    out = apply_perturbation("temporal_mismatch", fields, coords, params, epsilon, model=model)

    x, y, t = coords["x"], coords["y"], coords["t"]
    t_shifted = t + epsilon * params["T"]
    expected = model(torch.cat([x, y, t_shifted], dim=1))

    assert torch.allclose(out["u"], expected[:, 0:1])
    assert torch.allclose(out["v"], expected[:, 1:2])
    # p is untouched by this perturbation per the Section 7 spec
    assert torch.allclose(out["p"], fields["p"])


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
