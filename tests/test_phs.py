"""
Unit tests for src.detection.phs and src.detection.evaluate_phs

Verifies the PHS formula module's building blocks in isolation, using a
small synthetic PerfectTGVModel (an nn.Module wrapping the analytical
Taylor-Green Vortex solution, in the same spirit as test_residuals.py and
test_perturbations.py's DummyModel) rather than a real trained checkpoint
-- so these tests are fast, deterministic, and don't depend on models/*.pth
existing on disk.

Three groups of tests:
  1. Component-level physics checks (a perfect field gives ~0 everywhere;
     the 3 PDE-residual perturbations raise Smom/Sdiv; the documented Sbc
     blind spot for the "boundary" perturbation, see phs.py's docstring).
  2. Pure bookkeeping checks (normalization, baseline scoring, threshold
     selection) against hand-computed expected values.
  3. An end-to-end synthetic detection check (evaluate_detection() recovers
     perfect AUC/F1 when hallucinated fields are obviously separated from
     clean ones).
"""

import numpy as np
import pandas as pd
import torch
import pytest

from src.physics.taylor_green import generate_tgv, compute_nu, compute_T
from src.models.scaling import ResidualScaler
from src.detection.phs import (
    compute_momentum_divergence_violation,
    compute_boundary_violation,
    compute_phs_components,
    compute_normalizers,
    normalize_components,
    compute_baseline_scores,
    select_threshold,
    PHS_COMPONENT_NAMES,
    BASELINE_DEFINITIONS,
)
from src.detection.evaluate_phs import evaluate_detection


class PerfectTGVModel(torch.nn.Module):
    """
    A model that returns the EXACT analytical Taylor-Green Vortex solution
    for a fixed set of case parameters. A perfect model should give Smom,
    Sdiv, Sbc, SE all ~0 (up to floating point error), since the analytical
    solution satisfies the PDEs (see test_residuals.py), is exactly
    periodic, and follows its own energy decay law by construction --
    making it a convenient, fully deterministic ground truth for testing
    the PHS component functions without a real trained checkpoint.
    """
    def __init__(self, U0, k, phi_x, phi_y, nu):
        super().__init__()
        self.U0, self.k, self.phi_x, self.phi_y, self.nu = U0, k, phi_x, phi_y, nu

    def forward(self, coords):
        """
        Inputs:
            coords (torch.Tensor): Flattened (x, y, t) coordinates, shape (N, 3).
        Outputs:
            torch.Tensor: Analytical (u, v, p), shape (N, 3).
        """
        x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        u, v, p = generate_tgv(x, y, t, self.U0, self.k, self.phi_x, self.phi_y, self.nu)
        return torch.cat([u, v, p], dim=1)


@pytest.fixture
def perfect_case():
    """
    A small, fixed TGV case with its PerfectTGVModel, ResidualScaler, and
    physical constants, for testing the PHS component functions against a
    known-exact solution.

    Inputs:
        None.

    Outputs:
        tuple: (model, case_meta, nu, T, scaler).
    """
    U0, Re, k, phi_x, phi_y = 1.0, 100.0, 1, 0.0, 0.0
    nu, T = compute_nu(U0, Re, k), compute_T(U0, Re, k)
    model = PerfectTGVModel(U0, k, phi_x, phi_y, nu)
    scaler = ResidualScaler(U0, k)
    case_meta = {"U0": U0, "k": k, "phi_x": phi_x, "phi_y": phi_y}
    return model, case_meta, nu, T, scaler


# ---------------------------------------------------------------------
# 1. Component-level physics checks
# ---------------------------------------------------------------------

def test_perfect_field_gives_near_zero_components(perfect_case):
    """
    Verifies that a model returning the EXACT analytical TGV solution
    produces Smom, Sdiv, Sbc, SE all near zero for the clean ("none")
    field.

    Inputs:
        perfect_case (tuple): The (model, case_meta, nu, T, scaler) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    model, case_meta, nu, T, scaler = perfect_case
    components = compute_phs_components(
        model, case_meta, nu, T, scaler, perturbation_name="none", epsilon=0.0,
        n_interior=2000, n_bc_per_axis=200, n_time=6, energy_res=12, chunk_size=2000,
    )
    for name in PHS_COMPONENT_NAMES:
        assert components[name] < 1e-6, f"{name} = {components[name]} should be ~0 for the exact solution"


@pytest.mark.parametrize("name", ["velocity_divergence", "momentum", "pressure"])
def test_pde_perturbations_raise_momentum_divergence(name, perfect_case):
    """
    Verifies Smom + Sdiv strictly increases from epsilon=0.01 to
    epsilon=0.1 for perturbation types known (from the Issue #9 audit) to
    inject a genuine PDE-residual violation: velocity_divergence, momentum,
    and pressure.

    Inputs:
        name (str): Perturbation name, parametrized.
        perfect_case (tuple): The (model, case_meta, nu, T, scaler) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    model, case_meta, nu, T, scaler = perfect_case
    params = {"U0": case_meta["U0"], "k": case_meta["k"], "T": T}

    Smom_small, Sdiv_small = compute_momentum_divergence_violation(
        model, T, params, name, 0.01, nu, scaler, n_interior=3000, chunk_size=3000)
    Smom_large, Sdiv_large = compute_momentum_divergence_violation(
        model, T, params, name, 0.1, nu, scaler, n_interior=3000, chunk_size=3000)

    assert (Smom_large + Sdiv_large) > (Smom_small + Sdiv_small)
    assert (Smom_large + Sdiv_large) > 1e-4  # meaningfully nonzero, not just "bigger than ~0"


def test_boundary_perturbation_blind_to_sbc_but_not_smom(perfect_case):
    """
    Regression test for the documented Sbc blind spot (see phs.py's module
    docstring): the "boundary" perturbation's m(x) satisfies m(0)=m(2*pi)
    exactly (and its sin(3y) factor vanishes exactly at y=0 and y=2*pi),
    so the perturbation contributes identically -- and therefore cancels
    -- on both sides of every boundary pair. Sbc should therefore stay
    ~unchanged from its clean value, while Smom should rise substantially.
    This is what makes PHS's sum still detect this perturbation type
    despite Sbc missing it (confirmed empirically in Issue #9's
    residual_summary.json: whole-domain mse_Ru rose ~500x across the
    epsilon sweep for this perturbation type while bc_u_mismatch stayed
    frozen).

    Inputs:
        perfect_case (tuple): The (model, case_meta, nu, T, scaler) fixture.

    Outputs:
        None (raises via assert on failure).
    """
    model, case_meta, nu, T, scaler = perfect_case
    params = {"U0": case_meta["U0"], "k": case_meta["k"], "T": T}

    Sbc_clean = compute_boundary_violation(
        model, T, params, "none", 0.0, case_meta["U0"], scaler.scale_p, n_bc_per_axis=500)
    Sbc_perturbed = compute_boundary_violation(
        model, T, params, "boundary", 0.1, case_meta["U0"], scaler.scale_p, n_bc_per_axis=500)
    Smom_clean, _ = compute_momentum_divergence_violation(
        model, T, params, "none", 0.0, nu, scaler, n_interior=3000, chunk_size=3000)
    Smom_perturbed, _ = compute_momentum_divergence_violation(
        model, T, params, "boundary", 0.1, nu, scaler, n_interior=3000, chunk_size=3000)

    assert Sbc_perturbed == pytest.approx(Sbc_clean, abs=1e-6)
    assert Smom_perturbed > 1e-4
    assert Smom_perturbed > Smom_clean


# ---------------------------------------------------------------------
# 2. Pure bookkeeping checks (no model, no physics -- just the formula)
# ---------------------------------------------------------------------

def test_normalize_and_baseline_scores_match_hand_computed_values():
    """
    Verifies compute_normalizers / normalize_components / compute_baseline_scores
    against hand-computed expected values on a tiny synthetic DataFrame,
    confirming the bookkeeping matches Section 8's formula exactly:
    S_bar_j = Sj / mean(Sj over clean validation fields), and each
    baseline score is the sum of its listed components' S_bar values.

    Inputs:
        None.

    Outputs:
        None (raises via assert on failure).
    """
    df = pd.DataFrame({
        "split": ["validation", "validation", "test", "test"],
        "label": ["clean", "clean", "clean", "hallucinated"],
        "mom": [1.0, 3.0, 2.0, 20.0],
        "div": [2.0, 2.0, 2.0, 40.0],
        "bc": [4.0, 4.0, 4.0, 4.0],
        "E": [1.0, 1.0, 1.0, 1.0],
    })
    valid_val = df[(df["split"] == "validation") & (df["label"] == "clean")]

    normalizers = compute_normalizers(valid_val)
    assert normalizers == pytest.approx({"mom": 2.0, "div": 2.0, "bc": 4.0, "E": 1.0})

    scored = compute_baseline_scores(normalize_components(df, normalizers))
    halluc_row = scored[(scored["split"] == "test") & (scored["label"] == "hallucinated")].iloc[0]

    assert halluc_row["mom_bar"] == pytest.approx(10.0)  # 20 / 2
    assert halluc_row["div_bar"] == pytest.approx(20.0)  # 40 / 2
    assert halluc_row["bc_bar"] == pytest.approx(1.0)  # 4 / 4
    assert halluc_row["E_bar"] == pytest.approx(1.0)  # 1 / 1
    assert halluc_row["Score1_momentum_only"] == pytest.approx(10.0)
    assert halluc_row["Score2_momentum_divergence"] == pytest.approx(30.0)
    assert halluc_row["Score3_PHS_full"] == pytest.approx(32.0)


def test_select_threshold_is_the_95th_percentile():
    """
    Verifies select_threshold matches numpy's percentile function directly
    (Section 8: tau = percentile95(PHS_valid)).

    Inputs:
        None.

    Outputs:
        None (raises via assert on failure).
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert select_threshold(values, percentile=95.0) == pytest.approx(np.percentile(values, 95.0))


def test_baseline_definitions_are_nested_subsets():
    """
    Verifies Score1 subset-of Score2 subset-of Score3's component sets,
    and that Score3 covers all 4 components -- matching WP5's intent that
    Score2/Score3 are strict supersets of the simpler baselines they are
    compared against. If this breaks, the AUC(PHS) > AUC(Score2)
    acceptance criterion stops meaning what it's supposed to.

    Inputs:
        None.

    Outputs:
        None (raises via assert on failure).
    """
    s1 = set(BASELINE_DEFINITIONS["Score1_momentum_only"])
    s2 = set(BASELINE_DEFINITIONS["Score2_momentum_divergence"])
    s3 = set(BASELINE_DEFINITIONS["Score3_PHS_full"])
    assert s1 <= s2 <= s3
    assert s3 == set(PHS_COMPONENT_NAMES)


# ---------------------------------------------------------------------
# 3. End-to-end synthetic detection check
# ---------------------------------------------------------------------

def test_evaluate_detection_recovers_perfect_separation():
    """
    Builds a synthetic DataFrame where every clean field's raw components
    are EXACTLY 1.0 (both splits -- zero within-class variance) and every
    hallucinated field's mom/div components are ~10x that (bc/E left at
    1.0, mirroring how a real perturbation only raises SOME components --
    see the Sbc blind-spot test above), and verifies evaluate_detection()
    recovers AUC=1.0 and F1=1.0 on the test split.

    Clean fields are held EXACTLY equal (not just close) so the 95th-
    percentile threshold from the 5 validation-clean fields exactly
    matches the 5 test-clean fields' own scores, with none exceeding it.
    A small amount of independent, real-valued noise on each side (as a
    real dataset would have) makes an occasional test-clean field land
    fractionally above a threshold estimated from only 5 points -- that
    is expected small-sample calibration noise (see the PHS distribution
    discussion), not a pipeline bug, so it does not belong in a
    correctness test for the pipeline itself.

    Inputs:
        None.

    Outputs:
        None (raises via assert on failure).
    """
    rng = np.random.default_rng(0)
    rows = []
    for split, n_clean, n_halluc in [("validation", 5, 0), ("test", 5, 15)]:
        for _ in range(n_clean):
            rows.append({"split": split, "label": "clean", "mom": 1.0, "div": 1.0, "bc": 1.0, "E": 1.0})
        for _ in range(n_halluc):
            rows.append({"split": split, "label": "hallucinated",
                         "mom": rng.uniform(9, 11), "div": rng.uniform(9, 11), "bc": 1.0, "E": 1.0})
    df = pd.DataFrame(rows)

    _, _, metrics_rows, _ = evaluate_detection(df, percentile=95.0)

    for m in metrics_rows:
        assert m["roc_auc"] == pytest.approx(1.0), f"{m['score_name']}: AUC={m['roc_auc']}"
        assert m["f1"] == pytest.approx(1.0), f"{m['score_name']}: F1={m['f1']}"


def test_evaluate_detection_raises_without_validation_clean_fields():
    """
    Verifies evaluate_detection() raises a clear RuntimeError (rather than
    silently producing garbage normalizers) when no clean validation-split
    rows are present -- guards the calibration precondition documented in
    Section 8 ("normalized using valid validation fields").

    Inputs:
        None.

    Outputs:
        None (raises via assert/pytest.raises on failure).
    """
    df = pd.DataFrame({
        "split": ["test", "test"],
        "label": ["clean", "hallucinated"],
        "mom": [1.0, 10.0], "div": [1.0, 10.0], "bc": [1.0, 1.0], "E": [1.0, 1.0],
    })
    with pytest.raises(RuntimeError):
        evaluate_detection(df, percentile=95.0)