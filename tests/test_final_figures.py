"""
Unit tests for src.detection.final_figures (#36).

On a tiny made-up benchmark (6 cases, 2 per k, each with its clean field and 2 damage
types at 2 strengths; worked out by hand below), verifies that:
  - recall by score x k x type x strength counts the flags, over all cases and over the
    cases whose clean field the score doesn't flag, a score equal to tau is not flagged,
    and each row gets the exact range of its count;
  - the case table keeps the case-file order and refuses inputs that disagree: a case
    missing or listed twice, a pass/fail that differs between loss_summary.csv, the true
    errors and the benchmark, or that doesn't follow from the acceptance metrics;
  - the caption helpers find the strengths of full recall, move labels only up, and
    refuse a zero-width range whose cases don't share one count;
  - the size of a damage follows the damage formulas (added damages, the time shift,
    the pressure damage that leaves the velocity alone);
  - a row of the natural errors counts the checkpoints by hand, with the AUC of
    inaccurate against accurate, and refuses a #39 row that doesn't follow from the flags;
  - the PDF of a figure has no title and the PNG has one.
And, on the committed results:
  - the inputs agree (the benchmark and the checkpoints follow from the frozen
    calibration, #35's metrics match the flags and the scores), and a changed flag,
    count, AUC, tau or range is refused;
  - both experiments are in the same terms, the synthetic rows being #35's numbers;
  - the captions give the numbers #36 asks them to explain, and are plain LaTeX;
  - run() writes every figure and table, a second run gives the same bytes, and the
    committed results/figures tables and captions are what the script writes.
"""

import hashlib
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.detection import benchmark_metrics as bm
from src.detection import final_figures as ff
from src.detection import frozen_calibration
from src.detection import natural_errors as ne
from src.detection.grouped_eval import clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations.perturbations import PERTURBATION_NAMES
from src.models.accuracy_audit import accuracy_grid, read_cases
from src.physics.taylor_green import compute_decay_timescale, compute_nu, compute_T, generate_tgv

REPO = Path(__file__).parent.parent
OUTPUT_DIR = REPO / "results" / "figures"

S1, S2, S3, S4, S5 = SCORE_DEFINITIONS
# Only mom is nonzero and the normalizers are 1, so every score of a field equals its mom
THRESHOLDS = {S1: 1.0, S2: 2.0, S3: 3.0, S4: 3.0, S5: 100.0}
K = {"case_30": 1, "case_31": 1, "case_40": 2, "case_41": 2, "case_50": 3, "case_51": 3}
FIELDS = (("none", 0.0), ("momentum", 0.0001), ("momentum", 0.01), ("pressure", 0.0001), ("pressure", 0.01))
VALUES = {  # mom of each field of FIELDS
    "case_30": (0.5, 0.6, 5.0, 0.4, 2.5),
    "case_31": (1.5, 1.6, 4.0, 1.4, 3.5),   # Clean above Score1's tau: already flagged by Score1
    "case_40": (0.2, 0.3, 2.5, 0.2, 0.9),
    "case_41": (0.1, 1.0, 1.2, 0.1, 6.0),   # 1.0 equals Score1's tau, so it is not flagged
    "case_50": (0.3, 0.3, 0.8, 0.3, 1.1),
    "case_51": (2.5, 2.6, 9.0, 2.4, 3.0),   # Already flagged by Score1 and Score2
}
# Worked out by hand: (k, type, strength): (flagged, out of; new flagged, out of)
BY_HAND = {
    S1: {(1, "momentum", 0.0001): (1, 2, 0, 1), (1, "momentum", 0.01): (2, 2, 1, 1),
         (1, "pressure", 0.0001): (1, 2, 0, 1), (1, "pressure", 0.01): (2, 2, 1, 1),
         (2, "momentum", 0.0001): (0, 2, 0, 2), (2, "momentum", 0.01): (2, 2, 2, 2),
         (2, "pressure", 0.0001): (0, 2, 0, 2), (2, "pressure", 0.01): (1, 2, 1, 2),
         (3, "momentum", 0.0001): (1, 2, 0, 1), (3, "momentum", 0.01): (1, 2, 0, 1),
         (3, "pressure", 0.0001): (1, 2, 0, 1), (3, "pressure", 0.01): (2, 2, 1, 1)},
    S2: {(1, "momentum", 0.0001): (0, 2, 0, 2), (1, "momentum", 0.01): (2, 2, 2, 2),
         (1, "pressure", 0.0001): (0, 2, 0, 2), (1, "pressure", 0.01): (2, 2, 2, 2),
         (2, "momentum", 0.0001): (0, 2, 0, 2), (2, "momentum", 0.01): (1, 2, 1, 2),
         (2, "pressure", 0.0001): (0, 2, 0, 2), (2, "pressure", 0.01): (1, 2, 1, 2),
         (3, "momentum", 0.0001): (1, 2, 0, 1), (3, "momentum", 0.01): (1, 2, 0, 1),
         (3, "pressure", 0.0001): (1, 2, 0, 1), (3, "pressure", 0.01): (1, 2, 0, 1)},
}


def make_calibration() -> dict:
    return {
        "format_version": frozen_calibration.FORMAT_VERSION,
        "description": "Made-up calibration of tests/test_final_figures.py",
        "normalizers": {component: 1.0 for component in PHS_COMPONENT_NAMES},
        "normalizer_eps": 0.0,
        "scores": {name: list(components) for name, components in SCORE_DEFINITIONS.items()},
        "thresholds": dict(THRESHOLDS),
    }


def make_fields() -> pd.DataFrame:
    """The made-up benchmark, laid out like results/benchmark/confirmation_scores_raw.csv."""
    rows = []
    for case_id, values in VALUES.items():
        for (perturbation, epsilon), mom in zip(FIELDS, values):
            rows.append({"case_id": case_id, "k": K[case_id], "perturbation": perturbation, "epsilon": epsilon,
                         "label": "clean" if perturbation == "none" else "hallucinated",
                         "mom": mom, "div": 0.0, "bc": 0.0, "E": 0.0})
    scored = frozen_calibration.apply(pd.DataFrame(rows), make_calibration())
    return bm.build_field_table(scored, {case_id: True for case_id in K})


def make_case_inputs(passed=None):
    """A case list, true errors, a loss summary and a benchmark for the made-up cases; passed: {case_id: bool}."""
    passed = passed or {}
    cases = [{"case_id": case_id, "k": k, "U0": 1.0 + i / 10, "Re": 50.0 + i, "phi_x": 0.1 * i, "phi_y": 0.2 * i,
              "seed_points": 3000 + int(case_id[5:]), "seed_train": 4000 + int(case_id[5:]),
              "seed_scoring": 1000 + int(case_id[5:]), "split": "confirmation"}
             for i, (case_id, k) in enumerate(K.items())]
    errors = pd.DataFrame([{"case_id": case["case_id"], "k": case["k"], "stage": stage,
                            "relL2_uv": 0.002 * (j + 1) if stage == "lbfgs_final" else 0.3,
                            "relL2_uv_T": 0.004 * (j + 1), "relL2_p": 0.005 * (j + 1),
                            "passed_acceptance": passed.get(case["case_id"], True)}
                           for j, case in enumerate(cases) for stage in ("adam_05000", "lbfgs_final")])
    loss = pd.DataFrame([{"case_id": case["case_id"], "rel_l2": 0.002 * (j + 1) if passed.get(case["case_id"], True)
                          else 0.2, "mse_rc": 1e-6, "mse_rm": 1e-5, "passed": passed.get(case["case_id"], True)}
                         for j, case in enumerate(cases)])
    fields = make_fields()
    fields["passed_acceptance"] = fields["case_id"].map(lambda c: passed.get(c, True))
    return cases, errors, loss, fields


# A made-up set of checkpoints: (case, k, stage, inaccurate, Score1), with Score1's tau 1
NATURAL_ROWS = [
    ("case_30", 1, "adam_05000", True, 5.0), ("case_30", 1, "lbfgs_final", False, 0.5),
    ("case_31", 1, "adam_05000", False, 3.0),    # Accurate but flagged: a false alarm
    ("case_31", 1, "lbfgs_final", False, 0.8),
    ("case_40", 2, "adam_05000", True, 0.9),     # Inaccurate but not flagged: missed
    ("case_40", 2, "lbfgs_final", False, 1.5),   # A false alarm
    ("case_41", 2, "adam_05000", True, 7.0), ("case_41", 2, "lbfgs_final", False, 0.2),
]
SIZED_CASE = {"case_id": "case_30", "k": 2, "U0": 1.3, "Re": 80.0, "phi_x": 0.4, "phi_y": 1.1}


def make_natural() -> pd.DataFrame:
    table = pd.DataFrame(NATURAL_ROWS, columns=["case_id", "k", "stage", "inaccurate", S1])
    table[f"flagged_{S1}"] = table[S1] > THRESHOLDS[S1]
    table["passed_acceptance"] = True
    return table


def stage_row(table: pd.DataFrame, n_boot: int = 20) -> pd.Series:
    """#39's row over all stages for Score1, as natural_errors.py writes it."""
    return pd.Series(ne.stage_by_k_row(table, S1, THRESHOLDS[S1], n_boot=n_boot, seed=0))


@pytest.fixture(scope="module")
def committed():
    """Every input of the committed results, read and checked (no file written)."""
    return ff.build()


# --- Recall by score x k x type x strength ---

@pytest.mark.parametrize("score", [S1, S2])
def test_recall_by_k_counts_by_hand(score):
    table = ff.recall_by_type_eps_k(make_fields())
    assert list(table.columns) == list(ff.RECALL_BY_K_COLUMNS)
    rows = table[table["score"] == score]
    assert len(rows) == len(BY_HAND[score])
    for (k, perturbation, epsilon), (flagged, n, new_flagged, n_new) in BY_HAND[score].items():
        row = rows[(rows["k"] == k) & (rows["perturbation"] == perturbation) & (rows["epsilon"] == epsilon)].iloc[0]
        assert (row["n_damaged_flagged"], row["n_damaged"]) == (flagged, n), (k, perturbation, epsilon)
        assert (row["n_damaged_new_flagged"], row["n_damaged_new"]) == (new_flagged, n_new), (k, perturbation, epsilon)
        assert row["recall"] == flagged / n
        assert (row["recall_low"], row["recall_high"]) == clopper_pearson(flagged, n)
        assert (row["recall_new_low"], row["recall_new_high"]) == clopper_pearson(new_flagged, n_new)
        assert row["n_cases"] == 2


def test_recall_by_k_covers_every_score_k_type_and_strength():
    table = ff.recall_by_type_eps_k(make_fields())
    assert len(table) == len(SCORE_DEFINITIONS) * 3 * 2 * 2
    assert not table.duplicated(["score", "k", "perturbation", "epsilon"]).any()
    # Score5's tau is above every field, so it flags nothing
    assert (table.loc[table["score"] == S5, "n_damaged_flagged"] == 0).all()


def test_recall_by_k_refuses_two_fields_of_a_case_at_one_strength():
    fields = make_fields()
    doubled = pd.concat([fields, fields[(fields["case_id"] == "case_30") & (fields["perturbation"] == "momentum")]])
    with pytest.raises(ValueError, match="two damaged fields"):
        ff.recall_by_type_eps_k(doubled)


# --- The case table ---

def test_case_table_in_case_file_order():
    cases, errors, loss, fields = make_case_inputs()
    table = ff.case_table(cases, errors, loss, fields)
    assert list(table.columns) == list(ff.CASE_COLUMNS)
    assert table["case_id"].tolist() == list(K)
    assert table["passed_acceptance"].all()
    # relL2_uv is the final model's (lbfgs_final), not an Adam checkpoint's
    assert table["relL2_uv"].tolist() == pytest.approx([0.002 * (j + 1) for j in range(len(K))])
    assert table["relL2_p"].tolist() == pytest.approx([0.005 * (j + 1) for j in range(len(K))])
    assert table["seed_points"].tolist() == [3000 + int(c[5:]) for c in K]
    # The flow quantities follow from U0, Re and k: nu = U0 / (Re k), tau_d = 1 / (2 nu k^2), T = min(2, tau_d)
    # Every row, since k = 1 can't tell k from k^2
    assert table["nu"].to_numpy() == pytest.approx((table["U0"] / (table["Re"] * table["k"])).to_numpy())
    assert table["tau_decay"].to_numpy() == pytest.approx((1 / (2 * table["nu"] * table["k"] ** 2)).to_numpy())
    assert (table["T"] == np.minimum(2.0, table["tau_decay"])).all()


def test_case_table_refuses_errors_without_pressure():
    cases, errors, loss, fields = make_case_inputs()
    with pytest.raises(ValueError, match="relL2_p"):
        ff.case_table(cases, errors.drop(columns="relL2_p"), loss, fields)


def test_case_table_keeps_a_failed_run():
    cases, errors, loss, fields = make_case_inputs(passed={"case_40": False})
    table = ff.case_table(cases, errors, loss, fields).set_index("case_id")
    assert not table.loc["case_40", "passed_acceptance"]
    assert table.drop(index="case_40")["passed_acceptance"].all()
    assert "5 of 6 runs passed (failed runs stay in every table)" in ff.caption_cases(table.reset_index())


@pytest.mark.parametrize("change", ["missing_in_loss", "twice_in_errors", "passed_differs", "not_following",
                                    "benchmark_differs", "k_differs"])
def test_case_table_refuses_inputs_that_disagree(change):
    cases, errors, loss, fields = make_case_inputs()
    if change == "missing_in_loss":
        loss = loss[loss["case_id"] != "case_41"]
    elif change == "twice_in_errors":
        errors = pd.concat([errors, errors[(errors["case_id"] == "case_41") & (errors["stage"] == "lbfgs_final")]])
    elif change == "passed_differs":
        errors.loc[errors["case_id"] == "case_41", "passed_acceptance"] = False
    elif change == "not_following":
        loss.loc[loss["case_id"] == "case_41", "mse_rm"] = 2e-3   # Above 1e-3, yet marked passed
    elif change == "benchmark_differs":
        fields.loc[fields["case_id"] == "case_41", "passed_acceptance"] = False
    elif change == "k_differs":
        errors.loc[errors["case_id"] == "case_41", "k"] = 3
    with pytest.raises(ValueError):
        ff.case_table(cases, errors, loss, fields)


def test_seed_rule():
    cases, errors, loss, fields = make_case_inputs()
    table = ff.case_table(cases, errors, loss, fields)
    assert ff._seed_rule(table, "seed_points") == " ($3000 + n$)"
    table.loc[0, "seed_points"] = 1
    assert ff._seed_rule(table, "seed_points") == ""


# --- Caption helpers ---

def test_full_recall_epsilon():
    table = pd.DataFrame({"score": S1, "perturbation": "momentum", "epsilon": [0.0001, 0.002, 0.01, 0.02],
                          "recall": [0.0, 1.0, 0.9, 1.0]})
    # 1.0 at 0.002 does not count: recall drops again at 0.01
    assert ff.full_recall_epsilon(table, "recall") == {(S1, "momentum"): 0.02}
    table["recall"] = [0.0, 0.5, 0.9, 0.9]
    assert np.isnan(ff.full_recall_epsilon(table, "recall")[(S1, "momentum")])


def test_recall_sentence_names_the_exceptions():
    # Most scores flag everything from 0.01 on; Score3 already from 0.002; Score1 only from 0.02
    counts = {S1: (1, 15, 30), S2: (10, 30, 30), S3: (30, 30, 30), S4: (12, 30, 30), S5: (9, 30, 30)}
    rows = [{"score": score, "perturbation": "boundary", "epsilon": epsilon, "recall": n / 30,
             "n_damaged_flagged": n, "n_damaged": 30}
            for score, values in counts.items() for epsilon, n in zip((0.002, 0.01, 0.02), values)]
    sentence = ff.recall_sentence(pd.DataFrame(rows), "recall", "n_damaged_flagged", "n_damaged")
    assert sentence.startswith("Every score flags every damaged field from $\\varepsilon = 0.01$ on")
    assert "(already from 0.002 for Score3 with boundary damage)" in sentence
    assert "except Score1 for boundary damage (15/30 at 0.01)" in sentence


def test_label_positions_move_only_up():
    heights = ff._label_positions([4.49, 4.28], min_ratio=1.16, floor=4.6)
    assert heights[1] == pytest.approx(4.6)             # The lower point's label, raised to the floor
    assert heights[0] == pytest.approx(4.6 * 1.16)      # The higher one, kept above it
    assert ff._label_positions([38.9, 23.4], min_ratio=1.16) == [38.9, 23.4]   # Far apart: unmoved


def test_two_significant_digits():
    assert [ff._two_digits(v) for v in (0.0074, 0.15, 1.043, 4.0, 15.1, 11.04)] == \
        ["0.0074", "0.15", "1.0", "4.0", "15", "11"]
    assert ff._size(0) == "0" and ff._size(0.01043) == "1.0"


def test_score_rises_by_hand():
    fields = make_fields()
    sizes = fields.loc[fields["label"] != "clean", ["case_id", "perturbation", "epsilon"]].assign(
        velocity_change=lambda d: np.where(d["perturbation"] == "pressure", 0.0, 1e-3))
    rises = ff.score_rises(fields, sizes, THRESHOLDS).set_index(["case_id", "epsilon"])
    assert set(rises["perturbation"]) == {"momentum"}   # The pressure damage leaves the velocity alone
    # case_30: clean 0.5, momentum at 0.01 gives 5.0, over Score1's tau of 1
    assert rises.loc[("case_30", 0.01), f"rise_{S1}"] == pytest.approx(4.5)
    assert rises.loc[("case_40", 0.01), f"rise_{S2}"] == pytest.approx((2.5 - 0.2) / THRESHOLDS[S2])


def test_relative_energy_error():
    clean = pd.DataFrame({"U0": [2.0], "E": [(0.01 * 2.0 ** 2 / 4) ** 2]})
    assert ff.relative_energy_error(clean).iloc[0] == pytest.approx(0.01)


def test_zero_width_recall_needs_one_shared_count():
    fields = make_fields()
    # Pretend the bootstrap gave Score1 no width for k = 2, whose cases flag 1 and 2 fields
    row = {"score": S1, "k": 2, "recall": 0.625, "recall_low": 0.625, "recall_high": 0.625}
    with pytest.raises(ValueError, match="same count"):
        ff.zero_width_recalls(pd.DataFrame([row]), fields)
    # Score2 for k = 1 is flagged twice in each case, so a zero width is consistent there
    row = {"score": S2, "k": 1, "recall": 0.5, "recall_low": 0.5, "recall_high": 0.5}
    (found_row, count, per_case), = ff.zero_width_recalls(pd.DataFrame([row]), fields)
    assert (count, per_case) == (2, 4)


def test_within_case_concordance():
    fields = make_fields()
    within, across = ff.within_case_concordance(fields, S1)
    # Damaged fields above their own clean field, ties counting one half: case_30 3, case_31 3,
    # case_40 3.5 (0.2 ties its clean 0.2), case_41 3.5 (0.1 ties), case_50 3 (two ties), case_51 3; of 24
    assert within == pytest.approx(19 / 24)
    assert 0 < across < 1


def test_names_cover_every_score_and_damage_type():
    assert set(ff.SCORE_LABELS) == set(ff.SCORE_COLOURS) == set(ff.SCORE_MARKERS) == set(SCORE_DEFINITIONS)
    assert set(ff.DAMAGE_LABELS) == set(ff.DAMAGE_MARKERS) == set(PERTURBATION_NAMES)
    assert len(set(ff.SCORE_COLOURS.values())) == len(SCORE_DEFINITIONS)
    assert len(set(ff.SCORE_MARKERS.values())) == len(SCORE_DEFINITIONS)
    assert len(set(ff.DAMAGE_MARKERS.values())) == len(PERTURBATION_NAMES)
    assert ff.NATURAL_MARKER not in ff.DAMAGE_MARKERS.values()
    assert set(ff.PNG_TITLES) == set(ff.FIGURES) and set(ff.NARROW_FIGURES) <= set(ff.FIGURES)


# --- The size of a damage ---

def test_damage_sizes_follow_the_damage_formulas():
    case = SIZED_CASE
    sizes = ff.damage_sizes([case], [0.002, 0.01]).set_index(["perturbation", "epsilon"])["velocity_change"]
    U0, k = case["U0"], case["k"]
    nu, T = compute_nu(U0, case["Re"], k), compute_T(U0, case["Re"], k)
    t, x, y = accuracy_grid(T)
    u, v, _ = generate_tgv(x, y, t, U0, k, case["phi_x"], case["phi_y"], nu)
    norm = torch.sqrt(torch.sum(u ** 2 + v ** 2)).item()
    # The velocity-divergence damage adds eps U0 sin(3x + 0.7) sin(2y) to u (perturbations.py)
    added = 0.01 * U0 * torch.sin(3 * x + 0.7) * torch.sin(2 * y)
    assert sizes[("velocity_divergence", 0.01)] == pytest.approx(torch.sqrt(torch.sum(added ** 2)).item() / norm,
                                                                 rel=1e-12)
    # An added damage grows linearly with eps; the pressure damage leaves the velocity alone
    assert sizes[("momentum", 0.01)] == pytest.approx(5 * sizes[("momentum", 0.002)], rel=1e-12)
    assert sizes[("pressure", 0.002)] == sizes[("pressure", 0.01)] == 0.0
    # The time shift reports the exact flow at t + eps * tau_decay, clamped to [0, T]
    shifted = torch.clamp(t + 0.002 * compute_decay_timescale(nu, k), 0, T)
    u_shifted, v_shifted, _ = generate_tgv(x, y, shifted, U0, k, case["phi_x"], case["phi_y"], nu)
    change = torch.sqrt(torch.sum((u_shifted - u) ** 2 + (v_shifted - v) ** 2)).item() / norm
    assert sizes[("temporal_mismatch", 0.002)] == pytest.approx(change, rel=1e-12)


def test_damage_size_table_summarizes_the_cases():
    sizes = pd.DataFrame({"case_id": ["case_30", "case_31", "case_32"] * 2, "k": 1,
                          "perturbation": ["momentum"] * 3 + ["pressure"] * 3, "epsilon": 0.01,
                          "velocity_change": [0.004, 0.001, 0.002, 0.0, 0.0, 0.0]})
    table = ff.damage_size_table(sizes)
    assert list(table.columns) == list(ff.DAMAGE_SIZE_COLUMNS)
    first = table.iloc[0]
    assert (first["perturbation"], first["n_cases"]) == ("momentum", 3)
    # The median (0.002), not the mean (0.00233)
    assert (first["velocity_change_min"], first["velocity_change_median"], first["velocity_change_max"]) == \
        (0.001, 0.002, 0.004)
    assert table.iloc[1]["velocity_change_max"] == 0.0
    sizes.loc[3, "velocity_change"] = 1e-3   # One pressure-damaged field whose velocity changed
    with pytest.raises(ValueError, match="some fields but not"):
        ff.damage_size_table(sizes)


def test_damage_field_maps_follow_the_derivatives():
    """The residual maps of two damages, against their derivatives worked out by hand."""
    case, epsilon, n = SIZED_CASE, 0.01, 32
    info = ff.damage_field_maps(case, epsilon, n=n)
    U0, k = case["U0"], case["k"]
    X, Y = np.meshgrid(info["axis"], info["axis"], indexing="ij")
    # Velocity divergence adds eps U0 sin(3x + 0.7) sin(2y) to u: its continuity residual is d/dx of it
    expected = np.abs(3 * epsilon * U0 * np.cos(3 * X + 0.7) * np.sin(2 * Y)) / (U0 * k)
    np.testing.assert_allclose(info["maps"]["velocity_divergence"]["continuity"], expected, rtol=1e-9, atol=1e-12)
    # The pressure damage adds eps U0^2 cos(5x) cos(4y) to p: it changes only the pressure, and its momentum
    # residual is the gradient of that
    pressure = info["maps"]["pressure"]
    assert pressure["change_of"] == "pressure"
    gradient = epsilon * U0 ** 2 * np.sqrt((5 * np.sin(5 * X) * np.cos(4 * Y)) ** 2 + (4 * np.cos(5 * X) * np.sin(4 * Y)) ** 2)
    np.testing.assert_allclose(pressure["momentum"], gradient / (U0 ** 2 * k), rtol=1e-9, atol=1e-12)
    assert pressure["continuity"].max() < 1e-12
    assert set(ff._vanishing(info["maps"], "continuity")) == {"pressure", "temporal_mismatch"}


def test_field_case_is_the_median_u0_of_k_1():
    cases = [{"case_id": f"case_{i}", "k": k, "U0": u} for i, (k, u) in
             enumerate([(1, 0.9), (1, 1.5), (2, 1.0), (1, 1.2), (1, 0.6)])]
    assert ff.field_case(cases)["case_id"] == "case_0"   # k = 1 by U0: 0.6, 0.9, 1.2, 1.5; lower middle 0.9


# --- The natural errors, in the terms of the benchmark ---

def test_natural_row_counts_by_hand():
    table = make_natural()
    row = ff._natural_row(table, stage_row(table), S1, THRESHOLDS[S1], "all", n_boot=20, seed=0)
    # Inaccurate: 3, one missed (case_40's Adam checkpoint at 0.9); accurate: 5, two flagged
    assert (row["n_positive"], row["n_negative"], row["n_missed"], row["n_false_alarms"]) == (3, 5, 1, 2)
    assert row["recall"] == pytest.approx(2 / 3) and row["fpr"] == pytest.approx(2 / 5)
    # Of the 15 (inaccurate, accurate) pairs, 13 have the inaccurate checkpoint scored higher
    assert row["auc"] == pytest.approx(13 / 15)
    assert row["auc_low"] <= row["auc"] <= row["auc_high"] and row["auc_n_boot"] == 20
    assert row["fpr_range_method"] == "case_bootstrap"


@pytest.mark.parametrize("column, value", [("n_false_alarms", 3), ("n_missed", 0), ("recall", 0.5), ("tau", 2.0)])
def test_natural_row_refuses_a_39_row_that_does_not_follow(column, value):
    table = make_natural()
    theirs = stage_row(table)
    theirs[column] = value
    with pytest.raises(ValueError, match="table_stage_by_k"):
        ff._natural_row(table, theirs, S1, THRESHOLDS[S1], "all", n_boot=10, seed=0)


def test_save_figure_titles_only_the_png(tmp_path):
    for with_legend in (False, True):
        fig, ax = plt.subplots(layout="constrained")
        line, = ax.plot([0, 1], [0, 1], label="a line")
        legend = fig.legend(handles=[line], loc="outside upper center") if with_legend else None
        titles, save = [], fig.savefig

        def recording(path, **kwargs):
            title = legend.get_title().get_text() if with_legend else "".join(t.get_text() for t in fig.texts)
            titles.append((Path(path).suffix, title))
            return save(path, **kwargs)

        fig.savefig = recording
        ff.save_figure(fig, tmp_path, "fig_roc", top_legend=legend)
        plt.close(fig)
        assert titles == [(".pdf", ""), (".png", ff.PNG_TITLES["fig_roc"])], with_legend


# --- The committed results ---

def test_committed_inputs_agree(committed):
    fields = committed["fields"]
    assert len(fields) == 930 and fields["case_id"].nunique() == 30
    assert len(committed["old"]) == 30
    assert len(committed["natural"]) == 30 * 7
    assert committed["tables"]["table_cases"]["case_id"].tolist() == [c["case_id"] for c in read_cases(ff.DEFAULT_METADATA)]
    assert len(committed["tables"]["table_recall_by_type_eps_k"]) == 5 * 3 * 5 * 6
    assert len(committed["sizes"]) == 30 * 5 * 6 and len(committed["tables"]["table_damage_size"]) == 5 * 6
    assert len(committed["tables"]["table_term_auc"]) == 4 * (1 + 5 + 1)   # Terms x (all, each type, natural)


def test_term_auc_agrees_with_the_scores(committed):
    """Score1 is the momentum term alone, so their AUCs are the same; a damage that leaves a term alone gives 0.5."""
    terms = committed["tables"]["table_term_auc"].set_index(["term", "experiment", "group"])
    overall = committed["metrics"]["overall"].set_index("score")
    assert terms.loc[("mom_bar", ff.SYNTHETIC, ff.ALL), "auc"] == pytest.approx(overall.loc[S1, "auc"], abs=1e-12)
    summary = committed["tables"]["table_synthetic_vs_natural"].set_index(["experiment", "score", "k"])
    assert terms.loc[("mom_bar", ff.NATURAL, ff.ALL), "auc"] == pytest.approx(
        summary.loc[(ff.NATURAL, S1, ff.ALL), "auc"], abs=1e-12)
    unchanged = terms[terms["unchanged"]]
    assert set(unchanged.index) == {("div_bar", ff.SYNTHETIC, "pressure"), ("E_bar", ff.SYNTHETIC, "pressure")}
    assert (unchanged["auc"] == 0.5).all()


def test_both_experiments_in_the_same_terms(committed):
    summary = committed["tables"]["table_synthetic_vs_natural"]
    assert list(summary.columns) == list(ff.SUMMARY_COLUMNS)
    assert len(summary) == 2 * 5 * 4   # Experiment x score x (all k, k = 1, 2, 3)
    pooled = summary[summary["k"] == ff.ALL].set_index(["experiment", "score"])
    assert (pooled.loc[ff.SYNTHETIC, ["n_positive", "n_negative"]] == [900, 30]).all().all()
    assert (pooled.loc[ff.NATURAL, ["n_positive", "n_negative", "n_missed"]] == [142, 68, 0]).all().all()
    by_k = summary[summary["k"] != ff.ALL].groupby(["experiment", "score"])[
        ["n_positive", "n_negative", "n_missed", "n_false_alarms"]].sum()
    for column in by_k.columns:
        assert (by_k[column] == pooled.loc[by_k.index, column]).all(), column


def test_synthetic_rows_are_35s(committed):
    summary = committed["tables"]["table_synthetic_vs_natural"]
    rows = summary[(summary["experiment"] == ff.SYNTHETIC) & (summary["k"] == ff.ALL)].set_index("score")
    overall = committed["metrics"]["overall"].set_index("score").loc[rows.index]
    for column in ("auc", "auc_low", "auc_high", "recall", "recall_low", "recall_high", "fpr", "fpr_low", "fpr_high",
                   "n_false_alarms", "tau"):
        assert (rows[column] == overall[column]).all(), column
    assert (rows["n_positive"] == overall["n_damaged"]).all()


def test_synthetic_row_refuses_another_tau_or_range(committed):
    row = committed["metrics"]["overall"].iloc[0].copy()
    score, tau = row["score"], row["tau"]
    ff._synthetic_row(row, score, "all", tau)   # As committed: accepted
    with pytest.raises(ValueError, match="tau"):
        ff._synthetic_row(row, score, "all", tau * 1.01)
    row["fpr_high"] += 0.01
    with pytest.raises(ValueError, match="exact range"):
        ff._synthetic_row(row, score, "all", tau)


def test_read_natural_refuses_a_changed_flag(tmp_path, committed):
    folder = tmp_path / "natural_errors"
    folder.mkdir()
    path = folder / ff.NATURAL_SCORES_CSV
    table = pd.read_csv(ff.DEFAULT_NATURAL_DIR / ff.NATURAL_SCORES_CSV, float_precision="round_trip")
    table.loc[3, f"flagged_{S2}"] = not table.loc[3, f"flagged_{S2}"]
    table.to_csv(path, index=False, lineterminator="\n")
    _, errors, _ = ff.read_case_inputs(ff.DEFAULT_METADATA, ff.DEFAULT_ERRORS_CSV, ff.DEFAULT_LOSS_SUMMARY)
    with pytest.raises(ValueError, match="raw components"):
        ff.read_natural(folder, committed["calib"], errors, committed["cases"], committed["fields"])


@pytest.mark.parametrize("change", ["clean_field_differs", "case_not_in_case_file"])
def test_read_natural_refuses_checkpoints_that_are_not_the_benchmarks(committed, change):
    """The final models must be the benchmark's clean fields, and the checkpoints those of the case file's cases."""
    _, errors, _ = ff.read_case_inputs(ff.DEFAULT_METADATA, ff.DEFAULT_ERRORS_CSV, ff.DEFAULT_LOSS_SUMMARY)
    fields, cases = committed["fields"].copy(), list(committed["cases"])
    if change == "clean_field_differs":
        fields.loc[fields.index[fields["label"] == "clean"][0], "mom"] *= 1.001
        match = "clean fields"
    else:
        cases = cases[1:]
        match = "every case of the case file"
    with pytest.raises(ValueError, match=match):
        ff.read_natural(ff.DEFAULT_NATURAL_DIR, committed["calib"], errors, cases, fields)


def test_recall_by_k_sums_to_the_benchmark(committed):
    """Summed over k, the type x strength x k counts are #35's type x strength counts."""
    by_k = committed["tables"]["table_recall_by_type_eps_k"]
    summed = by_k.groupby(["score", "perturbation", "epsilon"])[["n_damaged_flagged", "n_damaged",
                                                                 "n_damaged_new_flagged", "n_damaged_new"]].sum()
    type_eps = committed["metrics"]["by_type_eps"].set_index(["score", "perturbation", "epsilon"])
    for column in summed.columns:
        assert (summed[column] == type_eps.loc[summed.index, column]).all(), column


def copy_benchmark(tmp_path) -> Path:
    folder = tmp_path / "benchmark"
    shutil.copytree(ff.DEFAULT_BENCHMARK_DIR, folder)
    return folder


def test_read_fields_refuses_a_changed_flag(tmp_path):
    folder = copy_benchmark(tmp_path)
    path = folder / ff.FIELDS_CSV
    fields = pd.read_csv(path, float_precision="round_trip")
    fields.loc[5, f"flagged_{S3}"] = not fields.loc[5, f"flagged_{S3}"]
    fields.to_csv(path, index=False, lineterminator="\n")
    calib = frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON)
    with pytest.raises(ValueError, match="frozen calibration"):
        ff.read_fields(folder, calib, ff.DEFAULT_METADATA)


def test_read_fields_refuses_a_changed_preflag(tmp_path):
    folder = copy_benchmark(tmp_path)
    path = folder / ff.FIELDS_CSV
    fields = pd.read_csv(path, float_precision="round_trip")
    fields.loc[5, f"preflagged_{S1}"] = not fields.loc[5, f"preflagged_{S1}"]
    fields.to_csv(path, index=False, lineterminator="\n")
    calib = frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON)
    with pytest.raises(ValueError, match="preflagged"):
        ff.read_fields(folder, calib, ff.DEFAULT_METADATA)


@pytest.mark.parametrize("name, column, value", [("by_type_eps", "n_damaged_flagged", 1),
                                                 ("by_k", "n_false_alarms", 1), ("overall", "auc", 1e-9)])
def test_check_metrics_refuses_a_table_that_disagrees(committed, name, column, value):
    metrics = {key: table.copy() for key, table in committed["metrics"].items()}
    metrics[name].loc[3, column] += value
    with pytest.raises(ValueError, match=f"metrics_{name}"):
        ff.check_metrics(committed["fields"], metrics)


def test_read_old_clean_fields_refuses_another_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(Path(ff.DEFAULT_SOURCE_CSV).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        ff.read_old_clean_fields(source, frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON))


def test_old_clean_fields_give_the_calibration_back(committed):
    """tau is the 95th percentile of the old clean scores (PROTOCOL.md section 5), each with 2 of 30 above it."""
    thresholds = committed["calib"]["thresholds"]
    for score, tau in thresholds.items():
        assert np.percentile(committed["old"][score], 95) == pytest.approx(tau, rel=1e-12)
        assert (committed["old"][score] > tau).sum() == 2


def test_captions_explain_what_36_asks(committed):
    captions = committed["captions"]
    # An AUC range above its own estimate, and why
    assert "Score4 for $k = 1$" in captions["fig_roc"] and "0.927 [0.928, 0.941]" in captions["fig_roc"]
    assert "draws a case twice" in captions["fig_roc"]
    # A range with no width, and why
    assert "0.733 [0.733, 0.733]" in captions["fig_recall_vs_eps_by_k"]
    assert "22 of its 30 damaged fields" in captions["fig_recall_vs_eps_by_k"]
    # The false alarms by k, and the energy term
    text = captions["fig_clean_scores_vs_tau"]
    for case_id in ("case 46", "case 48", "case 53", "case 34", "case 45", "case 36"):
        assert case_id in text
    assert "$U_0^4$" in text and "$-0.80$" in text


def test_captions_of_the_two_experiments(committed):
    captions = committed["captions"]
    natural = captions["fig_natural_score_vs_error"]
    assert "Every score flags all 142 inaccurate checkpoints" in natural
    assert "the 38 Adam checkpoints with errors from 2.96 to 10\\%" in natural
    assert "between 0.73\\% (the least accurate final model, case 58) and 2.96\\%" in natural
    shared = captions["fig_score_vs_velocity_error"]
    assert "the rise of its score over its own clean field" in shared
    assert "Not shown: the pressure damage, which changes only the pressure" in shared
    assert "(32, 10, 29, 22 and 19 for Score1 to Score5" in shared
    assert "the median rise is 0.79, 15, 6.1, 6.3 and 11 $\\tau$" in shared
    summary = captions["table_synthetic_vs_natural"]
    assert "the 142 inaccurate ones" in summary
    assert "over the 4 strengths from $\\varepsilon = 0.01$ on it is 0.967--1.000" in summary
    assert "partly built in" in summary
    assert "case 34 ($k = 1$" in captions["fig_damage_fields"]
    assert "The pressure and temporal mismatch damages keep the flow divergence-free" in captions["fig_damage_fields"]
    assert "fitted slope is 6.1" in captions["fig_energy_term_vs_u0"]
    assert "above every old case of its $k$" in captions["fig_energy_term_vs_u0"]
    assert "0.420 [0.376, 0.465]" in captions["table_term_auc"]
    assert "pressure damage: div and E" in captions["table_term_auc"]
    assert "from a median of 0.60 (Adam, 100 steps) to 1.31" in captions["table_term_auc"]
    assert "0.23--1.61\\%" in captions["table_cases"] and "pressure errors 0.36--1.22\\%" in captions["table_cases"]
    assert "stops growing from $\\varepsilon = 0.03$" in captions["table_damage_size"]


def test_captions_are_plain_latex(committed):
    """
    No '#', and no '_' or '%' outside math unless escaped (they break LaTeX); no Greek
    letters outside math; no repository files or issue numbers, since the captions go into the paper.
    """
    for name, caption in committed["captions"].items():
        assert caption.count("$") % 2 == 0, name
        text = "".join(caption.split("$")[0::2])   # The caption outside math
        text = re.sub(r"\\ref\{[^}]*\}", "", text)  # Labels may hold underscores
        for bad in ("#", "PROTOCOL", ".md", "τ", "ε", "\\_", "95\\% range"):   # No code names; CIs by their name
            assert bad not in caption, (name, bad)
        assert "_" not in text.replace("\\_", ""), name
        assert "%" not in text.replace("\\%", ""), name


def test_run_writes_everything_and_reruns_identically(tmp_path):
    _, written = ff.run(output_dir=tmp_path / "a")
    names = sorted(path.name for path in written)
    expected = sorted([f"{stem}.{ext}" for stem in ff.FIGURES for ext in ("pdf", "png", "tex")]
                      + [f"{stem}.{ext}" for stem in ff.TABLES for ext in ("csv", "tex")])
    assert names == expected
    ff.run(output_dir=tmp_path / "b")
    for name in names:
        first, second = (tmp_path / "a" / name).read_bytes(), (tmp_path / "b" / name).read_bytes()
        assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest(), name


def test_committed_tables_and_captions_are_the_scripts(committed):
    """The committed .csv read back as the tables built now; the committed .tex are what the script writes."""
    tables, captions = committed["tables"], committed["captions"]
    for stem in ff.TABLES:
        saved = pd.read_csv(OUTPUT_DIR / f"{stem}.csv", float_precision="round_trip")
        built = tables[stem]
        # A count with missing values (Int64) reads back as floats with NaN
        built = built.astype({column: "float64" for column, dtype in built.dtypes.items() if str(dtype) == "Int64"})
        # Clopper-Pearson ends can differ in the last digit between operating systems (scipy)
        pd.testing.assert_frame_equal(saved, built, check_dtype=False, rtol=1e-12, obj=stem)
    expected = {
        **{f"{stem}.tex": ff.table_latex(stem, tables[stem], captions[stem]) for stem in ff.TABLES},
        **{f"{stem}.tex": ff.figure_tex(stem, captions[stem], wide=stem not in ff.NARROW_FIGURES)
           for stem in ff.FIGURES},
    }
    for name, text in expected.items():
        assert (OUTPUT_DIR / name).read_text(encoding="utf-8") == text, name
        assert "\\_" not in text, f"{name} names something by its code name"


def test_committed_figures_exist():
    for stem in ff.FIGURES:
        for ext in ("pdf", "png"):
            assert (OUTPUT_DIR / f"{stem}.{ext}").stat().st_size > 10_000, f"{stem}.{ext}"


def test_path_flags_default_to_the_repo_paths():
    args = ff.parse_args([])
    assert Path(args.benchmark_dir) == REPO.resolve() / "results" / "benchmark"
    assert Path(args.natural_dir) == REPO.resolve() / "results" / "natural_errors"
    assert Path(args.errors) == REPO.resolve() / "results" / "accuracy" / "reference_errors_confirmation.csv"
    assert Path(args.metadata) == REPO.resolve() / "cases" / "confirmation_cases.json"
    assert Path(args.loss_summary) == REPO.resolve() / "models" / "confirmation" / "loss_summary.csv"
    assert Path(args.calibration) == REPO.resolve() / "calibration" / "frozen_calibration.json"
    assert Path(args.source) == REPO.resolve() / "calibration" / "source_phs_components_raw.csv"
    assert Path(args.output_dir) == REPO.resolve() / "results" / "figures"
